"""Multi-validator quorum rehearsal.

Three independent ValidatorService instances share one mock chain and process
the same challenger reveals, each with its own state directory and its own
private task view. This is the local rehearsal of the network's core trust
claims:

* public halves are computed bit-identically by every validator (the backend is
  scripted, so per-task scores are fixed and only the protocol varies — this
  pins the derivation from scores to verdict, which is deterministic, not the
  GPU engine, which is not),
* coronation fires for everyone at the same theta-crossing block,
* a validator whose private pool dissents still follows the quorum king,
* a dead validator neither blocks coronation nor corrupts anyone's state,
* accept-stake below theta never crowns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from epago import constants
from epago.chain.client import MockChainClient, NeuronView
from epago.chain.mailbox import submission_prefix
from epago.config import load_config
from epago.core.reveal import build_reveal, parse_verdict
from epago.core.stats import adaptive_delta, bootstrap_lcb, bootstrap_seed, paired_half
from epago.core.types import DuelOutcome, ModelRef, VerdictDecision
from epago.validator.service import Deps, ValidatorService
from epago.validator.state import ValidatorState
from pathlib import Path

TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"

ROUND_AUTHORITY_HK = "round-authority"

N_TASKS = 200
VALIDATORS = (
    ("val-a", "ck-val-a", 40.0),
    ("val-b", "ck-val-b", 35.0),
    ("val-c", "ck-val-c", 25.0),
)


# --------------------------------------------------------------------- fabric


@dataclass
class ChainFabric:
    """One logical chain observed by several validator identities.

    All MockChainClient instances share the same reveal/status/neuron stores;
    only the publishing identity differs. ``advance`` keeps their block
    heights in lockstep the way one real chain would.
    """

    clients: dict[str, MockChainClient] = field(default_factory=dict)

    @classmethod
    def create(cls, hotkeys: list[str]) -> "ChainFabric":
        fabric = cls()
        base = MockChainClient(identity_hotkey=hotkeys[0])
        fabric.clients[hotkeys[0]] = base
        for hk in hotkeys[1:]:
            fabric.clients[hk] = MockChainClient(
                block=base.block,
                _neurons=base._neurons,
                _reveals=base._reveals,
                _status=base._status,
                identity_hotkey=hk,
            )
        return fabric

    @property
    def any(self) -> MockChainClient:
        return next(iter(self.clients.values()))

    def advance(self, blocks: int = 1) -> None:
        for c in self.clients.values():
            c.block += blocks


def _solves(skill: float, index: int) -> bool:
    return ((index * 37) % 96) / 96.0 < skill


def make_network(tmp_path, *, private_view: dict[str, float] | None = None):
    """Three validators over one fabric.

    ``private_view[hotkey]`` is the challenger's skill *as seen by that
    validator's private pool* (defaults to the public challenger skill, i.e.
    every private pool agrees with the public improvement).
    """
    cfg = load_config(TEMPLATE_TOML)
    # Every box needs the same round authority: a round is chain state, so all
    # three must agree on which competition is running.
    cfg = replace(cfg, chain=replace(cfg.chain, round_authority_hotkey=ROUND_AUTHORITY_HK))
    fabric = ChainFabric.create([hk for hk, _, _ in VALIDATORS])
    chain0 = fabric.any
    chain0.add_neuron(NeuronView(uid=0, hotkey="burn-hk", coldkey="burn-ck", stake=0.0, validator_permit=False))
    for uid, (hk, ck, stake) in enumerate(VALIDATORS, start=1):
        chain0.add_neuron(NeuronView(uid=uid, hotkey=hk, coldkey=ck, stake=stake, validator_permit=True))

    skills: dict[str, float] = {cfg.seed.seed_digest: 0.5}  # digest -> public skill
    dirs: dict[str, object] = {}
    def _write_model(d, payload: bytes) -> None:
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"architectures": ["EpagoTest"]}))
        (d / "model.safetensors").write_bytes(payload.ljust(1024, b"\0"))

    genesis_dir = tmp_path / "models" / "genesis"
    _write_model(genesis_dir, b"genesis")
    dirs[cfg.seed.seed_digest] = genesis_dir
    dir_to_digest: dict[str, str] = {str(genesis_dir): cfg.seed.seed_digest}

    def mint_model(name: str, skill: float) -> str:
        digest = "sha256:" + hashlib.sha256(name.encode()).hexdigest()
        d = tmp_path / "models" / name
        _write_model(d, name.encode())
        skills[digest] = skill
        dirs[digest] = d
        dir_to_digest[str(d)] = digest
        return digest

    services: dict[str, SimpleNamespace] = {}
    for uid, (hk, _ck, _stake) in enumerate(VALIDATORS, start=1):
        v = _make_validator(
            tmp_path, cfg, fabric.clients[hk], hk, skills, dirs, dir_to_digest,
            private_skill=(private_view or {}).get(hk),
        )
        # Mainnet-equivalent emission config (the testnet phase-gate zeros):
        # emissions are live from the first coronation, never burn-phase.
        v.state.genesis_block = -constants.PHASE_B_MIN_BLOCKS
        v.state.clean_duels = constants.PHASE_B_MIN_CLEAN_DUELS
        v.state.organic_dethrones = constants.PHASE_B_MIN_DETHRONES
        services[hk] = v

    return SimpleNamespace(
        cfg=cfg, fabric=fabric, services=services, mint_model=mint_model, skills=skills
    )


def _make_validator(tmp_path, cfg, chain, hotkey, skills, dirs, dir_to_digest, private_skill):
    state = ValidatorState.load(tmp_path / f"state-{hotkey}")

    def materialize(ref: ModelRef, cache_dir):
        return dirs[ref.digest]

    def generate_tasks(*, seed, release, corpus, n, king_probe):
        # Pure function of the seed: every validator mints the identical list.
        return [f"pub-{seed & 0xFFFFFFFF:08x}-{i:03d}" for i in range(N_TASKS)]

    def task_ids_digest(tasks):
        return "sha256:" + hashlib.sha256("|".join(map(str, tasks)).encode()).hexdigest()

    def run_duel(spec, env, backend_factory, llm_judge):
        king_skill = skills[dir_to_digest[str(spec.king_dir)]]
        chall_digest = dir_to_digest[str(spec.challenger_dir)]
        chall_skill = skills[chall_digest]
        pub = paired_half(
            [_solves(king_skill, i) for i in range(len(spec.public_tasks))],
            [_solves(chall_skill, i) for i in range(len(spec.public_tasks))],
        )
        # Private view: this validator's own pool may see a different picture.
        chall_priv = chall_skill if private_skill is None else private_skill
        priv = paired_half(
            [_solves(king_skill, i) for i in range(len(spec.private_tasks))],
            [_solves(chall_priv, i) for i in range(len(spec.private_tasks))],
        )
        seed = bootstrap_seed(spec.block_hash_at_reveal, spec.author_hotkey)
        lcb = bootstrap_lcb(pub.diffs, seed)
        delta = adaptive_delta(spec.king_acc_ema, spec.noise_floor)
        return DuelOutcome(
            public=pub,
            private=priv,
            lcb_pub=lcb,
            delta=delta,
            accepted=lcb > delta and priv.mu_hat > 0.0,
            boot_seed_hex=f"{seed:016x}",
            public_seed_hex="00" * 8,
            public_task_results=tuple(
                (str(t), d) for t, d in zip(spec.public_tasks, pub.diffs)
            ),
        )

    class Pool:
        epoch = 1
        digest = "sha256:" + hashlib.sha256(hotkey.encode()).hexdigest()

        def sample(self, n, seed):
            return [f"prv-{hotkey}-{seed % 997}-{i}" for i in range(64)]

        def rotation_due(self, block):
            return False

        def rotate(self, block):
            return None

    deps = Deps(
        chain=chain,
        cfg=cfg,
        state=state,
        corpus=None,
        env=None,
        backend_factory=None,
        run_duel=run_duel,
        run_calibration_duel=lambda *a, **k: 0.0,
        run_probes=lambda *a: [],
        generate_tasks=generate_tasks,
        task_ids_digest=task_ids_digest,
        private_pool=Pool(),
        wallet_hotkey=hotkey,
        clock=chain.current_block,
        materialize=materialize,
        cache_dir=tmp_path / f"cache-{hotkey}",
    )
    return SimpleNamespace(service=ValidatorService(deps), state=state, chain=chain)


def open_round(net, round_no: int | None = None) -> int:
    """Publish one `er1` on the shared fabric; every validator sees the same round."""
    from epago.core.reveal import build_round_start

    net.round_no = round_no if round_no is not None else getattr(net, "round_no", 0) + 1
    net.fabric.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
    net.fabric.any.inject_reveal(ROUND_AUTHORITY_HK, build_round_start(net.round_no))
    net.fabric.advance(1)
    return net.round_no


def settle(net, ticks: int = 3, round_no: int | None = None) -> None:
    for v in net.services.values():   # intake, so the field exists at round open
        v.service.tick()
    open_round(net, round_no)
    for _ in range(ticks):
        for v in net.services.values():
            v.service.tick()
        net.fabric.advance(constants.VERDICT_REVEAL_BLOCKS + 1)
    net.fabric.advance(constants.WEIGHT_INTERVAL_BLOCKS)
    for v in net.services.values():
        v.service.tick()


def reveal_challenger(net, name: str, skill: float, hotkey="miner-1", coldkey="5Miner001xxxx"):
    digest = net.mint_model(name, skill)
    chain0 = net.fabric.any
    uid = len(chain0.neurons())
    chain0.add_neuron(
        NeuronView(uid=uid, hotkey=hotkey, coldkey=coldkey, stake=1.0, validator_permit=False)
    )
    # A `sha256:` ref is a private upload, so it must sit in its author's own
    # prefix — the same shape a real submission has.
    ref = ModelRef(repo=f"{submission_prefix(hotkey)}EPAGO-DR-4B", digest=digest)
    king_digest = net.cfg.seed.seed_digest
    chain0.inject_reveal(hotkey, build_reveal(king_digest, ref))
    net.fabric.advance(1)
    return digest, uid


def verdicts_by_validator(net, digest: str):
    out = {}
    for rp in net.fabric.any.read_revealed_payloads():
        if rp.payload.startswith("ev3|"):
            v = parse_verdict(rp.payload, validator_hotkey=rp.hotkey, block=rp.block)
            if v.challenger_digest == digest:
                out[rp.hotkey] = v
    return out


# ---------------------------------------------------------------------- tests


def test_three_validators_converge_and_crown(tmp_path):
    net = make_network(tmp_path)
    digest, miner_uid = reveal_challenger(net, "improver", skill=0.62)
    settle(net)

    verdicts = verdicts_by_validator(net, digest)
    assert set(verdicts) == {"val-a", "val-b", "val-c"}
    # Public halves are bit-identical: same lcb to the microunit, same audit
    # inputs, ACCEPT from every validator.
    lcbs = {v.lcb_pub_e6 for v in verdicts.values()}
    assert len(lcbs) == 1
    assert all(v.decision is VerdictDecision.ACCEPT for v in verdicts.values())

    kings = {hk: v.state.king.ref.digest for hk, v in net.services.items()}
    assert set(kings.values()) == {digest}
    crowned = {hk: v.state.king.crowned_block for hk, v in net.services.items()}
    assert len(set(crowned.values())) == 1  # same theta-crossing block everywhere

    # Every validator's weight vector pays the same new king.
    for hk, v in net.services.items():
        weights = net.fabric.clients[hk].last_weights
        assert weights, f"{hk} never set weights"
        assert weights[miner_uid] == max(weights.values())


def test_dissenting_private_pool_still_follows_quorum(tmp_path):
    # val-c's private pool sees the challenger as no better than the king, so
    # it votes REJECT — but 75% of stake accepts, and val-c must follow.
    net = make_network(tmp_path, private_view={"val-c": 0.5})
    digest, _ = reveal_challenger(net, "improver", skill=0.62)
    settle(net)

    verdicts = verdicts_by_validator(net, digest)
    assert verdicts["val-a"].decision is VerdictDecision.ACCEPT
    assert verdicts["val-b"].decision is VerdictDecision.ACCEPT
    assert verdicts["val-c"].decision is VerdictDecision.REJECT  # dissent on record

    for hk, v in net.services.items():
        assert v.state.king.ref.digest == digest, f"{hk} did not follow quorum"


def test_dead_validator_does_not_stall_coronation(tmp_path):
    net = make_network(tmp_path)
    digest, _ = reveal_challenger(net, "improver", skill=0.62)

    live = {hk: v for hk, v in net.services.items() if hk != "val-c"}
    for v in live.values():
        v.service.tick()          # intake before the round opens
    open_round(net)
    for _ in range(3):
        for v in live.values():
            v.service.tick()
        net.fabric.advance(constants.VERDICT_REVEAL_BLOCKS + 1)
    net.fabric.advance(constants.WEIGHT_INTERVAL_BLOCKS)
    for v in live.values():
        v.service.tick()

    # val-c never posted a verdict, so it is not an active evaluator; the
    # other 75 stake units carry quorum on their own.
    for hk, v in live.items():
        assert v.state.king.ref.digest == digest, f"{hk} stalled without val-c"
    # The dead validator never ticked: it still has no king (or genesis) and,
    # critically, its absence changed nothing for the others.
    dead_king = net.services["val-c"].state.king
    assert dead_king is None or dead_king.ref.digest == net.cfg.seed.seed_digest


def test_minority_accept_does_not_crown(tmp_path):
    # Only val-c (25% of stake) sees a private improvement; a and b reject.
    # 25 < theta * 100: nobody crowns, the candidate stays pending until the
    # timeout lapses it.
    net = make_network(
        tmp_path, private_view={"val-a": 0.5, "val-b": 0.5}
    )
    digest, _ = reveal_challenger(net, "improver", skill=0.62)
    settle(net)

    verdicts = verdicts_by_validator(net, digest)
    decisions = {hk: v.decision for hk, v in verdicts.items()}
    assert decisions["val-c"] is VerdictDecision.ACCEPT
    assert decisions["val-a"] is VerdictDecision.REJECT
    assert decisions["val-b"] is VerdictDecision.REJECT

    for hk, v in net.services.items():
        assert v.state.king.ref.digest == net.cfg.seed.seed_digest, f"{hk} crowned below theta"


def test_audit_records_replay_identically_across_validators(tmp_path):
    net = make_network(tmp_path)
    digest, _ = reveal_challenger(net, "improver", skill=0.62)
    settle(net)

    # Each validator's audit record binds the same public inputs: identical
    # public seed, task-set digest, per-task diffs, lcb, and delta.
    fingerprints = set()
    for hk in net.services:
        log = tmp_path / f"state-{hk}" / "audit" / "audit.jsonl"
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        rec = next(r for r in records if r["challenger_digest"] == digest)
        fingerprints.add(
            (
                rec["public_seed"],
                rec["public_task_ids_digest"],
                rec["lcb_pub"],
                rec["delta_threshold"],
                json.dumps(rec["extra"]["public_diffs"]),
            )
        )
    assert len(fingerprints) == 1
