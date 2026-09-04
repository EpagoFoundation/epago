# Running Epago with Docker Compose

One command brings up the whole stack — validator (CPU), eval (GPU), dashboard,
and Watchtower for automatic updates.

## First run

```bash
cd docker
cp .env.example .env          # then edit: netuid, network, wallet, ports
docker compose up -d
```

- `docker/.env` is **gitignored** — it holds your config and secrets.
- Compose must run **from `docker/`** so `.env` is loaded for both `${VAR}`
  interpolation and the containers' `env_file`.
- The wallet directory (`EPAGO_WALLET_DIR`, default `~/.bittensor`) is mounted
  **read-only**.

Check status / logs:

```bash
docker compose ps
docker compose logs -f validator
```

The dashboard is served on `http://<host>:${EPAGO_DASHBOARD_PORT:-8080}/`.

## Images

CI (`.github/workflows/docker.yml`) publishes two images to GHCR **only when a
PR is merged to `main`**:

| Image | Built from | Runs |
|-------|-----------|------|
| `ghcr.io/epagofoundation/epago` | `Dockerfile.validator` (python-slim) | validator + dashboard |
| `ghcr.io/epagofoundation/epago-eval` | `Dockerfile.eval` (CUDA + vLLM) | eval GPU server |

Each build tags `:latest` and `:main-<sha>`. `docker compose up -d` pulls
`:latest`; to build locally instead:

```bash
docker compose build            # uses the Dockerfiles in this directory
```

## Automatic updates (Watchtower)

The `watchtower` service polls GHCR every `WATCHTOWER_INTERVAL` seconds and, when
a new image is published, pulls it and rolling-restarts the labelled containers
(`validator`, `eval`, `dashboard`). So a merge to `main` → CI build → validators
update themselves with no manual step.

**Registry access:** Watchtower must be able to pull the GHCR images. Simplest is
to make the two GHCR packages **public** (GitHub → repo → Packages → each package
→ Package settings → Change visibility → Public). To keep them private instead,
give Watchtower a credential by mounting a Docker `config.json` with a GHCR PAT:

```yaml
  watchtower:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ~/.docker/config.json:/config.json:ro
```

To exclude a service from auto-update (e.g. pin the GPU image), remove its
`com.centurylinklabs.watchtower.enable` label.

## Weekly holdout rotation (profile `auto-holdout`)

The private half of every duel is minted from papers published *after* a miner's
training cutoff. That only stays true while the feed keeps moving, so the
`holdout-rotator` service runs one full rotation every `EPAGO_HOLDOUT_INTERVAL_S`
seconds (default 604800 — one week):

```bash
# in docker/.env:  HUGGINGFACE_TOKEN=<HF token with write access to the holdout org>
docker compose --profile auto-holdout up -d
docker compose logs -f holdout-rotator
```

Each cycle harvests a fresh dated window, publishes it to a **private**, dated
dataset repo, and writes the new `[private_source]` pin into the state volume:

| File in `/state/holdout` | Contents |
|---|---|
| `last-rotation.json` | The machine-readable report: status, repo, revision, seed, kept papers, window |
| `<period>/private_source.toml` | The ready-to-paste contract block |
| `<period>/manifest.json`, `<period>/shard-*.parquet` | The build itself — seed and plan recorded, so the slice is reproducible |
| `rotations.jsonl` | One line per applied rotation; also how a restarted loop knows a period already went out |

**The container does not edit `chains/*.toml`.** A contract is consensus config —
every validator must read the same file — so the operator commits the emitted block
to the repo contract. Run the script with `--contract <path>` outside Docker to have
it rewritten in place (backup + the old pin kept in a comment).

| Env | Default | Effect |
|---|---|---|
| `HUGGINGFACE_TOKEN` | *(required)* | Write token, supplied by env only; the service refuses to start without it |
| `EPAGO_HOLDOUT_INTERVAL_S` | `604800` | Seconds between rotations |
| `EPAGO_HOLDOUT_MIN_PAPERS` | `800` | Refuse to publish a slice thinner than this — a rate-limited week must not become the live feed |
| `EPAGO_HOLDOUT_MODE` | `--apply` | Set `--dry-run` to rehearse: harvests and reports, publishes nothing |

The loop is `|| true`-wrapped, so a throttled harvest or a hub blip costs one week,
not the loop. Re-running a period that already went out (contract pin, ledger, or an
existing repo on the hub) is skipped rather than published twice, and the published
repo is verified private before a single shard is uploaded. A slice stays private
while it is live and is revealed only after it rotates out.

## Notes

- The `test` CI job (pytest + a vocabulary sweep) gates every PR; images build
  only after it passes on `main`.
- Building the `epago-eval` image needs the runner's spare disk — the workflow
  reclaims it first. On a constrained runner, build `epago-eval` on the GPU box
  with `docker compose build eval` instead.
- Keep validator and eval on the **same image generation**; Watchtower updates
  both together when CI publishes both.
