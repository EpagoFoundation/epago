"""Public mirroring of validator artifacts.

Everything a validator publishes locally — rotated private pools, the audit
log, delay-released public task sets, the dashboard export, and king mirror
manifests — is synced to one public Hugging Face *dataset* repo per validator,
so miners and external auditors never need access to a validator's disk.
Crowned king snapshots are additionally mirrored to HF *model* repos so
late-joining validators can materialize the king even after the original
miner deletes their upstream repo.
"""

from epago.publishing.publisher import (
    MirrorResolver,
    PublishReport,
    StatePublisher,
    publish_king_mirror,
    update_mirror_manifest,
)

__all__ = [
    "MirrorResolver",
    "PublishReport",
    "StatePublisher",
    "publish_king_mirror",
    "update_mirror_manifest",
]
