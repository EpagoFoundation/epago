"""Task generation: deterministic minting, QA, private pools, difficulty.

Public API of the subsystem; see module docstrings for the determinism and
transparency contracts each piece upholds.
"""

from epago.taskgen.difficulty import DifficultyController
from epago.taskgen.generator import (
    GenerationExhausted,
    KingProbe,
    generate_tasks,
    task_ids_digest,
)
from epago.taskgen.ingest import (
    HttpSource,
    IngestSource,
    LocalDirSource,
    OverlayCorpus,
    build_private_tasks,
)
from epago.taskgen.private_pool import PrivatePool
from epago.taskgen.qa import QaReport, verify_task
from epago.taskgen.templates import (
    GENERAL_VOCABULARY,
    MASK_EXEMPT_TEMPLATES,
    MEDICAL_VOCABULARY,
    RELEASE_VOCABULARY,
    RELEASES,
    VOCABULARIES,
    FindingVocabulary,
    TaskTemplate,
    base_mixture,
    content_task_id,
    finding_sentences,
    templates_for_release,
    vocabulary_for_release,
)

__all__ = [
    "DifficultyController",
    "FindingVocabulary",
    "GENERAL_VOCABULARY",
    "GenerationExhausted",
    "HttpSource",
    "IngestSource",
    "KingProbe",
    "LocalDirSource",
    "MASK_EXEMPT_TEMPLATES",
    "MEDICAL_VOCABULARY",
    "OverlayCorpus",
    "PrivatePool",
    "QaReport",
    "RELEASES",
    "RELEASE_VOCABULARY",
    "TaskTemplate",
    "VOCABULARIES",
    "base_mixture",
    "build_private_tasks",
    "content_task_id",
    "finding_sentences",
    "generate_tasks",
    "task_ids_digest",
    "templates_for_release",
    "verify_task",
    "vocabulary_for_release",
]
