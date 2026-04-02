"""Version archive — tracks prompt lineage with rollback support.

Each prompt version is persisted as a JSON file under
``data/archive/{stage}/{version_id}.json``.  A ``current.json`` file
in each stage directory points to the currently promoted version.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "archive"


@dataclass
class VersionRecord:
    """Snapshot of one prompt version with its evaluation results."""

    version_id: str
    parent_version: str | None
    stage: str
    status: str  # "baseline" | "promoted" | "rejected"
    created_at: str
    mutation_rationale: str
    fitness_mean: float
    fitness_ci: tuple[float, float]
    per_metric: dict[str, float] = field(default_factory=dict)
    per_persona: dict[str, float] = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VersionRecord:
        # Ensure fitness_ci is a tuple
        ci = data.get("fitness_ci", (0.0, 0.0))
        data["fitness_ci"] = tuple(ci) if isinstance(ci, list) else ci
        return cls(**data)


class Archive:
    """Manages prompt version lineage on disk."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or DEFAULT_ARCHIVE_DIR

    def _stage_dir(self, stage: str) -> Path:
        d = self.base_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _version_path(self, stage: str, version_id: str) -> Path:
        return self._stage_dir(stage) / f"{version_id}.json"

    def _current_path(self, stage: str) -> Path:
        return self._stage_dir(stage) / "current.json"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_version(self, record: VersionRecord) -> Path:
        """Persist a version record to disk."""
        path = self._version_path(record.stage, record.version_id)
        path.write_text(json.dumps(record.to_dict(), indent=2))
        logger.info("Archived version %s for %s -> %s", record.version_id, record.stage, path)
        return path

    def promote(self, stage: str, version_id: str) -> None:
        """Mark a version as the current promoted version for a stage."""
        record = self.load_version(stage, version_id)
        record.status = "promoted"

        # Save updated status
        self.save_version(record)

        # Update current pointer
        pointer = {"current_version": version_id, "promoted_at": datetime.utcnow().isoformat()}
        self._current_path(stage).write_text(json.dumps(pointer, indent=2))
        logger.info("Promoted %s to current for stage %s", version_id, stage)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_version(self, stage: str, version_id: str) -> VersionRecord:
        """Load a specific version record."""
        path = self._version_path(stage, version_id)
        if not path.exists():
            raise FileNotFoundError(f"Version {version_id} not found for stage {stage}")
        data = json.loads(path.read_text())
        return VersionRecord.from_dict(data)

    def get_current(self, stage: str) -> VersionRecord | None:
        """Load the currently promoted version for a stage, or None."""
        pointer_path = self._current_path(stage)
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text())
        version_id = pointer.get("current_version")
        if not version_id:
            return None
        return self.load_version(stage, version_id)

    def list_versions(self, stage: str) -> list[VersionRecord]:
        """List all archived versions for a stage, sorted by creation time."""
        stage_dir = self._stage_dir(stage)
        records: list[VersionRecord] = []
        for path in stage_dir.glob("*.json"):
            if path.name == "current.json":
                continue
            try:
                data = json.loads(path.read_text())
                records.append(VersionRecord.from_dict(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("Skipping malformed archive file: %s", path)
        records.sort(key=lambda r: r.created_at)
        return records

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, stage: str) -> VersionRecord:
        """Roll back to the parent of the current promoted version.

        Raises:
            ValueError: If there's no current version or no parent to roll back to.
        """
        current = self.get_current(stage)
        if current is None:
            raise ValueError(f"No current version to roll back from for stage {stage}")
        if current.parent_version is None:
            raise ValueError(f"Current version {current.version_id} has no parent to roll back to")

        parent = self.load_version(stage, current.parent_version)
        self.promote(stage, parent.version_id)
        logger.info(
            "Rolled back %s from %s to %s",
            stage, current.version_id, parent.version_id,
        )
        return parent
