"""
Blurb store for RoboRanger.

Reads species blurbs from `corpus.db`. The actual schema stores blurbs as
JSON blobs rather than as separate columns:

    species(species_id PK, species_name, common_name, source,
            blurb_text, blurb_json)

Two columns hold the blurb data:
  - blurb_json: canonical structured form, keys defined in build_blurbs.py
  - blurb_text: pre-rendered prose fallback for display

This module reads `blurb_json` (canonical) and adapts the parsed dict to
the `Blurb` dataclass. The species table's own `common_name` column is
ignored — it's frequently empty in the wild (the build script doesn't
populate it), while `blurb_json.common_name` is reliably present.

Defensive behavior:
  - Unknown keys in blurb_json are ignored (forward compat).
  - Missing fields become None (prompt builder skips them).
  - Known typos like 'apprearance' (sic) map to 'appearance' so a single
    misspelled JSON entry doesn't lose its identification text. The right
    fix is upstream in build_blurbs.py — closing the JSON schema with
    additionalProperties: false — but defending here means a stray
    misspelling doesn't tank one species' answers in the meantime.
  - Unknown values in danger enums map to None rather than raising.
  - Failed JSON parse logs and returns None so the caller can refuse
    cleanly rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys

sys.path.insert(1, "../species_identification/llm-tuning")
from prompt_builder import Blurb, DANGER_LEVELS

log = logging.getLogger(__name__)


# Map from blurb_json key to Blurb dataclass field. Most are identity;
# the typo is handled explicitly so future drafting bugs don't silently
# drop content.
JSON_KEY_TO_FIELD: dict[str, str] = {
    "common_name":         "common_name",
    "appearance":          "appearance",
    "apprearance":         "appearance",   # observed typo from drafter
    "size":                "size",
    "habitat":             "habitat",
    "diet":                "diet",
    "behavior":            "behavior",
    "dangerous_to_humans": "dangerous_to_humans",
    "dangerous_to_pets":   "dangerous_to_pets",
    "notable":             "notable",
    "size_prose":          "size_prose",
    "appearance_prose":    "appearance_prose",
    "behavior_prose":      "behavior_prose",
    "diet_prose":          "diet_prose",
    "notable_prose":       "notable_prose",
}

# Required columns on the species table for this store to work.
REQUIRED_COLUMNS = ("species_id", "species_name", "blurb_json")


class BlurbStoreError(RuntimeError):
    """Raised when the species table is missing required columns."""


class BlurbStore:
    """
    Looks up structured blurbs by species_id.

    Usage:
        conn = open_db("corpus.db")
        store = BlurbStore(conn)
        blurb = store.get("Crotalus_oreganus_helleri")
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        table: str = "species",
    ) -> None:
        self.conn = conn
        self.table = table
        self._verify_schema()

    def _verify_schema(self) -> None:
        try:
            rows = self.conn.execute(
                f"PRAGMA table_info({self.table})"
            ).fetchall()
        except sqlite3.Error as e:
            raise BlurbStoreError(
                f"Could not inspect table {self.table!r}: {e}"
            ) from e
        if not rows:
            raise BlurbStoreError(
                f"Table {self.table!r} not found. Has the corpus been built?"
            )
        existing = {r[1] for r in rows}
        missing = [c for c in REQUIRED_COLUMNS if c not in existing]
        if missing:
            raise BlurbStoreError(
                f"Species table is missing required columns: {missing}. "
                f"Existing columns: {sorted(existing)}."
            )

    def get(self, species_id: str) -> Blurb | None:
        """Return the blurb for this species, or None if not found."""
        row = self.conn.execute(
            f"SELECT species_name, blurb_json FROM {self.table} "
            f"WHERE species_id = ?",
            (species_id,),
        ).fetchone()
        if row is None:
            return None
        binomial, blurb_json = row

        # scientific_name comes from species_name, with a fallback.
        scientific_name = (
            binomial.strip() if binomial and binomial.strip()
            else species_id.replace("_", " ")
        )

        # Parse the JSON blob. Bad JSON shouldn't crash the whole pipeline —
        # log and return a minimal Blurb with just the names so the LLM at
        # least knows what species it's talking about.
        parsed: dict = {}
        if blurb_json:
            try:
                parsed = json.loads(blurb_json)
            except json.JSONDecodeError as e:
                log.warning(
                    "Bad blurb_json for %s: %s. Returning minimal Blurb.",
                    species_id, e,
                )

        # Map JSON keys to dataclass fields.
        prose_reviewed = bool(parsed.get("prose_reviewed", False))
        data: dict[str, str | None] = {}
        for json_key, value in parsed.items():
            field = JSON_KEY_TO_FIELD.get(json_key)
            if field is None:
                continue   # unknown key, ignore (forward compat)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            # If two json keys map to the same field (e.g. typo), prefer the
            # first non-empty one. This usually doesn't happen but is safe.
            data.setdefault(field, value)

        # Defensive: drop bad enum values rather than crashing.
        for enum_field in ("dangerous_to_humans", "dangerous_to_pets"):
            v = data.get(enum_field)
            if v is not None and v not in DANGER_LEVELS:
                log.info(
                    "Dropping %s=%r for %s (not in %s)",
                    enum_field, v, species_id, DANGER_LEVELS,
                )
                data.pop(enum_field)

        # common_name fallback if missing in JSON.
        if not data.get("common_name"):
            data["common_name"] = scientific_name

        return Blurb(
            scientific_name=scientific_name,
            prose_reviewed=prose_reviewed,
            **data,
        )

    def exists(self, species_id: str) -> bool:
        row = self.conn.execute(
            f"SELECT 1 FROM {self.table} WHERE species_id = ?",
            (species_id,),
        ).fetchone()
        return row is not None