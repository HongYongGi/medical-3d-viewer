"""Patient study database for tracking analysis history."""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path


DB_PATH = Path("data/studies.db")


@dataclass
class StudyRecord:
    """A single analysis study record."""
    id: int | None = None
    patient_id: str = ""
    patient_name: str = ""
    study_date: str = ""
    modality: str = "CT"
    description: str = ""
    input_path: str = ""
    seg_paths: str = ""  # JSON dict: model_id -> path
    model_used: str = ""
    pipeline_used: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""
    tags: str = ""  # comma-separated
    thumbnail_path: str = ""

    def seg_paths_dict(self) -> dict[str, str]:
        if not self.seg_paths:
            return {}
        try:
            return json.loads(self.seg_paths)
        except json.JSONDecodeError:
            return {}

    def tags_list(self) -> list[str]:
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]


class StudyDatabase:
    """SQLite database for managing patient studies."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT DEFAULT '',
                    patient_name TEXT DEFAULT '',
                    study_date TEXT DEFAULT '',
                    modality TEXT DEFAULT 'CT',
                    description TEXT DEFAULT '',
                    input_path TEXT NOT NULL,
                    seg_paths TEXT DEFAULT '{}',
                    model_used TEXT DEFAULT '',
                    pipeline_used TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    thumbnail_path TEXT DEFAULT ''
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON studies(patient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON studies(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON studies(created_at)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_study(self, record: StudyRecord) -> StudyRecord:
        now = datetime.now().isoformat()
        record.created_at = now
        record.updated_at = now
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO studies (
                    patient_id, patient_name, study_date, modality, description,
                    input_path, seg_paths, model_used, pipeline_used, status,
                    created_at, updated_at, notes, tags, thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.patient_id, record.patient_name, record.study_date,
                record.modality, record.description, record.input_path,
                record.seg_paths, record.model_used, record.pipeline_used,
                record.status, record.created_at, record.updated_at,
                record.notes, record.tags, record.thumbnail_path,
            ))
            record.id = cursor.lastrowid
        return record

    def update_study(self, study_id: int, **kwargs) -> StudyRecord | None:
        kwargs["updated_at"] = datetime.now().isoformat()
        set_clauses = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(study_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE studies SET {set_clauses} WHERE id = ?", values)
        return self.get_study(study_id)

    def get_study(self, study_id: int) -> StudyRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_studies(
        self,
        status: str | None = None,
        patient_id: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at DESC",
    ) -> list[StudyRecord]:
        conditions = []
        params: list = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if patient_id:
            conditions.append("patient_id = ?")
            params.append(patient_id)
        if search:
            conditions.append(
                "(patient_name LIKE ? OR patient_id LIKE ? OR description LIKE ? OR tags LIKE ?)"
            )
            like = f"%{search}%"
            params.extend([like, like, like, like])

        where = " AND ".join(conditions) if conditions else "1=1"
        allowed_orders = {
            "created_at DESC", "created_at ASC", "updated_at DESC",
            "patient_name ASC", "patient_name DESC", "study_date DESC",
        }
        if order_by not in allowed_orders:
            order_by = "created_at DESC"

        query = f"SELECT * FROM studies WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_studies(self, status: str | None = None) -> int:
        with self._conn() as conn:
            if status:
                row = conn.execute("SELECT COUNT(*) FROM studies WHERE status = ?", (status,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM studies").fetchone()
        return row[0]

    def delete_study(self, study_id: int) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM studies WHERE id = ?", (study_id,))
        return cursor.rowcount > 0

    def get_unique_patients(self) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT patient_id, patient_name,
                       COUNT(*) as study_count,
                       MAX(created_at) as last_study
                FROM studies WHERE patient_id != ''
                GROUP BY patient_id ORDER BY last_study DESC
            """).fetchall()
        return [
            {"patient_id": r["patient_id"], "patient_name": r["patient_name"],
             "study_count": r["study_count"], "last_study": r["last_study"]}
            for r in rows
        ]

    @staticmethod
    def _row_to_record(row) -> StudyRecord:
        return StudyRecord(
            id=row["id"], patient_id=row["patient_id"],
            patient_name=row["patient_name"], study_date=row["study_date"],
            modality=row["modality"], description=row["description"],
            input_path=row["input_path"], seg_paths=row["seg_paths"],
            model_used=row["model_used"], pipeline_used=row["pipeline_used"],
            status=row["status"], created_at=row["created_at"],
            updated_at=row["updated_at"], notes=row["notes"],
            tags=row["tags"], thumbnail_path=row["thumbnail_path"],
        )
