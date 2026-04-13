"""Tests for StudyDatabase CRUD operations."""
import tempfile
from pathlib import Path

from medical_viewer.core.database import StudyDatabase, StudyRecord, ALLOWED_UPDATE_COLUMNS


def _make_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    return StudyDatabase(tmp)


def test_create_and_get():
    db = _make_db()
    record = StudyRecord(input_path="/tmp/test.nii.gz", status="pending")
    created = db.create_study(record)
    assert created.id is not None
    fetched = db.get_study(created.id)
    assert fetched.input_path == "/tmp/test.nii.gz"
    assert fetched.status == "pending"


def test_update():
    db = _make_db()
    record = db.create_study(StudyRecord(input_path="/tmp/test.nii.gz"))
    updated = db.update_study(record.id, status="completed", notes="done")
    assert updated.status == "completed"
    assert updated.notes == "done"


def test_update_rejects_invalid_column():
    db = _make_db()
    record = db.create_study(StudyRecord(input_path="/tmp/test.nii.gz"))
    try:
        db.update_study(record.id, **{"invalid_col": "value"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_list_and_search():
    db = _make_db()
    db.create_study(StudyRecord(input_path="/tmp/a.nii.gz", patient_name="홍길동", tags="TAVR"))
    db.create_study(StudyRecord(input_path="/tmp/b.nii.gz", patient_name="김철수", tags="cardiac"))

    all_studies = db.list_studies()
    assert len(all_studies) == 2

    searched = db.list_studies(search="TAVR")
    assert len(searched) == 1
    assert searched[0].patient_name == "홍길동"


def test_delete():
    db = _make_db()
    record = db.create_study(StudyRecord(input_path="/tmp/test.nii.gz"))
    assert db.delete_study(record.id)
    assert db.get_study(record.id) is None


def test_count():
    db = _make_db()
    db.create_study(StudyRecord(input_path="/tmp/a.nii.gz", status="completed"))
    db.create_study(StudyRecord(input_path="/tmp/b.nii.gz", status="failed"))
    assert db.count_studies() == 2
    assert db.count_studies("completed") == 1
    assert db.count_studies("failed") == 1


if __name__ == "__main__":
    test_create_and_get()
    test_update()
    test_update_rejects_invalid_column()
    test_list_and_search()
    test_delete()
    test_count()
    print("All database tests passed!")
