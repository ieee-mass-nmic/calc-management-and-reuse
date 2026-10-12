import pytest

from peft_reuse.graph import ArtifactSpec
from peft_reuse.store import ArtifactStore, CacheConflict, CorruptArtifact, StoreBusy


@pytest.fixture
def spec():
    return ArtifactSpec.create("feature", [], {"text": "content"}, {"backend": "test"})


def test_atomic_commits_are_immutable_and_checksums_detect_corruption(tmp_path, spec):
    store = ArtifactStore(tmp_path)
    with store.writer():
        assert store.commit(spec, [1.0, 2.0]) > 0
        assert store.commit(spec, [1.0, 2.0]) == 0
        with pytest.raises(CacheConflict):
            store.commit(spec, [3.0])
    assert store.load(spec) == [1.0, 2.0]
    (store.path(spec.key) / "payload.json").write_text("[9.0]", encoding="utf-8")
    with pytest.raises(CorruptArtifact, match="checksum"):
        store.load(spec)


def test_incomplete_and_temporary_artifacts_are_never_hits(tmp_path, spec):
    store = ArtifactStore(tmp_path)
    (tmp_path / ".tmp-interrupted").mkdir()
    assert store.load(spec) is None
    target = store.path(spec.key)
    target.mkdir()
    (target / "payload.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptArtifact, match="Incomplete"):
        store.load(spec)


def test_failed_publication_leaves_no_artifact(tmp_path, spec, monkeypatch):
    store = ArtifactStore(tmp_path)

    def fail(*args):
        raise OSError("Injected rename failure")

    monkeypatch.setattr("peft_reuse.store.os.rename", fail)
    with store.writer(), pytest.raises(OSError, match="Injected"):
        store.commit(spec, [1])
    assert store.load(spec) is None
    assert not list(tmp_path.glob(".tmp-*"))


def test_store_rejects_competing_writers_and_traversal(tmp_path, spec):
    first, second = ArtifactStore(tmp_path), ArtifactStore(tmp_path)
    with first.writer(), pytest.raises(StoreBusy):
        with second.writer():
            pass
    with pytest.raises(StoreBusy):
        first.commit(spec, [1])
    with pytest.raises(ValueError):
        first.path("../../escape")
