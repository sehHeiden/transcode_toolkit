from pathlib import Path

from transcode_toolkit.chain import Chain


class TestChainDiscover:
    def test_finds_files_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp3").touch()
        (tmp_path / "b.flac").touch()
        (tmp_path / "c.txt").touch()
        result = Chain(tmp_path).discover(extensions=[".mp3", ".flac"])
        assert len(result._files) == 2

    def test_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.mp3").touch()
        (tmp_path / "b.mp3").touch()
        result = Chain(tmp_path).discover(extensions=[".mp3"])
        assert len(result._files) == 2


class TestChainFilter:
    def test_filters_by_predicate(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp3").write_bytes(b"\x00" * 100)
        (tmp_path / "b.mp3").write_bytes(b"\x00" * 10)
        result = Chain(tmp_path).discover(extensions=[".mp3"]).filter(lambda f: f.stat().st_size > 50)
        assert len(result._files) == 1
