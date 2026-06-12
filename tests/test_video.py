from transcode_toolkit.video import _PRESETS


class TestDefaultPresets:
    def test_generates_presets(self):
        assert len(_PRESETS) == 12

    def test_all_have_required_keys(self):
        for p in _PRESETS:
            assert "codec" in p
            assert "crf" in p
            assert "speed" in p
            assert "label" in p

    def test_contains_x265(self):
        codecs = {p["codec"] for p in _PRESETS}
        assert "libx265" in codecs

    def test_contains_nvenc(self):
        codecs = {p["codec"] for p in _PRESETS}
        assert "hevc_nvenc" in codecs
