from app.multimodal.frames import FrameExtractor
from app.models.schemas import Chunk


def make_chunk(i=0):
    return Chunk(chunk_id=f"c{i}", video_id="v", video_url="u", text="t",
                 start_time=10.0 * i, end_time=10.0 * i + 9, index=i)


def test_graceful_when_tools_missing(tmp_path, monkeypatch):
    fe = FrameExtractor(tmp_path)
    monkeypatch.setattr(FrameExtractor, "available", staticmethod(lambda: False))
    n = fe.attach_frames("vid", [make_chunk(0), make_chunk(1)])
    assert n == 0    # no crash, no frames


def test_attach_frames_uses_existing_files(tmp_path, monkeypatch):
    """If frames already exist on disk (e.g. previous run), they are attached
    without invoking ffmpeg."""
    fe = FrameExtractor(tmp_path)
    monkeypatch.setattr(FrameExtractor, "available", staticmethod(lambda: True))
    monkeypatch.setattr(fe, "_stream_url", lambda vid: "http://fake/stream")
    c = make_chunk(0)
    fdir = fe.frames_dir("vid")
    fdir.mkdir(parents=True)
    (fdir / f"{c.chunk_id}.jpg").write_bytes(b"fakejpeg")
    n = fe.attach_frames("vid", [c])
    assert n == 1
    assert c.extra["frame_path"].endswith(".jpg")
    assert c.extra["frame_time"] == 4.5     # midpoint of 0..9
    # frame path flows into vector-store metadata
    assert "frame_path" in c.metadata()


def test_extract_frame_handles_failure(tmp_path, monkeypatch):
    fe = FrameExtractor(tmp_path)
    monkeypatch.setattr(fe, "extract_frame", lambda *a, **k: False)
    monkeypatch.setattr(FrameExtractor, "available", staticmethod(lambda: True))
    monkeypatch.setattr(fe, "_stream_url", lambda vid: "http://fake/stream")
    assert fe.attach_frames("vid", [make_chunk(0)]) == 0
