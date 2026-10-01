import pytest

from app.utils.youtube import InvalidYouTubeURLError, extract_video_id, watch_url

VALID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VALID}",
    f"https://youtube.com/watch?v={VALID}&t=123s",
    f"https://m.youtube.com/watch?v={VALID}",
    f"https://youtu.be/{VALID}",
    f"https://youtu.be/{VALID}?si=abc123",
    f"https://www.youtube.com/shorts/{VALID}",
    f"https://www.youtube.com/live/{VALID}",
    f"https://www.youtube.com/embed/{VALID}",
    f"www.youtube.com/watch?v={VALID}",
    VALID,
])
def test_extract_valid(url):
    assert extract_video_id(url) == VALID


@pytest.mark.parametrize("bad", [
    "", "https://example.com/watch?v=abc", "https://youtu.be/tooshort",
    "https://www.youtube.com/", "not a url at all",
    "https://www.youtube.com/playlist?list=PL123",
])
def test_extract_invalid(bad):
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id(bad)


def test_watch_url():
    assert watch_url(VALID) == f"https://www.youtube.com/watch?v={VALID}"
    assert watch_url(VALID, 763) == f"https://www.youtube.com/watch?v={VALID}&t=763s"

