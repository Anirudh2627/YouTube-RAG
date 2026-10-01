import pytest

from app.utils.timefmt import format_timestamp, parse_timestamp


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00"), (9, "0:09"), (75, "1:15"), (763, "12:43"),
    (3599, "59:59"), (3600, "1:00:00"), (3725, "1:02:05"),
])
def test_format(seconds, expected):
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize("text,expected", [
    ("0:00", 0), ("12:43", 763), ("1:02:05", 3725), ("90", 90), (" 1:15 ", 75),
])
def test_parse(text, expected):
    assert parse_timestamp(text) == expected


def test_roundtrip():
    for s in [0, 1, 59, 60, 61, 3599, 3600, 86399]:
        assert parse_timestamp(format_timestamp(s)) == s


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_timestamp("1:2:3:4")
