from app.generation.rag_engine import render_citations, render_sources_section, strip_markers
from app.models.schemas import Source


def make_sources():
    return [
        Source(video_id="vid12345678", title="My Video", start_time=763, end_time=812,
               text="about attention", url="https://www.youtube.com/watch?v=vid12345678&t=763s",
               timestamp_label="12:43", score=0.9),
        Source(video_id="vid12345678", title="My Video", start_time=1101, end_time=1150,
               text="about rnns", url="https://www.youtube.com/watch?v=vid12345678&t=1101s",
               timestamp_label="18:21", score=0.8),
    ]


def test_render_citations_replaces_markers():
    md, cited = render_citations("Attention is fast [C1] and beats RNNs [C2].", make_sources())
    assert "[12:43](https://www.youtube.com/watch?v=vid12345678&t=763s)" in md
    assert "[18:21](https://www.youtube.com/watch?v=vid12345678&t=1101s)" in md
    assert cited == {0, 1}
    assert "[C1]" not in md


def test_out_of_range_markers_dropped():
    md, cited = render_citations("Claim [C9] and [C1].", make_sources())
    assert "C9" not in md
    assert cited == {0}


def test_strip_markers():
    assert strip_markers("Hello [C1] world [C2]") == "Hello world"


def test_sources_section_links():
    section = render_sources_section(make_sources(), cited={1})
    assert "**Relevant sections:**" in section
    assert "[12:43](https://www.youtube.com/watch?v=vid12345678&t=763s)" in section
    assert "↳ cited" in section


def test_multi_video_labels_include_title():
    srcs = make_sources()
    srcs[1].video_id = "otherVideo99"
    md, _ = render_citations("Compare [C1] with [C2].", srcs)
    assert "My Video" in md          # title disambiguates the second video
    assert "otherVideo99" in md or "18:21" in md


def test_empty_sources():
    md, cited = render_citations("No sources [C1].", [])
    assert cited == set()
    assert render_sources_section([], set()) == ""
