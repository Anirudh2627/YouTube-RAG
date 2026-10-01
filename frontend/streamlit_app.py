"""Streamlit UI for the YouTube RAG Assistant.

Pure presentation layer: every capability comes from the FastAPI backend
(see app/api). Keeps business logic out of the UI, so the same API serves
CLI, tests, and this app.

Run:  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎬",
                   layout="wide")


# ------------------------------------------------------------------ api calls
def api_get(path: str):
    r = httpx.get(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, timeout: float = 300):
    r = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎬 YouTube RAG")
    backend_url_input = st.text_input("Backend URL", value=API_URL)
    API_URL = backend_url_input.rstrip("/")

    try:
        health = api_get("/health")
        st.success("backend online", icon="✅")
        st.caption(f"embedder: `{health['embedding_model']}`  \n"
                   f"vector store: `{health['vectorstore']}`  \n"
                   f"llm: `{health['llm']}`  \n"
                   f"reranker: `{health['reranker']}`  \n"
                   f"chunks stored: `{health['chunks']}`")
    except Exception:
        st.error("backend unreachable — start it with "
                 "`uvicorn app.api.main:app`", icon="❌")
        health = None

    st.divider()
    st.subheader("Retrieval settings")
    scope_mode = st.radio("Scope", ["Single video", "All videos (cross-video)"],
                          horizontal=True)
    top_k = st.slider("Stage-1 candidates (top_k)", 3, 30, 15)
    top_n = st.slider("Stage-2 context chunks (top_n)", 1, 8, 4)
    debug_mode = st.toggle("Debug mode (retrieval trace)", value=True)

    st.divider()
    if st.button("➕ Load offline demo video"):
        try:
            with st.spinner("ingesting demo fixture..."):
                res = api_post("/videos/process",
                               {"url": "fixture:data/fixtures/demo_lecture.json"})
            st.session_state["active_video"] = res["video"]["video_id"]
            st.rerun()
        except Exception as e:
            st.error(f"demo load failed: {e}")
    if st.button("🧹 New conversation"):
        st.session_state.pop("conversation_id", None)
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------- header
st.title("YouTube RAG Assistant")
st.caption("Grounded Q&A over video transcripts — every answer cites clickable "
           "timestamps from the video.")

# ------------------------------------------------------------ ingest section
with st.container(border=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input("YouTube URL (video or playlist)",
                            placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        st.write("")
        as_playlist = st.checkbox("Playlist", value=False)
        process = st.button("Process", type="primary", use_container_width=True,
                            disabled=not url)
    if process:
        try:
            if as_playlist:
                with st.spinner("ingesting playlist (this can take a while)..."):
                    res = api_post("/videos/process-playlist", {"url": url})
                ok = [r for r in res["results"] if r.get("video")]
                bad = [r for r in res["results"] if r.get("error")]
                st.success(f"ingested {len(ok)}/{res['n_requested']} videos")
                for r in bad:
                    st.warning(r["error"][:200])
                if ok:
                    st.session_state["active_video"] = ok[0]["video"]["video_id"]
                st.rerun()
            else:
                with st.spinner("extracting transcript → chunking → embedding..."):
                    res = api_post("/videos/process", {"url": url})
                v, cached = res["video"], res["cached"]
                st.session_state["active_video"] = v["video_id"]
                st.success(("loaded from cache" if cached else
                            f"processed in {res['elapsed_s']}s") +
                           f" — {v['n_chunks']} chunks")
                st.rerun()
        except Exception as e:
            st.error(str(e))

# ------------------------------------------------------------- video library
videos: list[dict] = []
try:
    videos = api_get("/videos")
except Exception:
    pass

if not videos:
    st.info("No videos ingested yet. Paste a YouTube URL above — or load the "
            "offline demo from the sidebar (no network needed).")
    st.stop()

active_id = st.session_state.get("active_video")
if scope_mode == "Single video":
    labels = {f"{v['title'][:60] or v['video_id']} ({v['video_id']})": v["video_id"]
              for v in videos}
    default_label = next((l for l, vid in labels.items() if vid == active_id),
                         next(iter(labels)))
    picked = st.selectbox("Active video", list(labels), index=list(labels).index(default_label))
    active_id = labels[picked]
    st.session_state["active_video"] = active_id
else:
    active_id = None

# ------------------------------------------------------------- video info card
if active_id:
    v = next((x for x in videos if x["video_id"] == active_id), None)
    if v:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.markdown(f"**{v['title'] or v['video_id']}**")
            c2.markdown(f"📺 {v['channel'] or '—'}")
            dur = v.get("duration_s")
            c3.markdown(f"⏱ {int(dur // 60)}:{int(dur % 60):02d}" if dur else "⏱ —")
            c4.markdown(f"🧩 {v['n_chunks']} chunks")
            if v.get("language"):
                c1.caption(f"language: {v['language']} · ingested: {v.get('ingested_at', '')}")

# ------------------------------------------------------------------ chat area
st.subheader("Chat")
msgs = st.session_state.setdefault("messages", [])
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"📎 sources ({len(m['sources'])})"):
                for s in m["sources"]:
                    st.markdown(
                        f"[**{s['timestamp_label']}**]({s['url']}) "
                        f"*(score {s['score']:.3f})*  \n"
                        f"<span style='color:gray'>{s['text'][:280]}…</span>",
                        unsafe_allow_html=True)
                    if s.get("frame_path") and os.path.exists(s["frame_path"]):
                        st.image(s["frame_path"], width=220)
        if m.get("debug"):
            with st.expander("🔬 debug trace"):
                st.json(m["debug"])

if q := st.chat_input("Ask a question about the video…"):
    msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            resp = api_post("/chat", {
                "query": q,
                "video_id": active_id,
                "conversation_id": st.session_state.get("conversation_id"),
                "top_k": top_k, "top_n": top_n,
                "debug": debug_mode,
            })
            st.session_state["conversation_id"] = resp["conversation_id"]
            st.markdown(resp["answer_markdown"] or resp["answer"])
            entry = {"role": "assistant",
                     "content": resp["answer_markdown"] or resp["answer"],
                     "sources": resp["sources"],
                     "debug": resp.get("debug")}
            if debug_mode and resp.get("debug"):
                with st.expander("🔬 debug trace"):
                    st.json(resp["debug"])
            if resp["sources"]:
                with st.expander(f"📎 sources ({len(resp['sources'])})"):
                    for s in resp["sources"]:
                        st.markdown(f"[**{s['timestamp_label']}**]({s['url']}) "
                                    f"*(score {s['score']:.3f})*  \n"
                                    f"<span style='color:gray'>{s['text'][:280]}…</span>",
                                    unsafe_allow_html=True)
                        if s.get("frame_path") and os.path.exists(s["frame_path"]):
                            st.image(s["frame_path"], width=220)
            msgs.append(entry)
        except Exception as e:
            st.error(f"chat failed: {e}")
            msgs.append({"role": "assistant", "content": f"⚠️ {e}"})
