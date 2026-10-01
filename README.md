# 🎬 YouTube RAG Assistant

**Grounded, conversational Q&A over YouTube video transcripts — with clickable timestamp citations, hybrid retrieval, cross-encoder reranking, query rewriting, and a real evaluation harness.**

Paste a video URL → the system extracts the transcript, chunks it on sentence boundaries while preserving timestamps, embeds it into ChromaDB → then answers your questions *strictly from the video*, citing `[12:43](https://youtube.com/watch?v=…&t=763s)`-style links for every claim.

```
User:  What learning rate does the speaker recommend for fine-tuning?

AI:    For fine-tuning a small Transformer like BERT base, the speaker
       recommends a learning rate of about 2e-5, up to 5e-5 [6:50] …

       Relevant sections:
       - [6:50](…&t=410s) (score 0.931) ↳ cited
       - [10:34](…&t=634s) (score 0.212)
```

---

## Table of contents

1. [Problem](#problem)
2. [Solution & architecture](#solution--architecture)
3. [Quickstart](#quickstart)
4. [The RAG pipeline, stage by stage](#the-rag-pipeline-stage-by-stage)
5. [Technical decisions & tradeoffs](#technical-decisions--tradeoffs)
6. [Evaluation (real numbers)](#evaluation-real-numbers)
7. [Debug mode](#debug-mode)
8. [API reference](#api-reference)
9. [Configuration](#configuration)
10. [Project structure](#project-structure)
11. [Testing](#testing)
12. [Docker](#docker)
13. [Limitations & future work](#limitations--future-work)

---

## Problem

Video is the largest corpus of technical knowledge that **search cannot reach inside**. A 90-minute lecture contains exactly one 40-second answer to your question, but:

- scrubbing a timeline to find it is O(video length) human time;
- YouTube's chapter markers are coarse and often missing;
- full transcripts (when available) are a wall of unpunctuated ASR text;
- stuffing a whole transcript into an LLM is expensive, exceeds context limits on long videos, and dilutes attention over irrelevant content.

## Solution & architecture

A retrieval-augmented generation pipeline that indexes the transcript once and answers from the *few relevant seconds* of it — with citations that jump the video player to the exact moment.

```
                        ┌────────────────────────── INGESTION (once per video) ─────────────────────────┐
 YouTube URL ──▶ video_id ──▶ cache? ──hit──▶ serve from Chroma + disk cache
                              │miss
                              ▼
                   TranscriptProvider chain            youtube-transcript-api → yt-dlp fallback
                              ▼                        (cookies supported for blocked IPs)
                   Cleaning                            [Music]/♪/URL artifacts, ALL-CAPS fix,
                              ▼                        whitespace, consecutive-duplicate removal
                   Timestamp-aware chunking            sentence boundaries · token budget · sentence
                              ▼                        overlap · char→time interpolation
                   Embedder (BAAI/bge-small-en-v1.5)   ──▶ ChromaDB (cosine HNSW)  +  BM25 side-index
                        └────────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────────────── QUERY PATH ────────────────────────────────────┐
 question + history ──▶ QueryRewriter (heuristic) ──▶ standalone query
                              ▼
                   Stage 1: hybrid retrieval               vector top-K (K=15) ┐
                                                           BM25 top-K          ┴─▶ RRF fusion
                              ▼
                   Stage 2: cross-encoder rerank           ms-marco-MiniLM over (query, chunk) pairs → top-N (N=4)
                              ▼
                   Context construction                    numbered [C1..CN] blocks with timestamp headers
                              ▼
                   LLM (Groq / any OpenAI-compatible)      strict grounding system prompt
                              ▼
                   Citation post-processing                [Cn] → [12:43](watch?v=…&t=763s) · sources list
                              ▼
                   Answer + clickable timestamps
                        └────────────────────────────────────────────────────────────────────────────────┘
```

Every arrow above is a small, replaceable module with an ABC at the seam (`TranscriptProvider`, `Chunker`, `Embedder`, `VectorStore`, `Reranker`, `LLM`, `Judge`).

## Quickstart

### Option A — Docker (backend + Streamlit UI)

```bash
cp .env.example .env          # put your Groq key in LLM_API_KEY
docker compose up --build
# UI: http://localhost:8501   API docs: http://localhost:8000/docs
```

### Option B — Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -r requirements.txt
cp .env.example .env                                                 # add your LLM_API_KEY

# terminal 1 — API
uvicorn app.api.main:app --reload --port 8000
# terminal 2 — UI
streamlit run frontend/streamlit_app.py
```

### Option C — No API key, no network (offline demo)

Everything runs offline with a deterministic hashing embedder and an
extractive mock LLM — the same fallbacks the CI test-suite uses:

```bash
python scripts/chat.py fixture:data/fixtures/demo_lecture.json
```

Or click **“Load offline demo video”** in the UI sidebar. The demo fixture is a
**synthetic 11-minute lecture transcript** (authored for this repo, see
`scripts/make_demo_fixture.py`) — synthetic on purpose so the evaluation
ground truth is exact and reproducible. For real videos just paste a URL
(requires network; from datacenter IPs YouTube may block transcript access —
see `YTDLP_COOKIES_FILE` in `.env.example`).

```bash
python scripts/ingest.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python scripts/chat.py   dQw4w9WgXcQ
python scripts/evaluate.py --embedding bge --tag prod
```

## The RAG pipeline, stage by stage

### 1. Transcript extraction (`app/ingestion/`)
Provider chain with graceful degradation: `youtube-transcript-api` (fast, hits
the timed-text endpoint, prefers manual captions over ASR) → `yt-dlp`
subtitles (slower, more robust negotiation, also supplies title/channel/
duration metadata). Both raise a typed `TranscriptUnavailableError` that the
API maps to HTTP 422. Compatible with both the modern (≥1.0) and legacy
(0.6.x) `youtube-transcript-api` APIs.

### 2. Cleaning (`app/ingestion/cleaning.py`, `app/utils/text.py`)
NFKC normalization + zero-width stripping → artifact removal (`[Music]`,
`[Applause]`, `♪`, URLs) → ALL-CAPS de-shouting (acronym-safe) → whitespace
normalization → consecutive-duplicate removal (rolling-caption artifact).
Timestamps are never touched — cleaning must not shift the timeline.

### 3. Timestamp-aware chunking (`app/ingestion/chunking.py`)
**Not** fixed-character splitting. The default `TimestampChunker`:

- merges caption segments and cuts at **sentence boundaries** (abbreviation-safe splitter);
- enforces a token budget (`target=160`, `max=240`) with **whole-sentence overlap** (`≈40` tokens) so answers straddling a boundary stay retrievable from one chunk;
- maps every sentence back to the timeline via **char→time interpolation** inside each caption segment, so sub-segment timestamps stay accurate;
- merges sub-`min_tokens` tail fragments backwards (no 3-word orphan chunks).

Each chunk carries `chunk_id, video_id, video_url, title, start_time, end_time,
index, text`.

### 4. Embeddings (`app/embeddings/`)
Default `BAAI/bge-small-en-v1.5` (384-d, strong MTEB for its size; swap to
`bge-base-en-v1.5` via env). BGE's retrieval query instruction is applied to
queries only, per the model card. Vectors are L2-normalized so cosine =
dot product. A deterministic `HashingEmbedder` exists **for CI only**.

### 5. Vector store (`app/vectorstore/`)
`VectorStore` ABC + two backends: **Chroma** (persistent, cosine HNSW,
precomputed embeddings — the DB never owns the embedding decision) and an
in-memory numpy store (tests, tiny deployments). The app only touches the
ABC, so additional backends can be added behind the `VectorStore` ABC.
`video_id` metadata filtering gives per-video scoping *and* collection-wide
cross-video retrieval from a single store.

### 6. Hybrid retrieval (`app/retrieval/`)
- **Vector leg**: query embedding → top-K (K=15) cosine search.
- **BM25 leg**: Okapi BM25 **implemented from scratch** (inverted index, ~60
  lines) over stored documents; lazily rebuilt per scope and cached.
- **Fusion**: **Reciprocal Rank Fusion** — rank-based, scale-free, one
  parameter (k=60), the default hybrid strategy in Elasticsearch/Weaviate.
  Vector and BM25 scores live on incomparable scales; RRF sidesteps
  normalization entirely.

Why both legs: vector search handles paraphrase (“main idea of the video”),
BM25 handles exact terms vector models blur (“learning rate”, “RoPE”,
numbers, code identifiers).

### 7. Reranking (`app/retrieval/reranker.py`)
Cross-encoder `ms-marco-MiniLM-L-6-v2` scores (query, chunk) pairs jointly —
much sharper than bi-encoder cosine, but O(K) forward passes, hence the
**retrieve wide (15) → rerank narrow (4)** funnel. Scores are sigmoided to
0..1 for interpretability. A deterministic `HeuristicReranker` (token-F1 +
stage-1 blend) is the offline fallback. Measured impact below: **P@1 0.80 →
0.93, MRR 0.87 → 0.96**.

### 8. Grounded generation (`app/generation/`)
System prompt hard rules: answer **only** from context; cite every factual
sentence with `[Cn]`; refuse with the exact sentence *“I couldn't find
enough information about that in the video.”*; redirect off-topic asks;
follow the transcript over parametric knowledge on conflict. The provider is
one OpenAI-compatible httpx client — **Groq today, OpenAI/Together/Ollama/vLLM
by changing `LLM_BASE_URL`**, no vendor SDK, with retry/backoff on 429.

### 9. Citations (`render_citations`)
`[Cn]` markers are mapped to `[12:43](https://youtube.com/watch?v=ID&t=763s)`;
out-of-range markers (LLMs do that) are dropped; `cited_indices` reports which
sources were actually used; a **“Relevant sections”** block guarantees
clickable timestamps even when the model forgets inline markers. In
multi-video mode, labels include the video title.

### 10. Conversation memory + query rewriting (`app/generation/rewriter.py`)
History lives in a TTL conversation store. Follow-ups are rewritten into
**standalone queries before retrieval** (cheap small model, e.g.
`llama-3.1-8b-instant`): “Why is it useful?” → “Why is the attention mechanism
useful?”. Rewriting is gated by an anaphora detector (short query + pronouns
like *it/that/those*) so standalone questions are never mangled, and a
conservative heuristic (topic carry-over) covers the no-key/offline path.
Only the rewritten query hits the retriever; the last few turns go to the LLM
for coherence — never the whole transcript, never the whole history.

### 11. Long videos
Nothing ever loads a full transcript into the LLM. A 3-hour lecture becomes
~700 chunks; a query embeds once, retrieves 15, reranks to 4 — the LLM sees
~600 tokens of context regardless of video length. Per-video caching
(`data/cache/<video_id>/`) + persistent Chroma means each video is ingested
exactly once.

### 12. Multiple videos
Ingest videos individually into the **shared** store; chatting with
`video_id: null` searches the whole collection, so “compare the approaches in
these three videos” works, with per-source video attribution in citations.

## Technical decisions & tradeoffs

| Decision | Why | Tradeoff accepted |
|---|---|---|
| Hand-rolled BM25, RRF, metrics | ~60 lines each; keeps fusion logic inspectable & tunable | no exotic BM25 variants (BM25F, field weights) |
| LangChain/LlamaIndex **not** used | every stage is a small ABC; frameworks would hide exactly the parts worth understanding | re-implemented a few commodity glue pieces |
| Chroma embedded (no server) | zero-ops persistence, cosine HNSW, metadata filters | single-node scale; additional backends can be added behind the `VectorStore` ABC |
| Groq via raw OpenAI-compatible httpx | one client for Groq/OpenAI/Ollama/vLLM; timeouts+retries observable | no streaming yet (see future work) |
| BGE-small default | best quality/latency/size ratio for CPU; `-base` is a config flip | 384-d misses a little vs large models |
| Cross-encoder rerank on 15 candidates | measured P@1 +13pts, MRR +8pts (below) | +50–150 ms CPU per query |
| Sentence-boundary chunks w/ overlap + fragment merging | complete propositions embed better; overlap saves boundary-straddling answers | ~1.3× storage from overlap duplication |
| Timestamp interpolation inside caption segments | sub-segment citation accuracy without ASR word timings | linear speaking-rate assumption |
| Mock LLM + hashing embedder fallbacks | entire stack testable offline & deterministic; refusal paths included | mock answers are extractive, not generative |
| Synthetic eval fixture | exact ground truth, reproducible forever, no creator-content licensing | doesn't capture ASR noise — real-video spot checks recommended |

## Evaluation (real numbers)

**Nothing in this section is hand-written.** Every figure is produced by
`python scripts/evaluate.py` and persisted with full config + per-question
detail under `data/eval/reports/` (JSON + Markdown).

**Dataset**: `data/eval/demo_dataset.json` — 17 questions over the synthetic
demo lecture: 15 answerable (gold = timeline intervals, several with *two*
acceptable locations, e.g. explanation **and** summary) + 2 negatives (one
unanswerable-in-video, one off-topic). Gold intervals are authored against
the fixture, so relevance judgments are verifiable.

**Retrieval** — relevance = chunk time-span overlaps a gold interval.
Stage 2 (post-rerank, top_n=4) numbers, stage 1 in parentheses:

| Config | MRR | Recall@5 | Precision@1 | HitRate@3 |
|---|---|---|---|---|
| hashing + vector, no rerank *(CI baseline)* | 0.63 (0.65) | 0.58 (0.66) | 0.53 | 0.73 |
| BGE vector, no rerank | 0.87 | 0.68 (0.79) | 0.80 | 0.93 |
| BGE vector + cross-encoder | **0.96** (0.87) | 0.80 (0.79) | **0.93** (0.80) | **1.00** (0.93) |
| BGE BM25-only + cross-encoder | **0.96** (0.92) | **0.84** (0.85) | **0.93** (0.87) | **1.00** (0.93) |
| BGE **hybrid** + cross-encoder *(prod)* | **0.96** (0.90) | 0.80 (0.82) | **0.93** (0.80) | **1.00** (1.00) |

Takeaways, stated honestly:
- **Reranking is the single biggest win** (vector-only MRR 0.87 → 0.96; P@1 0.80 → 0.93; HitRate@3 → 1.00) — the retrieve-wide/rerank-narrow funnel earns its latency.
- On this small in-domain set, **lexical matching is unusually strong** (queries share vocabulary with the lecture), so BM25+rerank ≈ hybrid+rerank; hybrid's advantage shows on paraphrased/out-of-vocabulary queries (“main idea” → intro, where BM25 alone scored 0.92 stage-1 MRR vs vector 0.87).
- The hashing-embedder row exists to keep CI meaningful, not to be good.

**Generation** — prod config, offline path (MockLLM extractive answers +
deterministic `HeuristicJudge`), so treat these as a **conservative floor**:

| Metric | Score | What it measures |
|---|---|---|
| answer_correctness | 0.69 | stemmed keyword coverage of the reference answer |
| faithfulness | **0.97** | fraction of answer sentences lexically supported by retrieved context |
| context_relevance | 0.78 | fraction of context chunks on-topic for the question |
| citation_accuracy | 0.84 | cited spans ∩ gold spans (precision + coverage) |
| refusal_accuracy | **1.00** | both negatives correctly refused/redirected, zero hallucinated answers |

Reproduce everything:

```bash
python scripts/evaluate.py --embedding bge --mode hybrid --tag prod      # ~1 min on CPU
python scripts/evaluate.py --embedding hashing --no-rerank --tag ci      # seconds, offline
```

Reports: `data/eval/reports/report_bge-hybrid-rerank_*.json` (this README's
numbers come from the 2026-09-23 runs; re-running regenerates equivalents).

## Debug mode

Every query can return the full trace (`"debug": true` on `/chat`, or the
sidebar toggle in the UI):

```
original query → rewritten query (+ which method) → retrieval mode
→ stage-1 candidates [vector_score · bm25_score · fused_score per chunk]
→ reranked top-N [rerank_score, final_rank] → exact final context text
→ llm model + per-stage timings (embed, vector, bm25, rerank, llm, total)
```

Terminal: `python scripts/chat.py …` then `/debug on`. This is the tool for
answering “*why* did it cite 6:34 and not 4:10?” — scores for both are right
there.

## API reference

Interactive docs: **http://localhost:8000/docs**

| Method & path | Purpose |
|---|---|
| `POST /videos/process` | ingest `{url, language?, force?}` → `{video, cached, elapsed_s}` (also accepts `fixture:<path>`) |
| `GET /videos` | ingested library |
| `GET /videos/{video_id}` | metadata (title, channel, duration, n_chunks…) |
| `GET /videos/{video_id}/sources` | every stored chunk with timestamps |
| `DELETE /videos/{video_id}` | remove video, vectors & cache |
| `POST /chat` | `{query, video_id?, conversation_id?, top_k?, top_n?, debug?}` → `{answer, answer_markdown, sources[], cited_indices[], conversation_id, debug?}` |
| `GET /health` | liveness + pipeline fingerprint |

Example `POST /chat` response (abridged):

```json
{
  "answer": "The original paper uses eight attention heads, each with a dimension of sixty-four…",
  "answer_markdown": "…sixty-four [4:13](https://www.youtube.com/watch?v=demoLctr001&t=253s)…",
  "sources": [{"video_id": "demoLctr001", "start_time": 252.6, "end_time": 327.7,
               "url": "https://www.youtube.com/watch?v=demoLctr001&t=252s",
               "timestamp_label": "4:13", "score": 0.997, "text": "…"}],
  "cited_indices": [0],
  "conversation_id": "9f2c1a7de3b04c11"
}
```

## Configuration

Everything via env / `.env` (see `.env.example`). Highlights:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` \| `openai-compatible` \| `mock` |
| `LLM_API_KEY` | — | **never committed**; without it the app boots in offline mock mode |
| `LLM_MODEL` | `openai/gpt-oss-120b` | any Groq model |
| `LLM_BASE_URL` | Groq's | point at OpenAI/Ollama/vLLM/Together |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | any sentence-transformers model |
| `VECTORSTORE_BACKEND` | `chroma` | `chroma` \| `memory` |
| `RETRIEVAL_MODE` | `hybrid` | `vector` \| `bm25` \| `hybrid` |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_N` | `15` / `4` | the funnel widths |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | e.g. `BAAI/bge-reranker-base` for more accuracy |
| `CHUNK_TARGET_TOKENS` / `_MAX_` / `_OVERLAP_` | `160/240/40` | chunk budget |
| `YTDLP_COOKIES_FILE` | — | cookies for IP-blocked environments |

## Project structure

```
youtube-rag/
├── app/
│   ├── api/            # FastAPI: main, routes_videos, routes_chat, deps (DI)
│   ├── ingestion/      # providers (yt-api, yt-dlp, file), cleaning, chunking
│   ├── embeddings/     # Embedder ABC + sentence-transformers & hashing impls
│   ├── vectorstore/    # VectorStore ABC + Chroma & in-memory backends
│   ├── retrieval/      # BM25 (from scratch), RRF hybrid, rerankers, pipeline
│   ├── generation/     # LLM ABC, OpenAI-compat client, mock, prompts, rewriter, RAG engine
│   ├── memory/         # TTL conversation store
│   ├── services/       # VideoService (ingest+cache), Container (composition root)
│   ├── evaluation/     # metrics, deterministic evaluation judge, runner/reports
│   ├── models/         # shared Pydantic schemas (== API schemas)
│   └── utils/          # youtube URLs, timefmt, text cleaning, logging, stemming
├── frontend/streamlit_app.py     # thin UI over the HTTP API
├── scripts/            # ingest.py · chat.py · evaluate.py · make_demo_fixture.py
├── tests/              # pytest suite, offline by default
├── notebooks/          # chunking & retrieval ablations
├── data/               # fixtures · eval dataset · reports · cache · vectorstore
├── Dockerfile · docker-compose.yml · Makefile · .env.example
```

## Testing

```bash
pytest -m "not slow"     # 130 offline tests, <3s, no downloads, deterministic
pytest                   # + 2 slow tests (real BGE embeddings, real cross-encoder)
```

Covered: URL parsing (10 shapes), timestamp round-trips, cleaning artifacts,
chunker invariants (coverage, monotonicity, overlap, no micro-fragments,
metadata), embedder determinism, both vector-store backends incl. upsert/
filter/persistence, BM25 ranking, RRF fusion math, rerankers, retrieval
pipeline (gold-span hits, video scoping, cache invalidation), rewriter
(gating, heuristic path), citation rendering (incl.
out-of-range markers), engine behavior (grounding, refusal, off-topic,
follow-up rewriting, debug trace), all API endpoints (incl. 400/404/422),
metric formulas vs hand-computed values, deterministic evaluation,
and two end-to-end journeys.

## Docker

```bash
docker compose up --build     # backend :8000 + Streamlit :8501
```

- CPU-only torch wheel (the CUDA wheel would add ~2.5 GB).
- Model weights cached in a named volume (`hf-cache`) — downloaded once.
- `./data` bind-mount persists Chroma + the per-video cache across restarts.
- Secrets flow from `.env` → compose `environment:` — never baked into images.
## Demo

Offline (no keys, no network) terminal session — real output:

```
you> How many attention heads does the original Transformer use?
assistant> The original paper uses eight attention heads, each with a dimension
of sixty-four, for a total model dimension of five hundred twelve. [4:13]
…
```

UI screenshots/GIF: run `docker compose up`, load the offline demo from the
sidebar, and capture `localhost:8501` — the layout follows the spec (URL bar,
video card, chat, sources with clickable timestamps, debug expander).

## Limitations & future work

**Known limitations (stated plainly):**
- ASR captions lack punctuation on some videos → sentence splitting degrades to segment-boundary chunking.
- Datacenter IPs are often blocked by YouTube; cookies file supported as workaround.
- No answer streaming yet; the UI waits for the full completion.
- Evaluation dataset is one synthetic video (17 Qs) — enough to regression-gate the pipeline, not enough to claim generalization; extend with real-video QA pairs.
- The heuristic faithfulness judge is lexical; paraphrased-but-faithful answers can be under-scored.

**Roadmap:**
- Streaming responses (SSE) + websocket debug trace.
- Query routing/agentic retrieval (multi-hop: retrieve → reason → re-retrieve).
- Better rerankers (`bge-reranker-v2-m3`), optional ColBERT-style late interaction.
- Multilingual transcripts (Whisper fallback, per-language embedders).
- pgvector/Qdrant backend behind the existing ABC; Redis conversation store.
- Auto-generated chapter summaries as an extra retrieval layer (parent-document retrieval).

---

### Reproducibility checklist

- `make test` — deterministic offline test suite.
- `make eval` / `make eval-ci` — regenerate every number in this README.
- Reports carry the full config snapshot + per-question traces.
- Pinned fallbacks (mock LLM, hashing embedder, memory store) keep CI honest without network or keys.
