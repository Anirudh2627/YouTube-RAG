````markdown
# 🎬 YouTube RAG Assistant

**Grounded, conversational Q&A over YouTube video transcripts — with clickable timestamp citations, hybrid retrieval, Reciprocal Rank Fusion, cross-encoder reranking, conversation-aware query rewriting, evaluation, FastAPI, Streamlit, and Docker.**

> **Built to explore a practical question: what actually happens when a RAG system stops being a simple `embed → retrieve → generate` demo?**

Paste a YouTube URL → the system extracts and cleans the transcript → creates timestamp-aware chunks → indexes them using both semantic and lexical retrieval → fuses the results with RRF → reranks candidates with a cross-encoder → generates a grounded answer → maps citations back to the exact part of the video.

---

## 📌 Table of Contents

1. [Problem](#problem)
2. [Solution](#solution)
3. [Architecture](#architecture)
4. [What Makes This More Than Basic RAG](#what-makes-this-more-than-basic-rag)
5. [The RAG Pipeline](#the-rag-pipeline)
6. [Transcript Ingestion](#1-transcript-ingestion)
7. [Transcript Cleaning](#2-transcript-cleaning)
8. [Timestamp-Aware Chunking](#3-timestamp-aware-chunking)
9. [Embeddings](#4-embeddings)
10. [Vector Storage](#5-vector-storage)
11. [Hybrid Retrieval](#6-hybrid-retrieval)
12. [Reciprocal Rank Fusion](#7-reciprocal-rank-fusion)
13. [Cross-Encoder Reranking](#8-cross-encoder-reranking)
14. [Grounded Generation](#9-grounded-generation)
15. [Citation System](#10-citation-system)
16. [Conversation Memory](#11-conversation-memory)
17. [Query Rewriting](#12-query-rewriting)
18. [Long Videos](#13-long-videos)
19. [Multiple Videos](#14-multiple-videos)
20. [Where RAG Broke and How I Fixed It](#where-rag-broke-and-how-i-fixed-it)
21. [Architectural Decisions](#architectural-decisions)
22. [Evaluation](#evaluation)
23. [Debug Mode](#debug-mode)
24. [API](#api)
25. [Configuration](#configuration)
26. [Project Structure](#project-structure)
27. [Testing](#testing)
28. [Docker](#docker)
29. [Quickstart](#quickstart)
30. [Example](#example)
31. [Engineering Lessons](#engineering-lessons)
32. [Limitations](#limitations)
33. [Future Work](#future-work)
34. [Security](#security)

---

# Problem

Technical knowledge is increasingly stored inside long-form video.

A 60–90 minute technical lecture may contain the exact answer to a question for only 30–60 seconds.

Finding that answer manually means:

- scrubbing through the timeline
- relying on coarse or missing chapter markers
- reading large, noisy ASR transcripts
- remembering approximately where the speaker discussed a topic

A naive RAG implementation seems to solve this:

```text
YouTube
   ↓
Transcript
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Search
   ↓
LLM
````

But this creates another set of problems.

A real system has to answer questions such as:

* What if semantic search misses an exact technical term?
* What if BM25 finds the right words but wrong context?
* What if the best semantic result is not the best final result?
* How large should chunks be?
* What happens when the answer lies across a chunk boundary?
* How do we preserve timestamps after cleaning?
* How do we prevent the LLM from using information outside the video?
* What happens when the user asks a follow-up question?
* What happens when the LLM API returns `429`?
* How do we distinguish an actual model refusal from an infrastructure failure?
* How do we know whether reranking actually improved retrieval?
* How do we test the system without downloading models or calling an LLM?
* How do we deploy the exact same system with Docker?

This project was built around those engineering problems.

---

# Solution

The final system uses a two-stage retrieval architecture:

```text
                         ┌───────────────────────────────┐
                         │          INGESTION             │
                         └───────────────────────────────┘

 YouTube URL
      │
      ▼
 Transcript Provider
      │
      ▼
 Cleaning
      │
      ▼
 Timestamp-Aware Chunking
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
 BGE Embeddings             BM25
      │                      │
      ▼                      ▼
 Chroma                 BM25 Index
      │                      │
      └──────────┬───────────┘
                 ▼
        Reciprocal Rank Fusion
                 │
                 ▼
        Stage-1 Candidates
                 │
                 ▼
        Cross-Encoder Reranker
                 │
                 ▼
          Top-N Context
                 │
                 ▼
         Grounded LLM
                 │
                 ▼
       Citation Post-Processing
                 │
                 ▼
        Timestamped Answer
```

The query path additionally supports conversation memory and query rewriting:

```text
User Question + Conversation History
                │
                ▼
       Query Rewriting
                │
                ▼
        Standalone Query
                │
                ▼
       Hybrid Retrieval
                │
                ▼
          Reranking
                │
                ▼
       Grounded Generation
                │
                ▼
        Final Answer + Sources
```

---

# Architecture

## High-Level Architecture

```text
                         ┌──────────────────────────────┐
                         │         YouTube URL          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │     Transcript Extraction    │
                         │                              │
                         │ youtube-transcript-api      │
                         │ yt-dlp fallback              │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       Transcript Cleaning    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Timestamp-Aware Chunking     │
                         │                              │
                         │ target: 160 tokens           │
                         │ max:    240 tokens           │
                         │ overlap: 40 tokens           │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
              ┌────────────────────┐       ┌────────────────────┐
              │ BGE Embeddings     │       │ BM25 Index         │
              │                    │       │                    │
              │ bge-small-en-v1.5  │       │ lexical retrieval  │
              └──────────┬─────────┘       └──────────┬─────────┘
                         │                            │
                         ▼                            ▼
              ┌────────────────────┐       ┌────────────────────┐
              │ Chroma             │       │ BM25 Retrieval     │
              │ Vector Store       │       │                    │
              └──────────┬─────────┘       └──────────┬─────────┘
                         │                            │
                         └──────────────┬─────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │ Reciprocal Rank Fusion       │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Stage-1 Candidate Set        │
                         │                              │
                         │ top-K = 15                   │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Cross-Encoder Reranker       │
                         │                              │
                         │ ms-marco-MiniLM-L-6-v2      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Final Context                │
                         │                              │
                         │ top-N = 4                    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Grounded LLM                 │
                         │                              │
                         │ GPT-OSS-120B                 │
                         │ OpenAI-compatible API        │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Citation Renderer            │
                         │                              │
                         │ [C1] → YouTube timestamp    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                                  Final Answer
```

The architecture is intentionally modular.

Major subsystems are separated behind interfaces so that components can be replaced without rewriting the entire pipeline.

---

# What Makes This More Than Basic RAG

The core pipeline is not:

```text
embed → retrieve → prompt → generate
```

Instead:

```text
ingest
   ↓
clean
   ↓
timestamp-aware chunk
   ↓
embed
   ↓
vector retrieval
   +
BM25 retrieval
   ↓
RRF
   ↓
cross-encoder reranking
   ↓
context construction
   ↓
grounded generation
   ↓
citation validation/rendering
```

Around this core are:

```text
conversation memory
query rewriting
evaluation
failure detection
debug tracing
FastAPI
Streamlit
Docker
automated tests
```

The important engineering goal was not to maximize the number of components.

It was to understand why each component exists and remove components that were adding complexity without solving the core problem.

---

# The RAG Pipeline

## 1. Transcript Ingestion

The ingestion layer converts a YouTube video into a structured transcript.

The pipeline uses a provider abstraction so transcript acquisition is not tightly coupled to one implementation.

Conceptually:

```text
YouTube URL
     │
     ▼
Transcript Provider
     │
     ├── youtube-transcript-api
     │
     └── yt-dlp fallback
     │
     ▼
Transcript segments
```

The resulting segments retain their timing information.

Each segment conceptually contains:

```text
text
start_time
duration / end_time
```

This timing information is preserved throughout the pipeline.

### Why?

Because timestamp citations are a product requirement, not an afterthought.

---

# 2. Transcript Cleaning

Raw transcript data can contain artifacts such as:

```text
[Music]
[Applause]
♪
URLs
excessive whitespace
rolling-caption duplication
```

Cleaning is therefore performed before indexing.

The important rule is:

> **Clean the text without changing the timeline.**

The timestamps are never shifted by text normalization.

This means:

```text
raw transcript timestamp
        ↓
cleaned transcript
        ↓
same timestamp
```

rather than recalculating the video position later.

---

# 3. Timestamp-Aware Chunking

Chunking is one of the most important parts of this project.

The system does not use arbitrary fixed-character splitting.

The current chunker is designed around:

```text
target tokens  = 160
maximum tokens = 240
overlap        = 40
```

The chunker:

* merges transcript segments
* attempts to split on sentence boundaries
* preserves temporal information
* uses whole-sentence overlap
* avoids tiny orphan fragments
* maps text back to the original timeline

Each chunk carries metadata such as:

```text
chunk_id
video_id
video_url
title
start_time
end_time
index
text
```

---

## Why Sentence-Aware Chunking?

Consider:

```text
Chunk A:
"The model first converts the characters into embeddings."

Chunk B:
"These embeddings are then passed into the MLP."
```

If the semantic unit is split badly, retrieval may receive incomplete information.

Overlap helps preserve answers that cross boundaries:

```text
Chunk A
├── sentence 1
├── sentence 2
└── sentence 3

Chunk B
    ├── sentence 3
    ├── sentence 4
    └── sentence 5
```

The overlap is therefore intentional duplication used to improve retrieval continuity.

---

## Timestamp Interpolation

Transcript APIs often provide timestamps for caption segments rather than every sentence.

When a chunk contains multiple sentences inside one caption segment, the system maps sentence positions back to the timeline.

Conceptually:

```text
Caption segment
00:10 ─────────────────────── 00:20
       │          │          │
       S1         S2         S3
```

The system estimates the sentence-level position inside that interval.

This allows the final citation to point closer to the actual relevant content instead of always pointing to the beginning of a large caption segment.

---

# 4. Embeddings

The default embedding model is:

```text
BAAI/bge-small-en-v1.5
```

with:

```text
384 dimensions
```

The system uses the sentence-transformers implementation.

BGE-small was chosen because it provides a useful balance between:

* retrieval quality
* memory requirements
* CPU compatibility
* local development speed
* Docker practicality

The architecture keeps the embedding layer abstract.

The production embedder can therefore be replaced without rewriting retrieval logic.

A deterministic hashing embedder is also available for offline testing.

---

# 5. Vector Storage

The primary vector backend is:

```text
Chroma
```

A memory-backed implementation is also supported.

The abstraction is:

```text
VectorStore
    │
    ├── Chroma
    │
    └── Memory
```

This separation is useful because:

```text
Production
    → persistent Chroma

Tests / offline evaluation
    → in-memory store
```

The retrieval system does not need to know which backend is being used.

Metadata such as `video_id` allows:

```text
per-video retrieval
```

and:

```text
cross-video retrieval
```

when multiple videos are indexed.

---

# 6. Hybrid Retrieval

The system uses two retrieval signals:

```text
Semantic retrieval
+
Lexical retrieval
```

### Vector Retrieval

The query is embedded using BGE.

The vector store returns the top semantic candidates.

This is useful for paraphrases.

For example:

```text
Query:
"Why does the model need embeddings?"

Transcript:
"Characters are represented using learned vectors."
```

The wording differs, but the semantic meaning is related.

---

### BM25 Retrieval

BM25 provides lexical matching.

This is particularly useful for:

```text
learning rate
Adam
RoPE
BERT
specific numbers
code identifiers
technical terminology
```

A semantic embedding model can blur some exact terms.

BM25 provides a complementary signal.

---

# 7. Reciprocal Rank Fusion

Vector similarity and BM25 scores do not naturally live on the same numerical scale.

Instead of trying to normalize them into a shared score space, the system uses:

**Reciprocal Rank Fusion (RRF).**

The basic formula is:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

with:

```text
k = 60
```

Conceptually:

```text
Vector ranking
    ↓
1. chunk A
2. chunk B
3. chunk C

BM25 ranking
    ↓
1. chunk C
2. chunk A
3. chunk D

        ↓

RRF

        ↓

Combined ranking
```

This allows both retrieval methods to contribute based on ranking position.

---

# 8. Cross-Encoder Reranking

Hybrid retrieval is the first stage.

The result is not immediately sent to the LLM.

Instead, the top candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder receives:

```text
(query, candidate_chunk)
```

and evaluates their relevance jointly.

The architecture deliberately uses:

```text
retrieve wide
    ↓
15 candidates
    ↓
rerank
    ↓
4 candidates
    ↓
LLM
```

rather than running the expensive cross-encoder over the entire corpus.

This is the central retrieval funnel:

```text
Large corpus
     ↓
Cheap retrieval
     ↓
15 candidates
     ↓
Expensive reranking
     ↓
4 context chunks
```

---

## Reranker Score

Cross-encoder outputs are converted into an interpretable score.

The final reranking score also retains a small stage-1 ranking prior.

Conceptually:

```text
final rerank score
=
cross-encoder relevance
+
small stage-1 ranking prior
```

The cross-encoder remains the dominant signal.

The stage-1 prior prevents a strong initial candidate from being completely discarded because of a small second-stage scoring difference.

A regression test covers this behavior.

---

# 9. Grounded Generation

The LLM is given only the retrieved context rather than the entire transcript.

The prompt contains:

```text
User question
+
Retrieved chunks
+
Conversation context
```

The system instructs the model to:

* answer using the supplied context
* avoid inventing unsupported information
* cite factual claims
* refuse when the video does not contain enough information
* prefer transcript evidence over outside knowledge when they conflict

Conceptually:

```text
Question
   +
Context
   +
Conversation
   ↓
Grounded Prompt
   ↓
LLM
   ↓
Answer
```

The current production model is:

```text
openai/gpt-oss-120b
```

through an OpenAI-compatible API.

The provider layer supports:

```text
Groq
OpenAI-compatible endpoints
Mock LLM
```

---

# 10. Citation System

The LLM uses internal citation markers:

```text
[C1]
[C2]
[C3]
```

These markers correspond to the numbered context blocks supplied to the model.

The application then converts them into timestamped YouTube links.

For example:

```text
LLM:

"The model uses embeddings to represent characters [C1]."
```

becomes conceptually:

```text
"The model uses embeddings to represent characters
[1:05:29](https://youtube.com/watch?v=VIDEO_ID&t=3929s)."
```

The citation renderer also:

* tracks which sources were cited
* removes invalid/out-of-range citation markers
* preserves timestamp metadata
* exposes cited source indices
* displays relevant retrieved sections

The goal is not just:

```text
Source: YouTube video
```

but:

```text
Source:
Video
└── 1:05:29
    └── Exact relevant section
```

---

# 11. Conversation Memory

A conversational RAG system needs more than the current question.

For example:

```text
User:
Why does the model use embeddings?

Assistant:
...

User:
Why are they useful?
```

The second question is ambiguous without the previous turn.

The system maintains conversation state using a TTL-based conversation store.

The history is deliberately bounded.

The entire conversation is not blindly appended to every prompt.

The current configuration uses:

```text
history_turns_for_rewrite = 4
conversation_ttl_minutes  = 120
```

This keeps conversational context useful without allowing history to grow indefinitely.

---

# 12. Query Rewriting

Follow-up questions can be incomplete:

```text
"Why is it useful?"
"What about that?"
"How does this work?"
```

The retriever needs a standalone query.

The system therefore performs query rewriting before retrieval.

The current implementation uses lightweight heuristic rewriting rather than requiring a second LLM call.

Conceptually:

```text
Conversation
     +
Follow-up question
     ↓
Query Rewriter
     ↓
Standalone query
     ↓
Retriever
```

For example:

```text
Previous topic:
attention mechanism

Question:
"Why is it useful?"

Rewritten:
"Why is the attention mechanism useful?"
```

The rewritten query is what enters retrieval.

The original conversation is still available to the generation layer for conversational coherence.

---

# 13. Long Videos

A long video should not cause the entire transcript to be sent to the LLM.

The system instead scales through retrieval.

For example:

```text
Long video
   ↓
Hundreds of chunks
   ↓
Query embedding
   ↓
Top 15 candidates
   ↓
Top 4 reranked chunks
   ↓
LLM
```

The amount of context seen by the LLM therefore depends primarily on:

```text
retrieval_top_k
rerank_top_n
chunk size
```

rather than the total duration of the video.

This is one of the fundamental reasons to use RAG.

---

# 14. Multiple Videos

Videos can be ingested individually into the shared store.

Retrieval can then be scoped to:

```text
specific video
```

or:

```text
all indexed videos
```

This allows questions involving multiple sources.

Conceptually:

```text
Video A ─┐
Video B ─┼──> Shared retrieval store
Video C ─┘
               │
               ▼
        Hybrid Retrieval
               │
               ▼
        Cross-Encoder
               │
               ▼
      Source-aware citations
```

The citation metadata retains the source video.

---

# Where RAG Broke and How I Fixed It

This section is one of the most important parts of the project.

The project was not treated as a finished RAG pipeline after the first successful answer.

The system was repeatedly tested against its own failure modes.

---

## Failure 1 — Too Many Chunking Strategies

An earlier implementation supported multiple chunking strategies.

That created unnecessary configuration and branching:

```text
TimestampChunker
SentenceChunker
FixedTokenChunker
```

The application fundamentally depends on timestamps.

Maintaining multiple strategies made the system harder to reason about without providing enough value for this project.

### Fix

The chunking layer was simplified to the timestamp-aware implementation.

The final system has one clear chunking path.

This reduced:

* configuration surface
* code paths
* test complexity
* maintenance overhead

without removing the core requirement.

---

# Failure 2 — Multimodal Infrastructure Was Not Core

The project previously contained multimodal/frame-related infrastructure.

The actual product problem was:

```text
YouTube transcript
→ retrieval
→ grounded Q&A
```

The additional multimodal path increased:

* dependencies
* configuration
* tests
* deployment complexity
* maintenance

without being necessary for the core transcript RAG experience.

### Fix

The unused multimodal pipeline was removed.

The project now focuses on the core RAG problem instead of maintaining features simply because they can exist.

---

# Failure 3 — Playlist Ingestion Expanded the Scope

Playlist ingestion added additional:

```text
routes
schemas
services
UI
tests
documentation
```

The core application did not require playlist-level ingestion.

### Fix

Playlist ingestion was removed.

The unit of ingestion is now a single video.

This keeps the core architecture easier to understand and validate.

---

# Failure 4 — LLM-Based Query Rewriting Added Another Failure Point

An LLM-based rewriter means:

```text
Question
   ↓
LLM call
   ↓
Rewritten question
   ↓
Retrieval
   ↓
LLM call
   ↓
Answer
```

That means:

* additional latency
* additional API usage
* another rate-limit surface
* another network failure point
* another source of nondeterminism

### Fix

Query rewriting was changed to a lightweight heuristic implementation.

The system still handles conversational follow-ups while removing a second LLM dependency from the retrieval path.

---

# Failure 5 — LLM-as-a-Judge Made Evaluation Dependent on Another LLM

An LLM-based judge introduces another external dependency into evaluation.

That can make evaluation:

```text
non-deterministic
rate-limit sensitive
slower
harder to reproduce
```

### Fix

The evaluation system was simplified to a deterministic heuristic judge.

It measures:

```text
answer correctness
faithfulness
context relevance
citation accuracy
refusal correctness
```

The judge is intended as a regression signal.

It is not treated as a replacement for human evaluation.

---

# Failure 6 — Cross-Encoder Could Silently Fall Back

A particularly dangerous production failure is a system that appears healthy but silently replaces a required component with a weaker fallback.

For example:

```text
Expected:
Cross Encoder

Actually running:
Heuristic fallback
```

The application may still produce answers, making the problem difficult to notice.

### Fix

Production reranking explicitly disables silent fallback:

```python
get_reranker(
    settings.reranker_enabled,
    settings.reranker_model,
    allow_fallback=False,
)
```

The deterministic heuristic reranker remains available for tests/offline infrastructure.

Production behavior is therefore explicit.

---

# Failure 7 — Reranking Could Overrule Useful Stage-1 Ranking

A second-stage reranker is powerful, but blindly discarding stage-1 ranking information can create unstable ordering.

### Fix

The final score combines:

```text
cross-encoder relevance
+
small stage-1 ranking prior
```

The cross-encoder remains dominant.

A regression test verifies that a strong stage-1 candidate is not unnecessarily discarded.

---

# Failure 8 — LLM Rate Limits Looked Like Model Refusals

During evaluation, the LLM provider returned HTTP `429` responses.

The generation layer returned a refusal-style answer after the generation failed.

A naive evaluator could then classify:

```text
API failure
```

as:

```text
model refusal
```

That produces incorrect evaluation metrics.

### Fix

The RAG engine now tracks generation failures explicitly.

The evaluator distinguishes:

```text
generation_status = ok
```

from:

```text
generation_status = error
```

Generation failures are excluded from generation-quality metrics rather than being counted as model behavior.

This is an important production evaluation distinction.

---

# Failure 9 — Evaluation Reports Failed on Windows Encoding

Evaluation reports initially encountered a Windows encoding issue when Unicode characters were written using the default system encoding.

### Fix

Reports are explicitly written as UTF-8:

```python
mpath.write_text(
    "\n".join(md),
    encoding="utf-8",
)
```

This makes generated Markdown reports portable across environments.

---

# Failure 10 — Docker Build and Runtime Are Different Problems

A project working locally does not guarantee that it works inside Docker.

The Docker environment has to reproduce:

```text
Python dependencies
application files
backend
frontend
model infrastructure
configuration
runtime behavior
```

### Fix

The project was validated through both:

```text
docker compose build
```

and:

```text
docker compose up -d
```

The final build successfully produced:

```text
✔ Image youtuberag-backend Built
✔ Image youtuberag-frontend Built
```

The runtime successfully started:

```text
✔ ytrag-backend Healthy
✔ ytrag-frontend Started
```

The health endpoint returned:

```text
status : ok
```

---

# Failure 11 — "Container Is Running" Is Not Enough

A container being `Up` does not prove the application initialized correctly.

The backend exposes:

```text
GET /health
```

The final Docker validation returned:

```text
status          : ok
embedding_model : BAAI/bge-small-en-v1.5
vectorstore     : chroma
llm             : openai-compatible:openai/gpt-oss-120b
reranker        : cross-encoder
videos          : {}
chunks          : 0
```

The empty video/chunk state was expected for the newly created Docker volume.

The important result was:

```text
status = ok
```

with the configured pipeline successfully initialized.

---

# Failure 12 — Feature Removal Left Stale Configuration

Removing a feature from a RAG system is not just deleting its main implementation.

A removed feature can leave behind:

```text
configuration fields
environment variables
imports
routes
schemas
tests
Docker settings
README references
```

A stale configuration option creates a false mental model of the system.

### Fix

After major removals, the repository was searched for stale references.

This included checking for:

```text
multimodal
playlist
chunk_strategy
SentenceChunker
FixedTokenChunker
LLMJudge
lazy_llm
```

The goal was to ensure that removed features were actually removed end-to-end.

---

# Architectural Decisions

## Why Hybrid Retrieval?

Because semantic and lexical retrieval fail differently.

```text
Vector retrieval
→ strong semantic similarity
→ weaker on exact terminology

BM25
→ strong lexical matching
→ weaker on paraphrases

Hybrid
→ combines both
```

---

## Why RRF?

Vector similarity and BM25 scores are not directly comparable.

RRF operates on ranking position.

That means there is no need to invent an arbitrary score normalization scheme.

---

## Why Cross-Encoder Reranking?

The first-stage retriever needs to be fast.

The cross-encoder can be slower because it only sees a small candidate set.

Therefore:

```text
fast retrieval → broad candidate set
slow reranking → narrow final set
```

This is the retrieve-wide/rerank-narrow architecture.

---

## Why BGE-small?

The project needs a model that is practical on a developer machine and inside Docker.

BGE-small provides a useful quality/resource trade-off.

A larger embedding model can be introduced through configuration rather than architectural changes.

---

## Why Chroma?

The project does not require a separate vector database server for the current scale.

Chroma provides:

* persistent local storage
* similarity search
* metadata filtering
* simple deployment

The vector-store abstraction also keeps the system open to future backends.

---

## Why Not LangChain or LlamaIndex?

The project deliberately keeps the major RAG stages explicit.

Instead of hiding:

```text
retrieval
fusion
reranking
prompt construction
citation handling
```

inside a large framework, the system implements its own small interfaces around these boundaries.

This makes the actual RAG behavior inspectable and testable.

The trade-off is that some infrastructure has to be implemented manually.

For this project, that trade-off is intentional.

---

## Why a Raw OpenAI-Compatible Client?

The generation layer uses an OpenAI-compatible HTTP interface rather than tightly coupling the application to a vendor SDK.

The same abstraction can point to different providers/endpoints.

This keeps:

```text
RAG engine
```

separate from:

```text
LLM provider
```

---

## Why a Deterministic Offline Path?

A RAG project should not require:

```text
API key
+
network
+
large model download
```

for every test.

The project therefore provides deterministic infrastructure such as:

```text
HashingEmbedder
MemoryVectorStore
MockLLM
HeuristicReranker
HeuristicJudge
```

These are primarily for testing and evaluation.

Production uses the real retrieval and generation components.

---

# Evaluation

Evaluation is treated as part of the application rather than a notebook that is run once.

Run:

```bash
python scripts/evaluate.py --help
```

Supported dimensions include:

```text
Embedding:
    bge
    hashing

Retrieval:
    vector
    bm25
    hybrid

Reranking:
    enabled
    disabled

Storage:
    chroma
    memory
```

---

# Current Retrieval Evaluation

The current demo evaluation contains:

```text
17 questions
```

The latest hybrid + cross-encoder run produced:

| Metric       |    Stage 1 |    Stage 2 |
| ------------ | ---------: | ---------: |
| MRR          |     0.9000 | **0.9222** |
| Recall@1     |     0.3800 | **0.3933** |
| Recall@3     |     0.6978 | **0.7311** |
| Recall@5     | **0.8200** |     0.8156 |
| Recall@10    | **0.9200** |     0.8156 |
| Precision@1  |     0.8000 | **0.8667** |
| Precision@3  |     0.5556 | **0.5778** |
| Precision@5  |     0.4400 |     0.4133 |
| Precision@10 |     0.2600 |     0.2067 |
| HitRate@1    |     0.8000 | **1.0000** |
| HitRate@3    |     1.0000 |     1.0000 |
| HitRate@5    |     1.0000 |     1.0000 |
| HitRate@10   |     1.0000 |     1.0000 |
| nDCG@1       |     0.8000 | **0.8667** |
| nDCG@3       |     0.8090 | **0.8620** |
| nDCG@5       |     0.8530 | **0.9283** |
| nDCG@10      |     0.9053 | **0.9283** |

These numbers come from the current demo evaluation dataset.

They should be interpreted as:

> **Regression measurements for this system and dataset, not a general benchmark for RAG.**

---

# Generation Evaluation

Generation evaluation tracks:

```text
answer correctness
faithfulness
context relevance
citation accuracy
refusal correctness
```

However, generation metrics are sensitive to external LLM availability.

During development, the LLM provider returned `429` rate-limit responses for some evaluation questions.

Those failures are now explicitly separated from model behavior.

Therefore:

```text
LLM API failure
≠
model refusal
```

This distinction is preserved in the evaluation reports.

---

# Debug Mode

Debug mode is one of the most useful development features in the project.

Instead of seeing only:

```text
Answer
```

the system can expose the retrieval path:

```text
original query
      ↓
rewritten query
      ↓
retrieval mode
      ↓
vector candidates
      ↓
BM25 candidates
      ↓
RRF scores
      ↓
reranked candidates
      ↓
final context
      ↓
LLM
      ↓
timings
```

A typical debug trace contains information such as:

```text
[ctx] [1:05:29–1:06:00]
vec=0.758354
bm25=8.1409
fused=0.032002
rerank=0.820784
```

and timing information:

```text
rewrite
embed + vector
bm25
rerank
llm
total
```

This makes it possible to answer:

> "Why did the system retrieve this chunk?"

instead of guessing from the final answer.

CLI:

```text
/debug on
```

or:

```text
/debug off
```

---

# API

The backend is implemented using FastAPI.

Interactive documentation:

```text
http://localhost:8000/docs
```

## Core Endpoints

| Method   | Endpoint                     | Purpose                       |
| -------- | ---------------------------- | ----------------------------- |
| `POST`   | `/videos/process`            | Ingest a YouTube video        |
| `GET`    | `/videos`                    | List ingested videos          |
| `GET`    | `/videos/{video_id}`         | Get video metadata            |
| `GET`    | `/videos/{video_id}/sources` | Get stored timestamped chunks |
| `DELETE` | `/videos/{video_id}`         | Remove a video                |
| `POST`   | `/chat`                      | Ask a grounded question       |
| `GET`    | `/health`                    | Check application health      |

---

## Chat Request

Conceptually:

```json
{
  "query": "Why does the model use embeddings?",
  "video_id": "TCH_1BHY58I",
  "conversation_id": null,
  "top_k": 15,
  "top_n": 4,
  "debug": true
}
```

---

## Chat Response

Conceptually:

```json
{
  "answer": "The model uses embeddings to represent characters...",
  "answer_markdown": "The model uses embeddings...[1:05:29](...)",
  "sources": [
    {
      "video_id": "TCH_1BHY58I",
      "start_time": 3929.0,
      "end_time": 3960.0,
      "score": 0.82,
      "text": "..."
    }
  ],
  "cited_indices": [0],
  "conversation_id": "..."
}
```

---

# Configuration

Important configuration values include:

| Configuration               | Default                                |
| --------------------------- | -------------------------------------- |
| `EMBEDDING_PROVIDER`        | `sentence-transformers`                |
| `EMBEDDING_MODEL`           | `BAAI/bge-small-en-v1.5`               |
| `EMBEDDING_DIM`             | `384`                                  |
| `VECTORSTORE_BACKEND`       | `chroma`                               |
| `RETRIEVAL_MODE`            | `hybrid`                               |
| `RETRIEVAL_TOP_K`           | `15`                                   |
| `RERANK_TOP_N`              | `4`                                    |
| `RERANKER_ENABLED`          | `true`                                 |
| `RERANKER_MODEL`            | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HYBRID_VECTOR_WEIGHT`      | `1.0`                                  |
| `HYBRID_BM25_WEIGHT`        | `1.0`                                  |
| `RRF_K`                     | `60`                                   |
| `CHUNK_TARGET_TOKENS`       | `160`                                  |
| `CHUNK_MAX_TOKENS`          | `240`                                  |
| `CHUNK_OVERLAP_TOKENS`      | `40`                                   |
| `LLM_PROVIDER`              | `groq`                                 |
| `LLM_MODEL`                 | `openai/gpt-oss-120b`                  |
| `LLM_TEMPERATURE`           | `0.1`                                  |
| `LLM_MAX_TOKENS`            | `1024`                                 |
| `LLM_TIMEOUT_S`             | `60`                                   |
| `REWRITER_ENABLED`          | `true`                                 |
| `HISTORY_TURNS_FOR_REWRITE` | `4`                                    |
| `CONVERSATION_TTL_MINUTES`  | `120`                                  |

---

# Project Structure

```text
YouTube-RAG/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── evaluation/
│   │   ├── judges.py
│   │   └── runner.py
│   │
│   ├── generation/
│   │   ├── base.py
│   │   ├── openai_compat.py
│   │   └── rag_engine.py
│   │
│   ├── ingestion/
│   │   ├── chunking.py
│   │   ├── service.py
│   │   └── youtube.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   └── vectorstore.py
│   │
│   └── ...
│
├── frontend/
│   └── streamlit_app.py
│
├── scripts/
│   ├── chat.py
│   └── evaluate.py
│
├── tests/
│
├── data/
│   ├── eval/
│   └── fixtures/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

# Testing

The project uses `pytest`.

Run the complete suite:

```bash
pytest -q
```

The test suite covers major areas including:

```text
chunking
retrieval
BM25
RRF
reranking
RAG engine
query rewriting
evaluation
YouTube utilities
configuration
API behavior
```

The full test suite has been repeatedly run after architectural changes.

The important testing principle was:

```text
Change architecture
      ↓
Run focused tests
      ↓
Run full suite
      ↓
Only then continue
```

This was especially important when removing features such as:

```text
multimodal processing
playlist ingestion
alternative chunking strategies
LLM-based evaluation
LLM-based query rewriting
```

---

# Docker

The project contains Docker support for both backend and frontend.

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up -d
```

Check:

```bash
docker compose ps
```

The final Docker validation produced:

```text
✔ Image youtuberag-backend Built
✔ Image youtuberag-frontend Built
```

and:

```text
ytrag-backend    Healthy
ytrag-frontend   Started
```

The backend health check returned:

```text
status          : ok
embedding_model : BAAI/bge-small-en-v1.5
vectorstore     : chroma
llm             : openai-compatible:openai/gpt-oss-120b
reranker        : cross-encoder
```

The Docker deployment therefore validates both:

```text
image creation
```

and:

```text
runtime initialization
```

---

# Docker Architecture

```text
                    Docker Compose
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      ┌──────────────┐       ┌──────────────┐
      │   Backend    │       │   Frontend   │
      │              │       │              │
      │   FastAPI    │◄──────│  Streamlit   │
      │   :8000      │ HTTP  │   :8501      │
      └──────┬───────┘       └──────────────┘
             │
             ▼
      RAG Pipeline
             │
      ┌──────┴────────┐
      │               │
      ▼               ▼
   Chroma          HF Cache
```

The Docker setup uses a persistent Hugging Face cache volume so model files do not need to be downloaded repeatedly.

---

# Quickstart

## Option A — Docker

Create `.env` from the example:

```bash
cp .env.example .env
```

Configure your LLM credentials.

Then:

```bash
docker compose up --build
```

Open:

```text
Frontend:
http://localhost:8501

API:
http://localhost:8000

API docs:
http://localhost:8000/docs
```

---

## Option B — Local

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment configuration:

```bash
cp .env.example .env
```

Start FastAPI:

```bash
uvicorn app.api.main:app --reload
```

Start Streamlit in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

---

# Chat CLI

The repository also provides an interactive CLI.

For an already ingested video:

```bash
python scripts/chat.py <video_id>
```

Cross-video:

```bash
python scripts/chat.py --all
```

The CLI supports:

```text
/debug on
/debug off
/video <video_id>
/video all
/quit
```

---

# Evaluation Commands

Hybrid retrieval:

```bash
python scripts/evaluate.py \
    --embedding bge \
    --mode hybrid \
    --store chroma
```

Vector-only:

```bash
python scripts/evaluate.py \
    --embedding bge \
    --mode vector \
    --store chroma
```

BM25:

```bash
python scripts/evaluate.py \
    --embedding bge \
    --mode bm25 \
    --store chroma
```

Without reranking:

```bash
python scripts/evaluate.py \
    --embedding bge \
    --mode hybrid \
    --no-rerank \
    --store chroma
```

---

# Example

After ingesting a technical lecture:

```text
User:
What is the main idea behind using an MLP in the makemore model?
```

The system:

```text
1. Rewrites the query if required
2. Embeds the question
3. Runs vector retrieval
4. Runs BM25
5. Fuses the rankings with RRF
6. Reranks the candidates
7. Selects the final context
8. Generates a grounded answer
9. Maps citations to video timestamps
```

A resulting answer can contain:

```text
The MLP is used to predict the next character by learning
a richer representation than the earlier simpler model.

[0:01]
[2:18]
[52:46]
```

The timestamps are generated from the retrieved transcript chunks.

---

# Engineering Lessons

## 1. RAG quality starts with retrieval

If the correct information never reaches the context window, the LLM cannot reliably use it.

Therefore:

```text
Better retrieval
        ↓
Better context
        ↓
Better grounded generation
```

---

## 2. Hybrid retrieval is useful because retrieval failures are different

Semantic retrieval and lexical retrieval fail in different ways.

Using both gives the system more opportunities to recover the correct chunk.

---

## 3. Reranking is valuable because retrieval is not the final ranking

The first-stage retriever is optimized for recall and efficiency.

The cross-encoder can spend more computation on a much smaller candidate set.

---

## 4. Production failures must be separated from model behavior

An API timeout or `429` is not a hallucination.

A provider outage is not a refusal.

A failed generation request should not silently become a model-quality metric.

---

## 5. Observability changes debugging

Without debug information:

```text
"Why did the system answer this?"
```

is difficult to answer.

With:

```text
vector score
BM25 score
RRF score
rerank score
timestamps
stage timings
```

the retrieval path becomes inspectable.

---

## 6. Removing complexity is also engineering

A feature should justify its maintenance cost.

This project deliberately removed:

```text
unused multimodal infrastructure
playlist ingestion
alternative chunking strategies
LLM-based query rewriting
LLM-as-a-judge
```

The result is a smaller and more coherent system.

---

## 7. Evaluation should be reproducible

A useful evaluation system should be able to answer:

```text
What configuration produced this number?
Which question produced the failure?
Which chunk was retrieved?
Which source was cited?
Did generation actually succeed?
```

The evaluation reports preserve per-question details so aggregate numbers do not hide individual failures.

---

# Limitations

The current system has several known limitations.

### Transcript quality

YouTube transcripts can contain:

* missing punctuation
* ASR mistakes
* repeated captions
* incomplete sentences

Chunking quality therefore depends partly on transcript quality.

---

### YouTube access

Some environments can be blocked by YouTube when requesting transcripts.

The ingestion layer supports fallback behavior, but external access restrictions cannot always be eliminated by application code.

---

### Evaluation size

The current evaluation dataset is small.

The retrieval metrics should therefore be interpreted as:

```text
system regression measurements
```

rather than claims of general RAG performance.

---

### Heuristic evaluation

The deterministic judge is lexical.

A semantically correct paraphrase can therefore receive a lower score than expected.

Human evaluation or a separate model-based evaluation layer would be useful for a larger benchmark.

---

### LLM dependency

Real generation requires an available LLM provider.

The project includes mock/offline infrastructure for testing, but the mock generator is not intended to represent production generation quality.

---

### CPU inference

The default embedding and reranking configuration is designed to remain practical on a developer machine.

GPU inference could reduce latency for larger workloads.

---

# Future Work

Potential improvements include:

### Retrieval

* larger evaluation datasets
* query expansion
* better retrieval weighting
* multilingual embeddings
* more advanced rerankers

### Generation

* streaming responses
* stronger citation verification
* structured answers
* answer confidence estimation

### Evaluation

* human-labeled real-video QA datasets
* larger regression suites
* retrieval error categorization
* automated evaluation dashboards
* comparison across embedding models

### Infrastructure

* pgvector
* Qdrant
* Redis-backed conversation memory
* background ingestion workers
* caching
* authentication
* rate limiting
* observability
* distributed deployment

These are future directions, not requirements of the current system.

---

# Security

Never commit API credentials.

Use:

```text
.env
```

for local secrets.

Only commit:

```text
.env.example
```

with placeholder values.

If an API key is accidentally exposed:

```text
1. Revoke/rotate it immediately.
2. Remove it from the repository.
3. Check Git history.
4. Replace it with a new key.
```

Secrets should never be baked into Docker images.

---

# Reproducibility Checklist

Run the full test suite:

```bash
pytest -q
```

Run the evaluation:

```bash
python scripts/evaluate.py \
    --embedding bge \
    --mode hybrid \
    --store chroma
```

Build Docker:

```bash
docker compose build
```

Run Docker:

```bash
docker compose up -d
```

Check containers:

```bash
docker compose ps
```

Check backend:

```bash
Invoke-RestMethod http://localhost:8000/health
```

The goal is that the same repository can be:

```text
tested
evaluated
built
deployed
```

without relying on undocumented local state.

---

# Final Architecture Summary

```text
                         ┌──────────────────────┐
                         │      YouTube URL     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Transcript Providers │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Cleaning             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Timestamp Chunking   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                    BGE Embeddings           BM25
                         │                     │
                         ▼                     ▼
                      Chroma               BM25 Index
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           Reciprocal Rank
                              Fusion
                                    │
                                    ▼
                           Top 15 Candidates
                                    │
                                    ▼
                         Cross-Encoder Reranker
                                    │
                                    ▼
                            Top 4 Context
                                    │
                                    ▼
                         Grounded LLM
                                    │
                                    ▼
                         Citation Renderer
                                    │
                                    ▼
                     Answer + YouTube Timestamps
```

---

# The Core Idea

The most important lesson from this project is simple:

> **Building a RAG demo is easy. Building one that you can inspect, evaluate, debug, deploy, and recover when individual components fail is the actual engineering problem.**

This project therefore focuses not only on:

```text
"Can the LLM answer?"
```

but also on:

```text
Where did the answer come from?

Why was this chunk retrieved?

Why was another chunk rejected?

Did reranking improve the result?

Was the query rewritten?

Was the generation successful?

Was the citation actually used?

What happens when the LLM fails?

Can the system be evaluated without the LLM?

Can the application run inside Docker?

Can the entire pipeline be tested deterministically?
```

That is the problem this repository is designed to explore.

```
```
