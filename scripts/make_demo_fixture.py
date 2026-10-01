"""Generate the offline demo fixture: a SYNTHETIC transcript of a fictional
12-minute lecture on Transformers ("demo_lecture_v1", an 11-char valid id).

Why synthetic?
  * Reproducible: the demo + evaluation dataset work without network access
    and without depending on a real creator's captions (which could change
    or disappear).
  * Groundable: because we author the transcript, every evaluation question
    has a verifiable gold answer + gold timestamp — no fabricated metrics.

For a real-video demo, run `scripts/ingest.py <youtube-url>` instead.
"""
from __future__ import annotations

import json
from pathlib import Path

# (start_seconds, sentence) — durations are derived from the next start.
SCRIPT: list[tuple[float, str]] = [
    (0.0, "Hi everyone, and welcome back to the channel. Today we are going to talk about the Transformer architecture, and specifically about why attention replaced recurrence in sequence modeling."),
    (9.5, "This video is aimed at people who already know what a neural network is, and who have maybe heard of word embeddings, but who want to understand attention from the ground up."),
    (19.0, "Here is the plan. First, I will motivate the problem with recurrent networks. Then we will build self-attention step by step, including queries, keys, and values. After that, multi-head attention, positional encodings, and finally some practical training tips, including the learning rate schedule I recommend."),
    (36.5, "Let's start with the problem. A recurrent neural network processes a sequence one token at a time. At each step it updates a hidden state, and that hidden state is supposed to carry all the information from the past."),
    (50.0, "There are two big problems with this design. The first problem is sequential computation. Because step t depends on step t minus one, you cannot parallelize across time. On modern GPUs, that is a disaster for training speed."),
    (65.0, "The second problem is long-range dependencies. Information from the beginning of a sentence has to travel through every intermediate hidden state to reach the end. Gradients have to flow back along the same long path, and they tend to vanish or explode."),
    (83.0, "LSTMs and GRUs help with the gradient problem using gating, but they do not fix the sequential bottleneck. The path length between two tokens still grows linearly with distance."),
    (95.5, "Now here is the core idea of attention. Instead of forcing information through a single hidden state that marches forward in time, let every token look directly at every other token, in one step."),
    (110.0, "That is the key claim of the famous Attention Is All You Need paper from 2017: self-attention connects all positions in the sequence with a constant number of operations. The path length between any two tokens becomes one, instead of proportional to their distance."),
    (130.0, "So how does self-attention actually work? Each token produces three vectors. A query vector, which says what I am looking for. A key vector, which says what I contain. And a value vector, which is the actual information I would pass along if someone attends to me."),
    (152.0, "The attention score between two tokens is the dot product of the query of the first token with the key of the second token. High dot product means the query and the key are aligned, so those two tokens are relevant to each other."),
    (168.0, "We divide those dot products by the square root of the dimension of the key vectors. This scaling matters: without it, when the dimension is large, the dot products get big, the softmax saturates, and gradients become tiny."),
    (186.5, "Then we apply a softmax across all the keys, which turns raw scores into a probability distribution that sums to one. Finally, we take a weighted average of the value vectors, using those probabilities as weights."),
    (203.0, "So the output for each token is a mixture of the values of all tokens, where the mixture weights come from query-key compatibility. That is the entire mechanism. In matrix form, this is softmax of Q K transpose over root d_k, times V."),
    (224.0, "One important detail: in the decoder of the original Transformer, we use masked self-attention. The mask sets the scores of future positions to negative infinity before the softmax, so a token can never attend to tokens that come after it. That keeps autoregressive generation honest."),
    (247.0, "Now, a single attention head can only express one kind of relationship. But language has many kinds of relationships at once: syntax, coreference, semantic similarity."),
    (260.0, "That is why we use multi-head attention. We project the queries, keys, and values into several smaller subspaces, run attention independently in each, and concatenate the results. The original paper uses eight attention heads, each with a dimension of sixty-four, for a total model dimension of five hundred twelve."),
    (284.0, "Intuitively, different heads learn to attend to different things. Some heads track local syntax, some heads do coreference resolution, connecting a pronoun to the noun it refers to, and some heads look at positional neighbors."),
    (300.0, "But there is a catch. Self-attention is permutation invariant. If you shuffle the input tokens, the set of outputs is the same, just shuffled. Attention has no built-in notion of order."),
    (315.0, "So we inject position information explicitly with positional encodings. The original paper uses sine and cosine functions of different frequencies, added to the token embeddings. The nice property is that the encoding for position plus k is a linear function of the encoding for position, which makes it easy for the model to learn relative attention."),
    (340.0, "Later models mostly switched to learned positional embeddings, and today rotary position embeddings, or RoPE, are very popular, because they encode relative positions directly in the query-key dot product."),
    (355.0, "Let's zoom out and compare complexity. For a sequence of length n and representation dimension d, a recurrent layer costs n times d squared, sequentially. A self-attention layer costs n squared times d, but fully in parallel."),
    (373.0, "So when n is smaller than d, which is true for most sentences, attention is actually cheaper, and massively faster on GPUs because of parallelism. The quadratic cost in n is the real weakness of attention, and that is what flash attention and sparse attention methods try to mitigate."),
    (394.0, "Now let me give you some practical training tips, because this is where most people get burned. First, the optimizer: use Adam, or better, AdamW with decoupled weight decay."),
    (410.0, "Second, and this is the question I get most often, what learning rate should you use? For fine-tuning a small Transformer like BERT base, I recommend a learning rate of about two times ten to the minus five, up to five times ten to the minus five. For training from scratch, use a larger learning rate, around one times ten to the minus three, with warmup."),
    (440.0, "Warmup is essential. Start from zero and increase the learning rate linearly for the first few thousand steps, four thousand in the original paper, then decay it. Without warmup, the Adam second-moment estimates are garbage early on, and large early updates destabilize training."),
    (465.0, "Third, use the right regularization. The Transformer uses dropout of zero point one on attention weights and on sublayer outputs, plus label smoothing of zero point one during training. Label smoothing hurts perplexity a little but improves accuracy and BLEU."),
    (488.0, "Fourth, normalize before each sublayer, so-called pre-norm, rather than the original post-norm. Pre-norm is much more stable to train, especially without careful warmup."),
    (505.0, "Let me also answer a common misconception. People say attention is just a fancy way of computing a weighted average, so it cannot be that powerful. But remember, the weights themselves are learned from the content of the tokens, not fixed in advance. A convolution applies the same kernel everywhere. Attention computes a different, input-dependent kernel for every single position. That adaptivity is the source of its power."),
    (536.0, "How does this compare with convolutions? A convolutional layer has a fixed receptive field, so modeling distant interactions requires stacking many layers. Self-attention reaches any position in one layer. Convolutions are still useful for very long sequences, like audio waveforms, because they are linear in sequence length."),
    (560.0, "Let's talk about what happened after the original paper, very briefly, because it explains why you should care. Transformers scaled beautifully. When you add more data and more parameters, attention models keep improving, and that scaling property is exactly what produced large language models."),
    (582.0, "The same architecture that translates sentences now writes code, answers questions, and generates images, with some modifications. The encoder-decoder design of 2017 became the decoder-only design of GPT and the encoder-only design of BERT."),
    (601.0, "If you remember three things from this video, remember these. One: self-attention gives every token a direct, constant-length path to every other token, which fixes both the parallelization problem and the long-range dependency problem of RNNs."),
    (620.0, "Two: the mechanism is query-key-value. Queries find, keys advertise, values deliver. Scores are scaled dot products passed through a softmax."),
    (634.0, "Three: in practice, success is mostly about the training recipe. Adam with warmup, a learning rate around two times ten to the minus five for fine-tuning, pre-norm, dropout, and label smoothing."),
    (652.0, "If you want to go deeper, read the original paper, and then read Jay Alammar's Illustrated Transformer, which visualizes everything I described here. In the next video, we will implement a minimal Transformer from scratch in PyTorch, in about a hundred lines."),
    (672.0, "If this video helped you, like and subscribe, and I will see you in the next one. Thanks for watching."),
]


def build_fixture() -> dict:
    segments = []
    for i, (start, text) in enumerate(SCRIPT):
        end = SCRIPT[i + 1][0] if i + 1 < len(SCRIPT) else start + 8.0
        segments.append({"text": text, "start": start, "duration": round(end - start, 3)})
    return {
        "meta": {
            "video_id": "demoLctr001",
            "title": "Transformers & Attention, Explained From Scratch (synthetic demo lecture)",
            "channel": "Demo ML Lectures (synthetic)",
            "duration_s": round(SCRIPT[-1][0] + 8.0, 1),
            "language": "en",
        },
        "segments": segments,
        "synthetic": True,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "demo_lecture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_fixture(), indent=2))
    print(f"wrote {out} ({len(SCRIPT)} segments, ~{SCRIPT[-1][0]/60:.0f} min)")
