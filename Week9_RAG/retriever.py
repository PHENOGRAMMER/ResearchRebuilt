# retriever.py
# Week 9 — #ResearchRebuilt
# Retrieval-Augmented Generation (Lewis et al., NeurIPS 2020)
# arxiv: 2005.11401
#
# This module reproduces the paper's retrieval component.
# Original: DPR (Dense Passage Retrieval) with BERT bi-encoders + FAISS over Wikipedia.
# Here:     sentence-transformers for dense embeddings + numpy MIPS (Maximum Inner
#           Product Search) over a hand-curated knowledge base of 50 ML/AI documents.
#
# The retrieval math is faithful to the paper:
#   p_η(z|x) ∝ exp( d(z)ᵀ q(x) )
#   where d(z) = encoder(document), q(x) = encoder(query)
#   top-k retrieved via dot-product similarity (MIPS)

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Knowledge base: 50 ML/AI documents (the "Wikipedia" of this demo) ─────────
# Deliberately includes outdated facts, gaps, and near-misses so the
# hallucination / retrieval contrast is clearly visible in results.

DOCUMENTS = [
    # Transformers & Attention
    "The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replaced recurrent networks with self-attention. It consists of stacked encoder and decoder layers, each containing multi-head attention and feed-forward sublayers.",
    "Scaled dot-product attention computes similarity between queries Q and keys K as softmax(QKᵀ/√d_k)V. The scaling by √d_k prevents vanishing gradients in high-dimensional spaces. Time complexity is O(n²d) with respect to sequence length n.",
    "BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) pre-trains a transformer encoder with masked language modelling and next-sentence prediction on BookCorpus and English Wikipedia (3.3B words).",
    "GPT-3 (Brown et al., 2020) has 175 billion parameters and was trained on ~300 billion tokens from Common Crawl, WebText, Books1, Books2, and Wikipedia. It demonstrated few-shot learning without gradient updates.",
    "Multi-head attention uses h parallel attention heads. With d_model=768 and h=12, each head has d_k = d_model / h = 64 dimensions. The total parameter count of the MHA block is 4 × d_model² regardless of head count.",

    # RAG & Retrieval
    "Retrieval-Augmented Generation (Lewis et al., NeurIPS 2020) combines a DPR retriever with a BART-large generator. The retriever fetches top-k documents from a 21-million-document Wikipedia index, and the generator conditions on query + retrieved context.",
    "Dense Passage Retrieval (DPR, Karpukhin et al., 2020) uses two BERT_BASE encoders — one for queries, one for documents — and retrieves via Maximum Inner Product Search (MIPS). It outperforms BM25 by a large margin on open-domain QA.",
    "RAG-Sequence uses the same retrieved documents for the entire generated sequence. RAG-Token can attend to different documents at each generation step, marginalizing over passages per token. RAG-Token outperforms RAG-Sequence on factual short-answer tasks.",
    "In the original RAG paper, the knowledge base is a December 2018 Wikipedia dump split into 100-word chunks, yielding 21 million documents. The FAISS index stores 768-dimensional vectors requiring approximately 15.3 billion float values.",
    "RAG achieved state-of-the-art results on Natural Questions (44.5 EM), TriviaQA (56.8 EM), WebQuestions (45.2 EM), and CuratedTrec (52.2 EM) without task-specific pre-training, outperforming T5-11B despite using only 626M trainable parameters.",
    "On the Natural Questions benchmark, closed-book baselines scored: T5-large 28.9 EM, T5-11B 34.5 EM, T5-11B with salient span masking (SSM) 36.6 EM. RAG-Sequence achieved 44.5 EM — outperforming T5-11B by nearly 10 points using only 626M parameters vs T5-11B's 11 billion. This makes RAG roughly 17x more parameter-efficient on NQ.",
    "RAG's index hot-swapping experiment showed 70% accuracy with a 2016 Wikipedia index on 2016 world leaders, and 68% with a 2018 index on 2018 leaders. Mismatched indices dropped accuracy to 4–12%, proving knowledge updates need only index replacement.",
    "Human evaluation of RAG on Jeopardy question generation showed RAG-Token was judged more factual in 42.7% of cases vs BART's 7.1%, and more specific in 37.4% vs 16.8% for BART.",

    # Fine-tuning & Adaptation
    "LoRA (Low-Rank Adaptation, Hu et al., 2022) fine-tunes large models by injecting trainable low-rank matrices A and B alongside frozen weights W. The update is ΔW = BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with r ≪ min(d,k).",
    "With LoRA rank r=8 on a 7B parameter model, only approximately 4-8 million parameters are trainable — roughly 0.1% of total parameters — while achieving comparable performance to full fine-tuning on many tasks.",
    "Instruction tuning (FLAN, Wei et al., 2021) fine-tunes language models on a mixture of NLP tasks formatted as instructions. Models instruction-tuned on 60+ tasks generalise to unseen tasks in a zero-shot manner.",
    "RLHF (Reinforcement Learning from Human Feedback) trains a reward model on human preference data, then fine-tunes the LM with PPO to maximise reward. Used by InstructGPT, ChatGPT, and Claude to align outputs with human intent.",

    # Inference & Reasoning
    "Chain-of-Thought prompting (Wei et al., 2022) improves reasoning by prompting models to produce intermediate reasoning steps. 'Let's think step by step' (Kojima et al., 2022) achieves CoT without few-shot examples.",
    "Self-Consistency (Wang et al., 2022) samples multiple reasoning paths at temperature > 0 and takes a majority vote over final answers. It improves CoT accuracy by 10-20% on arithmetic and commonsense benchmarks.",
    "The s1 paper (Muennighoff et al., 2025) showed that appending 'Wait' to a partial reasoning chain forces continued generation, improving accuracy on hard math problems without any additional training.",
    "Test-time compute scaling (Snell et al., 2024) shows that compute-optimal allocation depends on problem difficulty. Easy problems benefit from Best-of-N sampling; hard problems benefit from sequential beam search with a process reward model.",
    "TERMINATOR (Nagle et al., 2026) introduces a classifier trained on 'first answer positions' in reasoning chains to predict optimal exit points. It reduces CoT length by 14-55% with near-zero accuracy loss and 2× inference speedup.",

    # Training & Scaling
    "The Chinchilla scaling laws (Hoffmann et al., 2022) showed that compute-optimal training requires equal scaling of model parameters and training tokens. Chinchilla-70B, trained on 1.4T tokens, outperformed Gopher-280B.",
    "Flash Attention (Dao et al., 2022) rewrites the attention kernel to use SRAM instead of HBM, reducing memory bandwidth from O(n²) to O(n). It achieves 2-4× speedup on long sequences without approximation.",
    "Mixture of Experts (MoE) models route each token to a subset of expert feed-forward networks. Mistral 8x7B uses 8 experts with top-2 routing — 46.7B total parameters but only 12.9B active per token.",
    "Pre-training large language models on next-token prediction requires enormous compute. GPT-4's training run is estimated at ~2×10²⁵ FLOPs. A100 GPUs deliver ~312 TFLOPS of FP16, so training takes thousands of GPU-days.",

    # Memory & Vector Databases
    "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search. It supports exact L2, inner product, and approximate search via IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World).",
    "A FAISS flat index storing 1 million 768-dimensional float32 vectors requires 1,000,000 × 768 × 4 bytes = 3,072,000,000 bytes ≈ 3.07 GB (SI) or 2.86 GiB (binary). This is the minimum for exact search.",
    "Chroma, Pinecone, Weaviate, and Qdrant are vector databases designed for production RAG systems. They support metadata filtering, hybrid BM25+dense search, and horizontal scaling — features not in FAISS.",
    "Chunking strategy significantly impacts RAG quality. Larger chunks (512-1024 tokens) preserve context but reduce retrieval precision. Smaller chunks (128-256 tokens) improve retrieval precision but may lose context at generation.",
    "Hybrid search combines BM25 (sparse, keyword-based) with dense retrieval (semantic). Reciprocal Rank Fusion (RRF) merges sparse and dense result sets. Many production RAG systems use hybrid over pure dense retrieval.",

    # Hallucination & Factuality
    "Hallucination in LLMs refers to generating plausible-sounding but factually incorrect statements. Closed-book models hallucinate because they must reconstruct facts from compressed parametric memory with limited fidelity.",
    "TruthfulQA (Lin et al., 2021) benchmarks model truthfulness on questions humans often answer incorrectly due to misconceptions. GPT-3 achieved only 58% truthfulness; retrieval-augmented models significantly outperform pure parametric models.",
    "Groundedness in RAG measures whether the generated answer is supported by retrieved passages. RAGAS (Retrieval Augmented Generation Assessment) framework measures faithfulness, answer relevance, context precision, and context recall.",
    "Self-RAG (Asai et al., 2023) trains models to retrieve on-demand using special reflection tokens: [Retrieve], [IsRel], [IsSup], [IsUse]. This allows the model to decide when retrieval is needed rather than always retrieving.",

    # Evaluation
    "Exact Match (EM) measures whether the predicted answer exactly matches the gold answer after normalisation (lowercase, strip punctuation, strip articles). It is strict: '42' and 'forty-two' would not match.",
    "BLEU score measures n-gram precision overlap between generated and reference text. BLEU-1 through BLEU-4 measure 1- through 4-gram precision. It correlates poorly with human judgement for open-ended generation.",
    "BERTScore computes token-level similarity between generated and reference text using contextual BERT embeddings. It correlates better with human judgements than BLEU on abstractive summarisation and generation tasks.",
    "Perplexity measures how well a language model predicts a sequence. Lower perplexity means better prediction. A model with perplexity 10 is equally uncertain between 10 choices per token on average.",

    # ML Systems
    "Loading a 7-billion-parameter model in float32 requires 7 × 10⁹ × 4 bytes = 28 GB of GPU VRAM just for weights. In bfloat16 this halves to 14 GB. Training additionally requires optimizer states, gradients, and activations.",
    "Quantisation reduces model size by representing weights in lower precision. INT8 quantisation halves memory vs float16. GPTQ, AWQ, and bitsandbytes (NF4) are common LLM quantisation methods.",
    "KV-cache stores past key-value pairs to avoid recomputation during autoregressive generation. Memory grows as O(n × num_layers × d_model). For long contexts this dominates VRAM usage over model weights.",
    "Speculative decoding uses a small draft model to propose multiple tokens, then the large model verifies them in one forward pass. Speedup of 2-3× with identical outputs to standard greedy decoding.",

    # Agents & Multi-Agent
    "ReAct (Yao et al., 2022) interleaves reasoning and acting: the model generates a Thought, takes an Action (tool call), observes the Result, and continues until it reaches an Answer. It forms the backbone of most LLM agents.",
    "LangChain and LlamaIndex are frameworks for building RAG pipelines and LLM agents. LangChain emphasises composable chains; LlamaIndex specialises in document indexing, retrieval, and query engines.",
    "HERA (Hierarchical Emergent Reasoning Architecture) is a multi-agent framework that uses specialised sub-agents coordinated by an orchestrator. It improves performance on multi-hop question answering over flat RAG baselines.",
    "AutoGen (Wu et al., 2023) enables multi-agent conversations where LLM agents collaborate, critique, and iterate on solutions. Agents can be assigned roles (Planner, Executor, Critic) and communicate via natural language.",
    "Tool-use in LLM agents allows models to call external APIs, run code, search the web, or query databases. The model generates a structured action (function call), receives a result, and incorporates it into further reasoning.",

    # Datasets
    "Natural Questions (Kwiatkowski et al., 2019) contains 307,373 training questions derived from Google search queries, with Wikipedia passages as context. It is a standard benchmark for open-domain QA.",
    "MS-MARCO (Nguyen et al., 2016) is a reading comprehension dataset with 1 million real Bing queries and human-generated answers. It is used to evaluate both retrieval (ranking) and generation (abstractive QA) models.",
    "HotpotQA (Yang et al., 2018) requires multi-hop reasoning over multiple Wikipedia passages. Models must identify supporting facts across two or more documents to answer correctly.",
    "FEVER (Thorne et al., 2018) is a fact verification dataset with 185,455 claims derived from Wikipedia. Models must classify claims as SUPPORTED, REFUTED, or NOT ENOUGH INFO and provide evidence.",
    "TriviaQA (Joshi et al., 2017) contains 95,000 question-answer pairs authored by trivia enthusiasts, with evidence documents from Wikipedia and the web. It tests a wide range of factual knowledge.",
]

# ── Encoder ───────────────────────────────────────────────────────────────────

_encoder = None
_doc_embeddings = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        # Same model family as DPR's BERT encoders — sentence-transformers wraps BERT
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

def _get_doc_embeddings():
    global _doc_embeddings
    if _doc_embeddings is None:
        enc = _get_encoder()
        _doc_embeddings = enc.encode(DOCUMENTS, normalize_embeddings=True, show_progress_bar=False)
    return _doc_embeddings


# ── Retrieval strategies ───────────────────────────────────────────────────────

def retrieve_dense(query: str, k: int = 3) -> list[dict]:
    """
    Dense retrieval — faithful to DPR in the RAG paper.

    Computes p_η(z|x) ∝ exp(d(z)ᵀ q(x)) for all documents,
    returns top-k by dot-product similarity (MIPS).
    With normalised embeddings, dot product = cosine similarity.
    """
    enc = _get_encoder()
    q_emb = enc.encode([query], normalize_embeddings=True)[0]
    d_embs = _get_doc_embeddings()

    # MIPS: d(z)ᵀ q(x) for all z
    scores = d_embs @ q_emb

    top_k_idx = np.argsort(scores)[::-1][:k]
    return [
        {"text": DOCUMENTS[i], "score": float(scores[i]), "index": int(i)}
        for i in top_k_idx
    ]


def retrieve_bm25_approx(query: str, k: int = 3) -> list[dict]:
    """
    Approximate BM25 (sparse keyword retrieval) — the baseline the paper compares against.
    We simulate BM25 with TF overlap scoring (no IDF) for zero-dependency reproducibility.
    """
    query_tokens = set(query.lower().split())
    scored = []
    for i, doc in enumerate(DOCUMENTS):
        doc_tokens = doc.lower().split()
        tf = sum(doc_tokens.count(t) for t in query_tokens)
        scored.append((tf, i))
    scored.sort(reverse=True)
    return [
        {"text": DOCUMENTS[i], "score": float(s), "index": i}
        for s, i in scored[:k]
    ]