# ResearchRebuilt — Week 1: HERA (Multi-Agent RAG with Evolving Orchestration)

> Part of the **#ResearchRebuilt** series — weekly recreations of influential AI/ML research papers through implementation, experimentation, and analysis.

---

## 📄 Paper

**Experience as a Compass: Multi-Agent RAG with Evolving Orchestration and Agent Prompts**

**Authors:** Sha Li, Naren Ramakrishnan (Virginia Tech, 2026)

📄 Paper: https://arxiv.org/abs/2604.00901

---

# 🚀 Overview

Most Multi-Agent RAG systems execute a **fixed sequence of agents** for every query.

```
Retrieve
    ↓
Reason
    ↓
Answer
```

This works well for simple queries but struggles on complex multi-hop reasoning because:

- A mistake in one step propagates through the entire pipeline.
- Every query follows the same execution path.
- The system cannot improve from previous mistakes without retraining the LLM.

HERA introduces a different approach.

Instead of updating model weights, it learns from **structured natural language experiences** collected after every execution.

These experiences are stored in an **Experience Library**, allowing future queries to benefit from previous successes and failures.

This repository recreates the core architectural ideas proposed in the paper.

---

# 🧠 Key Idea

Instead of gradient-based learning:

```
Failure
      ↓
Analyze trajectory
      ↓
Extract insight
      ↓
Store in Experience Library
      ↓
Retrieve during future queries
```

The system improves its reasoning behavior **without changing model parameters**.

---

# 🏗️ Architecture

```
                    User Query
                         │
                         ▼
                Orchestrator Agent
                         │
        Chooses execution topology
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Query Decomposer                 Experience Library
        │                                 │
        ▼                                 │
   Retriever ─────────────────────────────┘
        │
        ▼
 Evidence Selector
        │
        ▼
 Conclusion Agent
        │
        ▼
                 Final Answer
```

---

# 🧩 Three-Layer Hierarchy

| Layer | Responsibility |
|--------|----------------|
| **Orchestrator** | Dynamically chooses which agents should execute and in what order. |
| **Experience Library** | Stores reusable reasoning experiences from previous executions. |
| **Execution Agents** | Specialized agents for decomposition, retrieval, evidence filtering, and answer generation. |

---

# ✅ Features Implemented

- Dynamic query decomposition
- Adaptive orchestration
- Experience Library
- Semantic retrieval using FAISS
- Wikipedia retrieval
- Multi-agent reasoning pipeline
- Structured experience storage

---

# ❌ Not Implemented

The original paper introduces **Role-Aware Prompt Evolution (RoPE)**.

RoPE automatically rewrites agent prompts after collecting multiple failed trajectories.

Since this requires a much larger execution history and prompt optimization loop, it is intentionally left out of this recreation.

The focus of this implementation is the **adaptive orchestration** and **Experience Library**, which form the architectural core of HERA.

---

# 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| LLM | llama-3.1-8b-instant (Groq) |
| Retrieval | Wikipedia API |
| Vector Database | FAISS |
| Embeddings | all-MiniLM-L6-v2 |
| Environment | python-dotenv |

---

# 📂 Project Structure

```text
Week1_HERA/
│
├── agents.py                 # Agent definitions & prompts
├── orchestrator.py           # Dynamic topology planner
├── retriever.py              # Wikipedia retrieval
├── experience_library.py     # Experience storage & retrieval
├── hera.py                   # Main pipeline
├── experience_library.json   # Generated experiences
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/PHENOGRAMMER/ResearchRebuilt.git

cd ResearchRebuilt
```

Create a virtual environment.

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Or install manually.

```bash
pip install groq wikipedia-api faiss-cpu sentence-transformers colorama python-dotenv
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from:

https://console.groq.com

---

# ▶️ Running

```bash
python hera.py
```

The pipeline executes three representative multi-hop reasoning queries.

You'll see:

- Query decomposition
- Retrieval
- Evidence selection
- Final reasoning
- Experience Library updates

After execution, `experience_library.json` will contain the accumulated reasoning experiences.

---

# 📊 Sample Results

| Query | Type | Result | Observation |
|------|------|--------|-------------|
| What is the capital of the country where the Eiffel Tower is located? | Bridge | ✅ Paris | Correct two-hop reasoning |
| Who was the president of the US when the Berlin Wall fell? | Temporal | ✅ George H. W. Bush | Three-step decomposition |
| Which university did the author of *The Old Man and the Sea* attend? | Bridge | ❌ Hallucinated answer | Retriever found insufficient evidence but the final agent hallucinated |

---

# 📚 Example Experience Library

```json
[
  {
    "query_type": "bridge",
    "insight": "QueryDecomposer → Retriever → EvidenceSelector → ConcludeAgent works well.",
    "utility": 0
  },
  {
    "query_type": "bridge",
    "insight": "Decompose more carefully before retrieval.",
    "utility": 0
  }
]
```

Unlike traditional machine learning, these experiences improve reasoning **without updating model parameters**.

---

# 🔍 Key Observations

The third query demonstrates an important failure mode.

- The **EvidenceSelector** correctly determined that reliable evidence was unavailable.
- The **ConcludeAgent** still generated a hallucinated answer.

This is precisely the scenario that motivates the paper's **Role-Aware Prompt Evolution (RoPE)** mechanism.

The Experience Library successfully records the failure but cannot modify agent behavior without prompt evolution.

---

# 📖 Comparison with the Paper

| Paper Contribution | Recreation |
|--------------------|------------|
| Adaptive orchestration | ✅ |
| Experience Library | ✅ |
| Semantic experience retrieval | ✅ |
| Learning without weight updates | ✅ |
| Role-Aware Prompt Evolution (RoPE) | ❌ |

---

# 💡 What I Learned

Implementing HERA highlighted several important ideas:

- Adaptive orchestration is often more valuable than adding more agents.
- Experience replay can improve reasoning without retraining the model.
- Separating retrieval failures from reasoning failures makes debugging much easier.
- Prompt engineering alone cannot solve every failure; orchestration matters just as much.

---

# 🔗 Connection to My Research

My published research project, **DeepRAG**, introduced a multi-agent Retrieval-Augmented Generation framework using static agent orchestration.

HERA extends that concept by introducing:

- Dynamic orchestration
- Experience-driven reasoning
- Self-improving execution strategies
- Natural-language experience replay

This recreation helped me understand how modern multi-agent systems can continuously improve without changing model weights.

---

# 📅 ResearchRebuilt Series

| Week | Paper | Status |
|------|-------|--------|
| Week 1 | HERA — Multi-Agent RAG with Evolving Orchestration | ✅ Complete |
| Week 2 | Coming Next Tuesday | 🔜 |

---

# 📚 References

- HERA Paper: https://arxiv.org/abs/2604.00901
- FAISS: https://github.com/facebookresearch/faiss
- Sentence Transformers: https://www.sbert.net/
- Wikipedia API: https://pypi.org/project/Wikipedia-API/
- Groq: https://console.groq.com

---

# 👨‍💻 Author

**Aryan Balani**

AI/ML Engineer • Multi-Agent Systems • RAG • LLMs

- GitHub: https://github.com/PHENOGRAMMER
- LinkedIn: https://www.linkedin.com/in/aryanbalani

---

⭐ If you found this project interesting, consider starring the repository and following the **ResearchRebuilt** series for weekly AI research implementations.