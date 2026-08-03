# Search-R1 🔎

## Introduction

Search-R1 is a lightweight, conceptual replication of the interleaving reasoning-and-search patterns demonstrated by models like DeepSeek-R1. This project demonstrates how a standard LLM can be prompted to autonomously decide *when* to search, *what* to search for, and *when* to stop and synthesize an answer, all without fine-tuning.

## Search-R1 Idea

The core insight behind this project is that while DeepSeek-R1 uses GRPO (Group Relative Policy Optimization) to inherently learn when to invoke search tools, we can simulate this decision boundary using strict XML tags. By constraining the model to a `<think>...</think>`, `<search>...</search>`, and `<answer>...</answer>` loop, the agent is forced to iteratively evaluate its own knowledge gaps against retrieved context before providing a final response.

## Architecture

The project is built on a simple architecture:
- **`main.py`**: The orchestrator that defines test queries and runs the comparison benchmark across three methods (No Retrieval, Naive RAG, and Search-R1).
- **`search_r1_agent.py`**: The core reasoning loop. It enforces the XML structure, queries the LLM, and manages the state machine (extracting tags, triggering searches, and verifying evidence).
- **`search_engine.py`**: A minimal Wikipedia search client that retrieves summaries and sections, scoring them based on query overlap and entity relevance.
- **`comparator.py`**: Contains the baseline implementations (Vanilla LLM and Naive RAG) for comparison.

## Implementation

The implementation relies on:
- **Groq API**: For fast, inexpensive LLM inference (using models like `llama-3.1-8b-instant`).
- **Wikipedia-API**: For live web retrieval without complex scraping.
- **Python**: Core logic, utilizing standard libraries alongside `colorama` for beautiful, readable terminal traces.

### The Agent Loop
1. **Think**: The model emits `<think>` tags to reason about what it knows and what it needs.
2. **Search**: The model emits a `<search>` query. The agent pauses inference, hits Wikipedia, and injects `<information>` blocks into the context window.
3. **Evaluate**: An external heuristic (`evidence_is_sufficient`) checks if the model has enough facts to answer the original question.
4. **Answer**: Once sufficient evidence is gathered, the model outputs `<answer>`.

## Results

When tested on multi-hop reasoning questions, Search-R1 significantly outperforms both the vanilla model and a single-shot Naive RAG pipeline.

| Query | Vanilla | Naive RAG | Search-R1 | Searches | Correct |
|---|---|---|---|---|---|
| Python Creator | ❌ | ❌ | ✅ | 2 | ✅ |
| Eiffel Tower | ❌ | ❌ | ✅ | 2 | ✅ |
| Berlin Wall | ❌ | ❌ | ✅ | 2 | ✅ |

Results are automatically saved to `results/run_XXX.json` for further analysis.

## Comparison with Paper

Unlike the actual DeepSeek-R1 model which internalizes the search policy via RL, this project uses a frozen, off-the-shelf instruction-tuned model. 
- **Pros**: It requires zero training compute, is entirely prompt-based, and allows for transparent debugging of the reasoning trace.
- **Cons**: It is highly sensitive to the prompt format, occasionally hallucinates search tags, and requires external guardrails (like forced synthesis and duplicate-search checks) to prevent infinite loops.

## Limitations

- **Search Quality**: The minimal Wikipedia search engine can sometimes fail to retrieve relevant pages if the entity extraction is imperfect.
- **Context Window**: Appending multiple `<information>` blocks can quickly fill up the context window for complex queries.
- **API Rate Limits**: Iterative loops hit the Groq API frequently, which can trigger rate limit errors on free tiers.

## Future Work

- **Better Search**: Integrate a more robust web search API (e.g., Google Custom Search, Tavily) instead of relying solely on Wikipedia.
- **Vector DB**: Implement a lightweight vector database to rank and retrieve only the most semantically relevant chunks from long Wikipedia articles.
- **Dynamic Turn Limits**: Implement a more sophisticated agent that can adjust its `max_turns` based on query complexity.
