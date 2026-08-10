# evaluator.py
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_with_memory(query: str, retrieved: list[str], agent_name: str) -> str:
    context = "\n".join(retrieved) if retrieved else "No relevant memories found."
    prompt = f"""You are a memory agent. Answer the question using ONLY the provided memory context.
If the context doesn't contain the answer, say "I don't know."

Memory context:
{context}

Question: {query}
Answer:"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

def score_answer(answer: str, expected: str) -> bool:
    """Simple substring match — mirrors EM scoring in paper."""
    answer_clean = answer.strip().lower()
    expected_clean = expected.strip().lower()

    if expected_clean in {"yes", "no"}:
        unknown_patterns = [
            "i don't know",
            "i do not know",
            "not enough information",
            "insufficient information",
            "cannot determine",
            "can't determine",
            "unknown",
        ]

        if any(pattern in answer_clean for pattern in unknown_patterns):
            return False
        first_word = answer_clean.split()[0] if answer_clean else ""
        return first_word == expected_clean
    
    return expected_clean in answer_clean

def run_competency_test(agent, test_name: str, turns: list[dict]) -> dict:
    """
    Run a multi-turn competency test.
    Each turn is: {"inject": text, "query": text, "expected": text, "action": "add|update|forget"}
    """
    agent.clear()
    results = []

    for turn in turns:
        # Inject information into memory
        if turn.get("action") == "add" and turn.get("inject"):
            agent.add(turn["inject"])
        elif turn.get("action") == "update" and turn.get("inject"):
            agent.update(turn.get("old_keyword", ""), turn["inject"])
        elif turn.get("action") == "forget" and turn.get("inject"):
            agent.forget(turn["inject"])

        # Query memory
        if turn.get("query"):
            retrieved = agent.retrieve(turn["query"], top_k=3)
            answer = answer_with_memory(turn["query"], retrieved, type(agent).__name__)
            correct = score_answer(answer, turn["expected"])
            results.append({
                "query": turn["query"],
                "expected": turn["expected"],
                "answer": answer[:80],
                "correct": correct,
                "retrieved_count": len(retrieved)
            })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return {"test": test_name, "accuracy": accuracy, "results": results}