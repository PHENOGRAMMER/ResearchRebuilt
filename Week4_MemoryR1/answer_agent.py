# answer_agent.py
from groq import Groq
from memory_bank import MemoryBank
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_with_distillation(bank: MemoryBank, question: str, top_k: int = 10) -> dict:
    """
    Answer Agent with Memory Distillation:
    1. Retrieve top_k candidates (simulating the paper's 60-candidate retrieval)
    2. Distill down to relevant entries only
    3. Reason over distilled set to answer
    """
    # Step 1: Retrieve
    retrieved = bank.retrieve(question, top_k=top_k)
    retrieved_text = "\n".join([f"[{i}] {m}" for i, m in retrieved])

    # Step 2: Distill (filter noise)
    distill_prompt = f"""You are a memory distillation agent. 
From the retrieved memories below, select ONLY the ones directly relevant to answering the question.
Return a JSON list of the relevant memory texts only.

Question: {question}

Retrieved memories:
{retrieved_text}

Return JSON: {{"relevant": ["memory text 1", "memory text 2"]}}"""

    distill_resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": distill_prompt}],
        temperature=0.1, max_tokens=300
    )
    distill_text = distill_resp.choices[0].message.content.strip()
    try:
        import json
        start, end = distill_text.find('{'), distill_text.rfind('}') + 1
        relevant = json.loads(distill_text[start:end]).get("relevant", [])
    except Exception:
        relevant = [m for _, m in retrieved[:3]]

    # Step 3: Answer over distilled memories
    answer_prompt = f"""Answer the question using ONLY the provided memories.
Be concise and direct.

Memories:
{chr(10).join(relevant) if relevant else "No relevant memories found."}

Question: {question}
Answer:"""

    ans_resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.1, max_tokens=150
    )

    return {
        "answer": ans_resp.choices[0].message.content.strip(),
        "retrieved_count": len(retrieved),
        "distilled_count": len(relevant),
        "distilled": relevant
    }