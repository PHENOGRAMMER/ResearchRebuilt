# rl_memory_manager.py
from groq import Groq
from memory_bank import MemoryBank
import os, json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rl_manage(bank: MemoryBank, new_info: str) -> str:
    """
    RL-simulated Memory Manager: applies consolidation-first reasoning.
    The key difference: treats additive information as UPDATE, not DELETE+ADD.
    This mirrors the behavior the paper's RL training produces.
    """
    related = bank.retrieve(new_info, top_k=3)
    related_text = "\n".join([f"[{i}] {m}" for i, m in related]) if related else "None"

    prompt = f"""You are an advanced memory manager trained to consolidate information correctly.

CRITICAL RULES (in priority order):
1. If new info ADDS TO or EXTENDS an existing memory (same topic, additive fact), use UPDATE to consolidate both facts into one entry. DO NOT delete the old one.
2. If new info CONTRADICTS an existing memory (old fact is now wrong), use UPDATE to replace it.
3. If new info is completely new with no related memory, use ADD.
4. If new info is already captured, use NOOP.
5. When updating, DO NOT lose historical facts (e.g. past medications) or negative constraints (e.g. dropped features). Explicitly state them.

Example of CONSOLIDATION (not contradiction):
- Existing: "Alice has a dog named Buddy"
- New: "Alice got another dog named Scout"
- Correct operation: UPDATE [idx] → "Alice has two dogs: Buddy and Scout" ✓
- Wrong operation: DELETE + ADD (fragments the information) ✗

Existing related memories:
{related_text}

New information: "{new_info}"

Think step by step:
1. Is this additive (extends existing fact) or contradictory (replaces existing fact)?
2. Choose the operation that best preserves all known information.

Respond with JSON only:
{{"reasoning": "one sentence", "operation": "ADD|UPDATE|DELETE|NOOP", "index": null_or_number, "content": "consolidated memory text"}}"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    text = resp.choices[0].message.content.strip()
    try:
        start, end = text.find('{'), text.rfind('}') + 1
        decision = json.loads(text[start:end])
    except Exception:
        decision = {"operation": "ADD", "index": None,
                    "content": new_info, "reasoning": "fallback"}

    op = decision.get("operation", "ADD").upper()
    idx = decision.get("index")
    content = decision.get("content", new_info)
    reasoning = decision.get("reasoning", "")

    if op == "ADD":
        bank.add(content or new_info)
    elif op == "UPDATE":
        if idx is not None:
            bank.update(int(idx), content or new_info)
        else:
            # Fallback: if it tries to update but provides no index (e.g. empty bank), add it instead.
            bank.add(content or new_info)
            op = "ADD (Fallback)"
    elif op == "DELETE":
        if idx is not None:
            bank.delete(int(idx))

    return op, reasoning