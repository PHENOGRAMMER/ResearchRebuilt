# naive_memory_manager.py
from groq import Groq
from memory_bank import MemoryBank
import os, json
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def naive_manage(bank: MemoryBank, new_info: str) -> str:
    """
    Naive heuristic: retrieves related memories and uses a vanilla LLM
    to decide ADD/UPDATE/DELETE/NOOP — no RL training, just in-context instruction.
    This is the fragmentation failure mode the paper demonstrates.
    """
    related = bank.retrieve(new_info, top_k=3)
    related_text = "\n".join([f"[{i}] {m}" for i, m in related]) if related else "None"

    prompt = f"""You are a memory manager. Given existing memories and new information,
decide what to do. Choose ONE operation:
- ADD: new information that doesn't exist yet
- UPDATE [index]: update an existing memory with new info (replace it)
- DELETE [index]: remove an outdated memory
- NOOP: no change needed

Existing related memories:
{related_text}

New information: "{new_info}"

Respond with JSON only:
{{"operation": "ADD|UPDATE|DELETE|NOOP", "index": null_or_number, "content": "new memory text or null"}}"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    text = resp.choices[0].message.content.strip()
    try:
        start, end = text.find('{'), text.rfind('}') + 1
        decision = json.loads(text[start:end])
    except Exception:
        decision = {"operation": "ADD", "index": None, "content": new_info}

    # Execute operation
    op = decision.get("operation", "ADD").upper()
    idx = decision.get("index")
    content = decision.get("content", new_info)

    if op == "ADD":
        bank.add(content or new_info)
    elif op == "UPDATE" and idx is not None:
        bank.update(int(idx), content or new_info)
    elif op == "DELETE" and idx is not None:
        bank.delete(int(idx))
    # NOOP: do nothing

    return op