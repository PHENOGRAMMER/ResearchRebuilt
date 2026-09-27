"""
lora_finetune.py — Week 10 · LoRA: Low-Rank Adaptation
#ResearchRebuilt | Hu et al., ICLR 2022 | arxiv: 2106.09685

Simulates the key LoRA insight live using the Groq API:
  BASE call    = frozen W₀  (general model, no task specialisation)
  ADAPTER call = W₀ + ΔW   (LoRA-adapted: task-specific residual injected)

Three experiments from the paper reproduced conceptually:
  1. NL → SQL  (WikiSQL task from paper Table 4)
  2. Summarisation  (SAMSum from paper Table 4)
  3. Domain adaptation  (the DeepRAG personal angle)

For each: base model vs LoRA-adapted, side-by-side, token cost logged.
The parameter analogy is printed after each run.
"""

import os, time
from groq import Groq
from colorama import Fore, Style, init
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"   # confirmed available on this Groq API key


def section(title: str, color=Fore.CYAN) -> None:
    bar = "─" * 62
    print(f"\n{color}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")


def call(client: Groq, system: str, user: str, temp: float = 0.1) -> tuple[str, int]:
    """Single Groq call. Returns (text, completion_tokens)."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature=temp,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip(), resp.usage.completion_tokens


def compare(client: Groq, label: str,
            base_system: str, lora_system: str, user: str,
            lora_r: int = 8, model_d: int = 4096, n_layers: int = 32) -> None:
    """
    Run base vs LoRA-adapted call, print side-by-side, log param analogy.
    lora_r / model_d / n_layers used only for the parameter calculation print.
    """
    print(f"  {Fore.WHITE}Task: {label}{Style.RESET_ALL}")
    print(f"  Input: {user[:120]}\n")

    base_out, base_tok = call(client, base_system, user)
    time.sleep(0.5)
    lora_out, lora_tok = call(client, lora_system, user)

    # print outputs
    print(f"  {Fore.YELLOW}[BASE MODEL — W₀ only]{Style.RESET_ALL}")
    for line in base_out.split("\n")[:6]:
        print(f"    {line}")
    print(f"    ... ({base_tok} tokens)\n")

    print(f"  {Fore.GREEN}[LoRA-ADAPTED — W₀ + BA]{Style.RESET_ALL}")
    for line in lora_out.split("\n")[:6]:
        print(f"    {line}")
    print(f"    ... ({lora_tok} tokens)\n")

    # Parameter analogy
    full_params  = model_d * model_d * 2 * n_layers      # Wq + Wv
    lora_params  = 2 * (lora_r * model_d + model_d * lora_r) * n_layers
    compression  = full_params / lora_params
    pct          = lora_params / full_params * 100

    print(f"  {Fore.CYAN}Parameter analogy (r={lora_r}, {model_d}-dim model, {n_layers} layers):")
    print(f"    Full fine-tune  : {full_params:>14,} params (Wq + Wv)")
    print(f"    LoRA adapter    : {lora_params:>14,} params")
    print(f"    Compression     : {compression:>14.0f}×  ({pct:.3f}% of full FT){Style.RESET_ALL}")
    print()


# ─────────────────────────────────────────────────────────────
# GUARD: no API key
# ─────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    print(Fore.RED + "\nGROQ_API_KEY not found in .env")
    print("Add GROQ_API_KEY=<your_key> to a .env file and re-run.")
    print("Get a free key at https://console.groq.com\n")
    exit(0)

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — NL → SQL (WikiSQL, paper Table 4)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 1 · NL→SQL  (Paper Table 4 — WikiSQL task)", Fore.YELLOW)

"""
Paper result (Table 4):
  GPT-3 Full FT     : 73.8% WikiSQL accuracy
  GPT-3 LoRA (37.7M): 74.0% WikiSQL accuracy  ← LoRA WINS with 4,636× fewer params

Here we contrast a general-purpose base call vs an SQL-specialised adapted call.
The base system prompt = frozen W₀. The LoRA system = W₀ + a task residual BA.
"""

BASE_SQL = "You are a helpful AI assistant. Answer the user's question."

LORA_SQL = """You are an expert SQL generator. Convert natural language questions
into precise SQL queries.
Rules:
- Use standard SQL syntax (SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT)
- Infer column names from context; use descriptive aliases
- Return ONLY the SQL query, no explanation
Schema context: you have access to common business tables (orders, customers, products, sales)"""

nl_question = "Find the top 5 customers by total revenue in Q3 2024, showing their name and total spend, highest first."

compare(
    client, "Natural Language → SQL  (WikiSQL benchmark)",
    BASE_SQL, LORA_SQL, nl_question,
    lora_r=8, model_d=4096, n_layers=32
)

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — SUMMARISATION (SAMSum, paper Table 4)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 2 · Summarisation  (Paper Table 4 — SAMSum task)", Fore.MAGENTA)

"""
Paper result (Table 4):
  GPT-3 Full FT     : ROUGE-1/2/L = 52.0/28.0/44.5
  GPT-3 LoRA (37.7M): ROUGE-1/2/L = 53.8/29.8/45.9  ← LoRA WINS again

SAMSum = abstractive summarisation of chat conversations. The LoRA adapter
teaches the model the specific style: concise, third-person, action-focused.
"""

BASE_SUM = "You are a helpful AI. Summarise the following conversation."

LORA_SUM = """You are a dialogue summarisation specialist trained on SAMSum.
Summarise conversations in 1-2 sentences, third-person, past tense.
Focus on: what was decided or agreed, action items, who is responsible.
Do not include greetings, filler phrases, or emotional commentary."""

conversation = """
Hannah: Hey, are we still on for the product demo Thursday 3pm?
Mike: Yes, confirmed. Can you prep the slide deck? I'll handle the live coding part.
Hannah: Sure. Should I include the new pricing slide?
Mike: Yes please, Sarah approved the new tiers yesterday. Also make sure to mention the API limits.
Hannah: Got it. I'll send you a draft by Wednesday evening so you can review.
Mike: Perfect. Let's meet 30 mins before to do a dry run.
Hannah: Sounds good, see you then!
"""

compare(
    client, "Chat dialogue summarisation  (SAMSum benchmark)",
    BASE_SUM, LORA_SUM, conversation,
    lora_r=8, model_d=4096, n_layers=32
)

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — DOMAIN ADAPTATION (DeepRAG personal angle)
# ─────────────────────────────────────────────────────────────
section("EXPERIMENT 3 · Domain Adaptation  (DeepRAG / embedding fine-tuning)", Fore.BLUE)

"""
Personal angle:
In DeepRAG (IJMER / IJIRT), embedding models were fine-tuned on domain-specific
corpora to improve retrieval precision.

Without LoRA: full BERT-base fine-tune = 84,934,656 trainable attention params
With LoRA r=8: 294,912 trainable params = 288× compression

This experiment simulates what that domain adaptation achieves:
base model = general-purpose retrieval reasoning
LoRA model = domain-adapted to multi-agent RAG / information retrieval
"""

BASE_RAG = "You are a helpful AI assistant. Answer questions about information retrieval."

LORA_RAG = """You are an expert in multi-agent Retrieval-Augmented Generation (RAG) systems,
fine-tuned on AI/ML research papers and embedding model literature.
When asked about retrieval, always address:
1. Vector similarity mechanics (cosine, dot-product)
2. Index types (FAISS flat, IVF, HNSW) and their trade-offs
3. Embedding model choices and why domain fine-tuning matters
4. Chunking strategy and its effect on recall
Assume the user is an AI engineer. Be precise, use exact numbers and names."""

rag_question = "Why does fine-tuning an embedding model on domain-specific data improve RAG precision? What exactly changes in the vector space?"

compare(
    client, "Domain adaptation for RAG  (DeepRAG personal angle)",
    BASE_RAG, LORA_RAG, rag_question,
    lora_r=8, model_d=768, n_layers=12    # BERT-base dimensions
)

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY — the paper's key numbers
# ─────────────────────────────────────────────────────────────
section("RESULTS SUMMARY — Paper Numbers Reproduced", Fore.CYAN)

print(f"  {'Task':<28} {'Full FT':>10}  {'LoRA':>10}  {'LoRA Params':>14}  {'Saving':>8}")
print(f"  {'─'*78}")

results = [
    ("WikiSQL (GPT-3 175B)",   "73.8%",  "74.0% ✓",  "37.7M",   "4,636×"),
    ("MNLI-m (GPT-3 175B)",   "89.5%",  "91.7% ✓",  "37.7M",   "4,636×"),
    ("SAMSum R1 (GPT-3 175B)","52.0",   "53.8 ✓",   "37.7M",   "4,636×"),
    ("E2E NLG (GPT-2 Med.)",  "68.2",   "70.4 ✓",   "0.35M",   "1,014×"),
    ("GLUE avg (RoBERTa-L)",  "88.9%",  "89.0% ✓",  "0.8M",    "443×"),
]

for name, full, lora_r, params, saving in results:
    print(f"  {name:<28} {full:>10}  {Fore.GREEN}{lora_r:>10}{Style.RESET_ALL}  {params:>14}  {saving:>8}")

print(f"  {Fore.YELLOW}Pattern across all tasks: LoRA matches or BEATS full fine-tune")
print(f"  with 443× to 4,636× fewer trainable parameters.{Style.RESET_ALL}")
print()