# orchestrator.py
import json
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def orchestrate(query, experience_library):
    """Generate agent execution topology for this query."""
    experiences = experience_library.retrieve(query, top_k=3)
    exp_text = "\n".join([f"- [{e['query_type']}] {e['insight']} (utility: {e['utility']})"
                          for e in experiences]) if experiences else "None yet."

    prompt = f"""You are an orchestrator for a multi-agent RAG system.
Available agents: QueryDecomposer, Retriever, EvidenceSelector, ConcludeAgent

Relevant past experiences:
{exp_text}

Given this query: "{query}"

Design the execution plan. Return ONLY valid JSON:
{{
  "query_type": "bridge|comparison|temporal|intersection",
  "reasoning": "one sentence why this topology",
  "topology": ["QueryDecomposer", "Retriever", "EvidenceSelector", "ConcludeAgent"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    text = response.choices[0].message.content
    # Clean JSON
    start = text.find('{')
    end = text.rfind('}') + 1
    return json.loads(text[start:end])