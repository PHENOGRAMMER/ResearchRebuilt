# agents.py
AGENTS = {
    "QueryDecomposer": {
        "role": "Break the multi-hop question into 2-3 simpler single-hop sub-questions that can be answered independently.",
        "tools": ["decompose"]
    },
    "Retriever": {
        "role": "Given a sub-question, search Wikipedia and return the top 3 most relevant passages.",
        "tools": ["wikipedia_search"]
    },
    "EvidenceSelector": {
        "role": "Given retrieved passages and a sub-question, extract only the sentences directly relevant to answering it.",
        "tools": ["filter"]
    },
    "ConcludeAgent": {
        "role": "Given all evidence gathered, synthesize a final answer to the original question. Be concise.",
        "tools": ["synthesize"]
    }
}

def get_agent_prompt(agent_name, task, context="", experiences=""):
    agent = AGENTS[agent_name]
    prompt = f"""You are the {agent_name} agent.
Your role: {agent['role']}

{f'Relevant past experiences to guide you:{chr(10)}{experiences}' if experiences else ''}

{f'Context from previous agents:{chr(10)}{context}' if context else ''}

Task: {task}

Respond with only your output, no preamble."""
    return prompt