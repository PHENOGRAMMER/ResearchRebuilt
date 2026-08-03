# hera.py
from groq import Groq
from agents import get_agent_prompt
from experience_library import ExperienceLibrary
from orchestrator import orchestrate
from retriever import search_wikipedia
from colorama import Fore, Style, init
import os
from dotenv import load_dotenv

load_dotenv()
init()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_agent(agent_name, task, context="", experiences=""):
    prompt = get_agent_prompt(agent_name, task, context, experiences)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()

def run_hera(query, experience_library, expected_answer=None):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"HERA Multi-Agent RAG Pipeline")
    print(f"Query: {query}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    # Step 1: Orchestrator selects topology
    print(f"{Fore.YELLOW}[Orchestrator] Planning topology...{Style.RESET_ALL}")
    plan = orchestrate(query, experience_library)
    print(f"  Query type: {plan['query_type']}")
    print(f"  Reasoning: {plan['reasoning']}")
    print(f"  Topology: {' → '.join(plan['topology'])}\n")

    # Retrieve relevant experiences for agents
    experiences = experience_library.retrieve(query)
    exp_text = "\n".join([f"- {e['insight']}" for e in experiences])

    context = ""
    sub_questions = []
    evidence = []

    for agent_name in plan['topology']:
        print(f"{Fore.GREEN}[{agent_name}] Running...{Style.RESET_ALL}")

        if agent_name == "QueryDecomposer":
            output = run_agent(agent_name, query, experiences=exp_text)
            sub_questions = [q.strip() for q in output.split('\n') if q.strip()]
            print(f"  Sub-questions: {sub_questions}")
            context = output

        elif agent_name == "Retriever":
            all_passages = []
            for sq in sub_questions[:3]:  # cap at 3 hops
                passage = search_wikipedia(sq)
                all_passages.append(f"[For: {sq}]\n{passage}")
                print(f"  Retrieved for: {sq[:50]}...")
            context = "\n\n".join(all_passages)

        elif agent_name == "EvidenceSelector":
            output = run_agent(agent_name,
                               f"Original query: {query}\nSub-questions: {sub_questions}",
                               context=context, experiences=exp_text)
            evidence = output
            context = output
            print(f"  Selected evidence: {output[:150]}...")

        elif agent_name == "ConcludeAgent":
            final_answer = run_agent(agent_name,
                                     f"Original query: {query}",
                                     context=context, experiences=exp_text)
            print(f"\n{Fore.MAGENTA}[Final Answer]{Style.RESET_ALL}")
            print(f"  {final_answer}")

    # Evaluate and update experience library
    if expected_answer:
        correct = expected_answer.lower() in final_answer.lower()
        print(f"\n{Fore.BLUE}[Evaluation]{Style.RESET_ALL}")
        print(f"  Expected: {expected_answer}")
        print(f"  Correct: {'✓' if correct else '✗'}")

        # Add insight to experience library (simplified GRPO logic)
        if correct:
            insight = f"For {plan['query_type']} questions, topology {' → '.join(plan['topology'])} works well."
            experience_library.add(plan['query_type'], insight)
            print(f"  [ExperienceLib] Insight added from success.")
        else:
            insight = f"For {plan['query_type']} questions, decompose more carefully before retrieval."
            experience_library.add(plan['query_type'], insight)

    experience_library.save()
    return final_answer


if __name__ == "__main__":
    lib = ExperienceLibrary()
    lib.load()

    # Multi-hop test questions (from the paper's benchmarks)
    test_cases = [
        {
            "query": "What is the capital of the country where the Eiffel Tower is located?",
            "answer": "Paris"
        },
        {
            "query": "Who was the president of the United States when the Berlin Wall fell?",
            "answer": "George H.W. Bush"
        },
        {
            "query": "Which university did the author of 'The Old Man and the Sea' attend?",
            "answer": "None"  # Hemingway didn't attend university
        }
    ]

    for tc in test_cases:
        run_hera(tc["query"], lib, tc["answer"])
        print("\n")