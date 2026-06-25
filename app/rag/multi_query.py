from typing import List, Dict
from app.llm.groq_llm import generate


QUERY_EXPANSION_PROMPT = """You are an AI assistant that generates multiple search queries from a single user question.

Generate exactly {num_queries} different search queries based on the following question.
Each query should approach the topic from a different angle to maximize retrieval coverage.

Original question: {query}

Return ONLY the queries, one per line, no numbering or bullet points:"""


def expand_query(query: str, num_queries: int = 3) -> List[str]:
    prompt = QUERY_EXPANSION_PROMPT.format(num_queries=num_queries + 1, query=query)
    response = generate(prompt)
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    if query not in queries:
        queries.insert(0, query)
    return queries[:num_queries + 1]


def multi_query_retrieve(query: str, retriever_fn, k: int = 20) -> List[Dict]:
    expanded_queries = expand_query(query)

    all_results = []
    seen = set()
    for q in expanded_queries:
        results = retriever_fn(q, k=k)
        for doc in results:
            if doc["content"] not in seen:
                seen.add(doc["content"])
                doc["source_query"] = q
                all_results.append(doc)

    return all_results
