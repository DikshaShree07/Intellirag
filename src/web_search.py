"""
Fallback tool: when retrieved documents are irrelevant even after
a rewrite, the agent searches the live web instead of hallucinating
an answer from bad context.
"""

from tavily import TavilyClient


def web_search(api_key, query, max_results=3):
    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results)
    # Return a simple list of text snippets, similar shape to retrieved chunks
    snippets = []
    for r in results.get("results", []):
        snippets.append(f"{r.get('title', '')}: {r.get('content', '')}")
    return snippets
