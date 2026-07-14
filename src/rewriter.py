"""
When retrieved chunks are graded irrelevant, this rewrites the
user's question to be clearer/more specific before trying
retrieval again. This mimics how a human would rephrase a
confusing search query.
"""

REWRITE_PROMPT = """The following question did not retrieve relevant results from a document search.
Rewrite it to be clearer and more specific, keeping the same intent.
Return ONLY the rewritten question, nothing else.

Original question: {question}
"""


def rewrite_query(model, question):
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.generate_content(prompt)
    return response.text.strip()
