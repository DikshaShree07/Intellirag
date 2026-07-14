"""
Improved relevance grader.

Instead of making a binary decision immediately,
Gemini scores each chunk from 0–10.

Chunks scoring >=6 are kept.
"""

import re

GRADE_PROMPT = """
You are helping a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Retrieved Document:
----------------
{chunk}
----------------

Rate how useful this document is for answering the question.

Return ONLY ONE NUMBER from 0 to 10.

0 = completely irrelevant

10 = directly answers the question

Do not explain.
"""


def grade_chunk(model, question, chunk):

    prompt = GRADE_PROMPT.format(
        question=question,
        chunk=chunk
    )

    try:

        response = model.generate_content(prompt)

        text = response.text.strip()

        match = re.search(r"\d+", text)

        if match:
            score = int(match.group())
        else:
            score = 0

        return score

    except Exception:

        return 0


def grade_all_chunks(model, question, chunks):

    relevant = []

    print("\n=========== GRADING RESULTS ===========\n")

    for i, chunk in enumerate(chunks):

        score = grade_chunk(
            model,
            question,
            chunk
        )

        print(f"Chunk {i+1} -> Score : {score}")

        if score >= 6:
            relevant.append(chunk)

    return relevant