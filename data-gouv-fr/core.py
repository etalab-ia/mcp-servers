import os
import requests

from clients.utils import log_and_raise_for_status, retry


def format_chunk(chunk: dict) -> str:
    context = ""
    if chunk.get("context"):
        context = " (" + " > ".join(chunk["context"]) + ")"

    text = ""
    text += f"URL: {chunk['url']}\n"
    text += f"TITLE: {chunk['title'] + context}\n"
    text += f"PASSAGE: {chunk['text']}"
    return text


def rerank(prompt: str, hits: list[str], model=None) -> list[str]:
    """
    Reranks a list of documents based on their relevance to the prompt.

    Args:
        prompt: The query or question to rerank documents against
        hits: List of document texts to rerank
        model: Optional model name to use for reranking (uses default if None)

    Returns:
        A reordered list of the input documents, sorted by relevance
    """
    # API endpoint (replace with your actual endpoint)
    url = "https://albert.api.etalab.gouv.fr/v1/rerank"

    # Prepare the request payload
    payload = {"prompt": prompt, "input": hits, "model": model}

    # Set headers
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('ALBERT_API_KEY')}"}

    # Send the request
    response = requests.post(url, headers=headers, json=payload)
    log_and_raise_for_status(response, "LLM API error")
    result = response.json()

    # Extract the scores and indices
    scored_indices = result["data"]

    # Sort the indices by score in descending order (higher score = more relevant)
    sorted_indices = sorted(scored_indices, key=lambda x: x.get("score", 0), reverse=True)

    # Reorder the original hits list based on the sorted indices
    reranked_hits = []
    for item in sorted_indices:
        index = item.get("index", 0)
        if 0 <= index < len(hits):
            reranked_hits.append(hits[index])

    return reranked_hits
