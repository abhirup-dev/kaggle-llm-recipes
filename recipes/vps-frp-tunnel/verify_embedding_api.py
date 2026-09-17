#!/usr/bin/env python3
"""Verify an OpenAI-compatible Qwen embedding endpoint."""

import argparse
import json
import math
import urllib.request


def get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=30) as response:
        return json.load(response)


def post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer no-key",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--model", default="qwen3-embedding-4b-q4")
    args = parser.parse_args()

    models = get(args.base_url, "/models")
    model_ids = {model["id"] for model in models["data"]}
    if args.model not in model_ids:
        raise RuntimeError(f"{args.model} not present in {sorted(model_ids)}")

    query = (
        "Instruct: Retrieve passages that answer the question\n"
        "Query: What is the capital of China?"
    )
    documents = [
        "Beijing is the capital city of China.",
        "Gravity attracts objects with mass toward one another.",
    ]
    scalar = post(
        args.base_url,
        "/embeddings",
        {"model": args.model, "input": query, "encoding_format": "float"},
    )
    batch = post(
        args.base_url,
        "/embeddings",
        {"model": args.model, "input": documents, "encoding_format": "float"},
    )
    query_vector = scalar["data"][0]["embedding"]
    document_vectors = [item["embedding"] for item in batch["data"]]
    vectors = [query_vector, *document_vectors]
    dimensions = {len(vector) for vector in vectors}
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    if dimensions != {2560}:
        raise RuntimeError(f"Unexpected embedding dimensions: {dimensions}")
    if any(not 0.99 <= norm <= 1.01 for norm in norms):
        raise RuntimeError(f"Embeddings are not normalized: {norms}")

    relevant_score = cosine(query_vector, document_vectors[0])
    irrelevant_score = cosine(query_vector, document_vectors[1])
    if relevant_score <= irrelevant_score:
        raise RuntimeError(
            f"Semantic ordering failed: relevant={relevant_score}, irrelevant={irrelevant_score}"
        )

    print(
        json.dumps(
            {
                "model": args.model,
                "dimensions": dimensions.pop(),
                "l2_norms": [round(norm, 6) for norm in norms],
                "relevant_cosine": round(relevant_score, 6),
                "irrelevant_cosine": round(irrelevant_score, 6),
                "prompt_tokens": scalar.get("usage", {}).get("prompt_tokens"),
                "batch_prompt_tokens": batch.get("usage", {}).get("prompt_tokens"),
            }
        )
    )


if __name__ == "__main__":
    main()
