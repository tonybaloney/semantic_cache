from openai import Client

from semantic_cache import semantic_cache

client = Client(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


@semantic_cache(
    embed_func=lambda text: tuple(
        client.embeddings.create(model="nomic-embed-text:latest", input=text)
        .data[0]
        .embedding
    ),
    max_distance=0.91,  # Similarity threshold in cosine similarity
    max_size=512,  # Cache size, LRU
)
def get_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


r1 = get_response("What is the capital of France?")
r2 = get_response("What is the capital of France!")  # Cached
r3 = get_response("What is the capital of Germany?")  # New query
