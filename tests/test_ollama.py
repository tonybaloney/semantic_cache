import openai
import pytest

from semantic_cache import FuzzyDict, semantic_cache

@pytest.fixture
def openai_client():
    return openai.Client(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )


def test_ollama_embedding(openai_client):
    def embed(text: str) -> tuple[float, ...]:
        response = openai_client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        )
        return tuple(response.data[0].embedding)

    f = FuzzyDict(max_distance=0.8, embed_func=embed)
    f['hello'] = 'world'
    f['goodbye'] = 'moon'
    f['Adios'] = 'luna'

    assert f['hello'] == 'world'
    assert f['hello!'] == 'world'
    assert f['goodbye!'] == 'moon'
    assert f['bye bye'] == 'moon'
    assert f['adiós'] == 'luna'
    assert f['Adiós!'] == 'luna'

    with pytest.raises(KeyError):
        _ = f['Galaxy']


def test_ollama_decorator(openai_client):
    @semantic_cache(
        embed_func=lambda text: tuple(
            openai_client.embeddings.create(
                model="nomic-embed-text:latest",
                input=text
            ).data[0].embedding
        ),
        max_distance=0.91,
        max_size=10
    )
    def get_response(prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    r1 = get_response("What is the capital of France?")
    r2 = get_response("What is the capital of France!")
    r3 = get_response("What is the capital of Germany?")

    assert r1 == r2  # Should hit the cache
    assert r1 != r3  # Different question, should not hit the cache
