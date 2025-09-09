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

    f = FuzzyDict(max_distance=0.9, embed_func=embed)
    f['hello'] = 'world'
    f['goodbye'] = 'moon'
    f['Adios'] = 'luna'

    assert f['hello'] == 'world'
    assert f['hello!'] == 'world'
    assert f['goodbye!'] == 'moon'
    assert f['bye'] == 'moon'
    assert f['adiós'] == 'luna'
    assert f['Adiós!'] == 'luna'
