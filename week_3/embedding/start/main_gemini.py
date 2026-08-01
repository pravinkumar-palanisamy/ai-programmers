from google import genai
import faiss
import numpy
import keys

# Example documents about movies
texts = [
    "The Godfather is a classic mafia crime drama",
    "Inception explores dreams within dreams",
    "The Shawshank Redemption is a story about hope and friendship",
]


def get_embedding(text):
    # 1. Create an OpenAI client
    client = genai.Client(api_key=keys.api_key)
    # 2. Make an API call to generate embeddings using the text-embedding-3-small model
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    # 3. Return the embedding vector from the response
    return result.embeddings[0].values

    # Placeholder for the actual implementation
    # return [0] * 1536  # Placeholder with the expected dimension


embeddings = []
for text in texts:
    embedding = get_embedding(text)
    embeddings.append(embedding)

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

index.add(numpy.array(embeddings, dtype='float32'))

query = 'Tell me about a prison movie'

query_embedding = get_embedding(query)
distances, indicies = index.search(numpy.array([query_embedding], dtype='float32'), 3)

for i in range(3):
    print(f"Match {i+1}, Distance: {distances[0][i]:.4f}")
    print(texts[indicies[0][i]])
