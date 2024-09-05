
#tests: non-fiction, crime, horror, war, love story
import cache


from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"

cache = cache.Cache()

def get_embedding(input_to_model):

    client = OpenAI(api_key = "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",)

    response = client.embeddings.create(
        input=input_to_model,
        model=EMBEDDING_MODEL
    )

    #print(response.data[0].embedding[:5])
    np.array(response.data[0].embedding[:5])
    return response.data[0].embedding[:5]


def normalize(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


# Function to find the nearest word by comparing distances
def nearest_word(word1, word2):
    # Get embeddings for both words
    word1_e = cache.read_from_cache(word1)
    #word2 = cache.read_from_cache(word2)
    if word1_e is None:
        word1_e = np.array(get_embedding(word1))
        print("l", word1_e)
        cache.write_to_cache(word1, np.array(get_embedding(word1)))

    word2 = np.array(get_embedding(word2))

    print("word_e", word1_e)

    word1_e = normalize(word1_e)

    word2 = normalize(word2)

    word1_e = word1_e.reshape(1, -1)  # Reshape to (1, n_features)
    word2 = word2.reshape(1, -1)  # Reshape to (1, n_features)

    # Compute cosine similarity
    similarities = cosine_similarity(word2, word1_e)

    # Compute cosine distance
    distance = 1 - similarities[0, 0]

   # print("Cosine distance: ", distance)
    return distance






