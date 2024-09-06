
#tests: non-fiction, crime, horror, war, love story
import cache_store
import os
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"

def init(cache_file):
    global cache 
    cache = cache_store.Cache(cache_file)
    return cache

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

def new_bucket(name):
    embedding = np.array(get_embedding(name))
    cache.write_to_cache(name, embedding)

def get_cache(cache_file):
    array = []
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            for i in range(len(cache)):
                array.append(cache.keys())
        print(f"Cache loaded from {cache_file}")
        return(array[0])
    else:
        return None

def start_cache(starting_array):

    for i in range(len(starting_array)):
        print(starting_array[i-1], get_embedding(starting_array[i-1]))
        cache.write_to_cache(starting_array[i-1], get_embedding(starting_array[i-1]))

def adjust(word, word2):
    #word = get_embedding(word)
    new_vec =  cache.adjusting_vectors(get_embedding(word), get_embedding(word2))
    print("old vec:", word2)
    cache.write_to_cache(word2, new_vec)


    




