
#tests: non-fiction, crime, horror, war, love story
import Module.cache_store as cache_store
import os
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np
from openai import OpenAI
from config import openai
client = OpenAI(api_key = openai,)
EMBEDDING_MODEL = "text-embedding-3-small"


def init(cache_file):
    global cache 
    cache = cache_store.Cache(cache_file)
    return cache

def get_embedding(input_to_model):
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
def nearest_word(word1, word2):  # embedding method
    # Get embeddings for both words
    word1_e = cache.read_from_cache(word1) # get the embedding vector from cache
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
    return distance


def new_bucket(name): # make new bucket method
    embedding = np.array(get_embedding(name)) # get embedding for new bucket name
    cache.write_to_cache(name, embedding) # write the new embedding to the cache


def get_cache(cache_file): # get cache
    array = []
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f: # read file
            cache = json.load(f)  # load cache
            for i in range(len(cache)): # loop through cache
                array.append(cache.keys()) # add the cache keys to a array 
        print(f"Cache loaded from {cache_file}")
        return(array[0]) # return array
    else:
        return None
    

def llm_call(input_message): #llm call method 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "give a one word answer"},
            {"role": "user", "content": input_message}
        ],
         max_tokens=5,
        temperature=0.1
    )
    local_response = response.choices[0].message.content
    return local_response


def start_cache(starting_array):  # preload cache with starting array of items

    for i in range(len(starting_array)):  # looping through all items in array
        print(starting_array[i-1], get_embedding(starting_array[i-1]))   
        cache.write_to_cache(starting_array[i-1], get_embedding(starting_array[i-1])) # getting embedding vector for word and adding it to the cache

def adjust(word, word2): # Here we are finding the average of the 2 embedding vectors and replacing the old vectors with the new ones
    #word = get_embedding(word)
    new_vec =  cache.adjusting_vectors(get_embedding(word), get_embedding(word2)) # find the average
    print("old vec:", word2) 
    cache.write_to_cache(word2, new_vec) # writing new vector to the cache

def nearest_word_E_D(word1, word2):  # functio to get the distance between to words using euclidean distance
    # Get embeddings for both words
    word1_e = cache.read_from_cache(word1) # get the embedding vector from cache
    #word2 = cache.read_from_cache(word2)
    if word1_e is None:
        word1_e = np.array(get_embedding(word1))
        print("l", word1_e)
        cache.write_to_cache(word1, np.array(get_embedding(word1)))
        word1_e = cache.read_from_cache(word1)

    word2 = np.array(get_embedding(word2))

    print("word_e", word1_e)

    word1_e = normalize(word1_e)

    word2 = normalize(word2)

    #word1_e = word1_e.reshape(1, -1)  # Reshape to (1, n_features)
    #word2 = word2.reshape(1, -1)  # Reshape to (1, n_features)

    # Compute
    distance = np.linalg.norm(word2 - word1_e)


    return distance

def averaging_and_compare(word1, word2):  # in progress
    word1_e = cache.read_from_cache(word1)

    if word1_e is None:
        word1_e = np.array(get_embedding(word1))
        print("l", word1_e)
        cache.write_to_cache(word1, np.array(get_embedding(word1)))
    

    word2 = np.array(get_embedding(word2))

    print("word_e", word1_e)

    word1_e = normalize(word1_e)

    word2 = normalize(word2)
    print(word1_e)
    #word1_e = word1_e.reshape(1, -1)  # Reshape to (1, n_features)
    #word2 = word2.reshape(1, -1)  # Reshape to (1, n_features)

    # Compute
    distance = np.linalg.norm(word2 - word1_e)


    return distance


    




