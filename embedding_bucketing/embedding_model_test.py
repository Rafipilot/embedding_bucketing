
#tests: non-fiction, crime, horror, war, love story
import os
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np



def config(apikey):
    global client 
    client = OpenAI(api_key = apikey,)
    #return client
EMBEDDING_MODEL = "text-embedding-3-small"


def init(cache_file, starting_buckets):
    global cache 
    cache = Cache(cache_file)
    Bucket_array = get_cache(cache_file) # get the list of buckets from cache

    if Bucket_array is None: # if the is no cache file
        print("no file")
        start_cache(starting_buckets) # add the starting elements to the cache so we have a base of buckets to start with
        Bucket_array = get_cache(cache_file) # get the list of buckets from cache

    return cache, Bucket_array


def get_embedding(input_to_model):
    response = client.embeddings.create(
        input=input_to_model,
        model=EMBEDDING_MODEL
    )

    return response.data[0].embedding


def normalize(embedding): # inbuilt
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


# Function to find the nearest word by comparing distances
def nearest_word(word1, word2):  # embedding method
    # Get embeddings for both words
    word1_e = cache.get_embedding_from_cache(word1) # get the embedding vector from cache
    #word2 = cache.read_from_cache(word2)
    if word1_e is None:
        word1_e = np.array(get_embedding(word1))
        cache.write_to_cache(word1, np.array(get_embedding(word1)))

    word2 = np.array(get_embedding(word2))


    word1_e = normalize(word1_e)

    word2 = normalize(word2)

    word1_e = word1_e.reshape(1, -1)  # Reshape to (1, n_features)
    word2 = word2.reshape(1, -1)  # Reshape to (1, n_features)

    # Compute cosine similarity
    similarities = cosine_similarity(word2, word1_e)

    # Compute cosine distance
    distance = 1 - similarities[0, 0]
    return distance


def new_bucket(name):
    embedding = get_embedding(name)  # Get embedding for the new bucket
    cache.write_to_cache(name, embedding, assign_id=True)  # Assign a unique ID


def get_cache(cache_file): 
    array = []
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            cache_data
            buckets = cache_data.get("buckets", {})
            array = list(buckets.keys())
        return array
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


def start_cache(starting_array):
    for item in starting_array:
        if cache.read_from_cache(item) is None:
            embedding = get_embedding(item)
            cache.write_to_cache(item, embedding, assign_id=True)
        else:
            print(f"{item} already exists in cache with ID {cache.get_id(item)}")


def adjust(word, word2): # Here we are finding the average of the 2 embedding vectors and replacing the old vectors with the new ones
    #word = get_embedding(word)
    new_vec =  cache.adjusting_vectors(get_embedding(word), get_embedding(word2)) # find the average
    cache.write_to_cache(word2, new_vec) # writing new vector to the cache

def nearest_word_E_D(word1, word2):  # functio to get the distance between to words using euclidean distance
    # Get embeddings for both words
    word1_e = cache.read_from_cache(word1) # get the embedding vector from cache
    #word2 = cache.read_from_cache(word2)
    if word1_e is None:
        word1_e = np.array(get_embedding(word1))
        cache.write_to_cache(word1, np.array(get_embedding(word1)))
        word1_e = cache.read_from_cache(word1)

    word2 = np.array(get_embedding(word2))

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
        cache.write_to_cache(word1, np.array(get_embedding(word1)))
    

    word2 = np.array(get_embedding(word2))

    word1_e = normalize(word1_e)

    word2 = normalize(word2)
    #word1_e = word1_e.reshape(1, -1)  # Reshape to (1, n_features)
    #word2 = word2.reshape(1, -1)  # Reshape to (1, n_features)

    # Compute
    distance = np.linalg.norm(word2 - word1_e)


    return distance


def auto_sort(word, max_distance, bucket_array, type_of_distance_calc, amount_of_binary_digits):
    Dis_list = []
    for bucket in bucket_array:
        if type_of_distance_calc.upper() == "EUCLIDEAN DISTANCE":    ## this was not working, please debug; when "EUCLIDEAN DISTANCE" is set, the else statement always prints
            distance = nearest_word_E_D(bucket, word)
        elif type_of_distance_calc.upper() == "COSINE SIMILARITY":
            distance = nearest_word(bucket, word)
        else:
            print("not classified distance calc type")
        Dis_list.append((bucket, distance))

    # Sort by distance
    Dis_list.sort(key=lambda x: x[1])

    # Print the results
    for bucket, distance in Dis_list:
        print(f"Bucket: {bucket}, Distance: {distance}")

    # Find the closest bucket
    closest_distance = Dis_list[0]
    closest_bucket  = Dis_list[0][0]
    if closest_distance[1]>max_distance:
        print("c", cache.next_id, amount_of_binary_digits*amount_of_binary_digits)  ## why is this print statement here? Please remove extraneous print statements as we get closer to using/publishing this
        if cache.next_id > (amount_of_binary_digits*amount_of_binary_digits):
            print("Unable to make new bucket due to insufficient amount of binary digits")
        else:
            print("Making New Bucket!")
            new_bucket(word) # make a new bucket for input word as closest distance is greater than max distance 
            closest_bucket = word

    bucket_id = cache.get_id(closest_bucket)
    num_binary_digits = amount_of_binary_digits
    bucket_binary = np.array(list(np.binary_repr(bucket_id, num_binary_digits)), dtype=int)


    return closest_distance, closest_bucket, bucket_id, bucket_binary


def adjusting_vectors(self, vec1, vec2):
    """Average two embedding vectors."""
    array1_np = np.array(vec1)
    array2_np = np.array(vec2)
    adjusted_embedding = (array1_np + array2_np) / 2
    return adjusted_embedding





class Cache:
    def __init__(self, cache_file):
        self.cache = {}
        self.cache_file = cache_file
        self.next_id = 0 # setting the next id to 0 at the start
        self.load_cache()

        ## add attributes / methods for
        # self.bucket_list
        # self.bucket_list_ids

    def write_to_cache(self, key, embedding, assign_id=True):  # optional to assign id
        """Store data in the cache with a given key and optional ID assignment."""
        if assign_id:
            current_id = self.next_id
            self.next_id += 1  # incrementing the next id so that the the next id is unique
        else:
            current_id = self.cache[key]['id'] if key in self.cache else None

        entry = {
            "id": current_id,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        }

        self.cache[key] = entry
        self.save_cache()
        if assign_id:
            print(f"Added {key} with ID {current_id} to cache")

    def read_from_cache(self, key):
        """Retrieve data from the cache."""
        if key in self.cache:
            return self.cache[key]["id"]
        else:
            return None
        
    def get_id(self, key):
        if key in self.cache:
            return self.cache[key]["id"]
        else:
            return None
    
    
    def get_embedding_from_cache(self, key):
        if key in self.cache:
            return self.cache[key]["embedding"]
        else:
            return None
    
    def save_cache(self):
        """Save the cache and next_id to a file."""
        data_to_save = {
            "next_id": self.next_id,
            "buckets": self.cache
        }
        with open(self.cache_file, 'w') as f:
            json.dump(data_to_save, f)
        print(f"Cache saved to {self.cache_file}")

    def load_cache(self):
        """Load the cache and next_id from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.next_id = data.get("next_id", 1)
                self.cache = data.get("buckets", {})
            print(f"Cache loaded from {self.cache_file}")
        else:
            print("No cache file found. Starting with an empty cache.")
    
    def clear_cache(self):
        """Clear the cache in memory and delete the cache file."""
        self.cache.clear()
        self.next_id = 1
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Cache cleared and cache file deleted.")