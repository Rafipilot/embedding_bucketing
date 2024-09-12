import embedding_buckting.embedding_model_test as em

from config import openai  # openai key

em.config(openai)

cache_file="cache_genre.json"
cache = em.init(cache_file)
print(cache)

start_Genre = ["Drama", "Commedy", "Action", "romance", "documentry"]

Genre = em.get_cache(cache_file) # get the list of buckets from cache
if Genre is None: # if the is no cache file
    print("no file")
    em.start_cache(start_Genre) # add the starting elements to the cache so we have a base of buckets to start with
    Genre = em.get_cache(cache_file) # get the list of buckets from cache

print(Genre)
EMBEDDING_MODEL = "text-embedding-3-small"

max_distance = 0.7


word2 = input("input genre: ")
closest_distance, closest_genre = em.auto_sort(word2, max_distance, Genre, type_of_distance_calc="EUCLIDEAN DISTANCE")

