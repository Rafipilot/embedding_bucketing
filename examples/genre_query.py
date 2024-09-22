import embedding_bucketing.embedding_model_test as em

from config import openai  # openai key

em.config(openai)

cache_file="cache_genre.json"
cache = em.init(cache_file)
print(cache)

start_Genre = ["Drama", "Comedy", "Action", "romance", "documentary"]

Genre = em.get_cache(cache_file) # get the list of buckets from cache
if Genre is None: # if the is no cache file
    print("no file")
    em.start_cache(start_Genre) # add the starting elements to the cache so we have a base of buckets to start with
    Genre = em.get_cache(cache_file) # get the list of buckets from cache

print(Genre)


max_distance = 0.5

input_genre = input("input genre: ")
closest_distance, closest_genre = em.auto_sort(input_genre, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY")



