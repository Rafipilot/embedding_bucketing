import embedding_bucketing.embedding_model_test as em
from config import openai_key


em.config(openai_key) # setting up the module, here you pass your personal Openai api key through
cache_file_name="cache_genre.json"  # name of the cache file to save the embedding and their buckts in

start_Genre = ["Drama", "Commedy", "Action", "romance", "documentry"]  # starting array of buckets, if there are no buckets found then this is the list of buckets that will be used
cache, Genre = em.init(cache_file_name, start_Genre) # init cache return cache object and the array of buckets, in this case genres

input_genre = input("Input a genre: ") #User input
max_distance = 0.5 # max distance a word can be from the closest bucket before we create a new bucket

closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(cache, input_genre, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)  # using autosort 
print(closest_genre) # printing the closest bucket's name
