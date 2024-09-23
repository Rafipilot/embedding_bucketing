import embedding_bucketing.embedding_model_test as em

from config import openai  # openai key

em.config(openai)

cache_file="cache_genre.json"

start_Genre = ["Drama", "Comedy", "Action", "romance", "documentary"]

cache, Genre = em.init(cache_file, start_Genre)

print("genre:", Genre)

max_distance = 0.7

input_genre = input("input genre: ")

closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(input_genre, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 4)
print(closest_genre, bucket_binary_encoding)





