import embedding_bucketing.embedding_model_test as em
from config import openai  
em.config(openai)

cache_file="cache_genre.json"
start_Genre = ["Drama", "Comedy", "Action", "romance", "documentary"]
cache, Genre = em.init(cache_file, start_Genre)

print("Buckets:", Genre)

max_distance = 0.55
input_genre = input("input genre: ")

closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(input_genre, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)
print("Bucket:", closest_genre, "  Binary encoding", bucket_binary_encoding)





