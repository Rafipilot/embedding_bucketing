import embedding_bucketing.embedding_model_test as em
from config import openai

em.config(openai)

cache_file="cache_theme.json"

start_theme = ["Love", "Sacrifice", "Sad", "Death", "Dark"]

cache, theme = em.init(cache_file, start_theme)

print("Buckets:", theme)

max_distance = 0.55
input_theme = input("input theme: ")

closest_distance, closest_theme, bucket_id, bucket_binary_encoding = em.auto_sort(input_theme, max_distance, theme, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)
print("Bucket:", closest_theme, "  Binary encoding", bucket_binary_encoding)

