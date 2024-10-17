import embedding_bucketing.embedding_model_test as em
from config import openai_key
em.config(openai)

cache_file="cache_comparititve_title.json"
starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]
cache, titles = em.init(cache_file, starting_comparitive_title_buckets)

print("Buckets:", titles)

max_distance = 0.55
inputs = ["Beauty and the beast", "The red october", "the big short"]

for i in range(len(inputs)):
    closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(cache, inputs[i], max_distance, titles, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)

    print("Encoded: ", inputs[i], "into", "Bucket:", closest_genre, "With binary encoding", bucket_binary_encoding)
