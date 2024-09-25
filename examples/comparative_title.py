import embedding_bucketing.embedding_model_test as em

from config import openai

em.config(openai)

starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]

cache_file="cache_comparititve_title.json"
cache, comparitive_title_buckets = em.init(cache_file, starting_comparitive_title_buckets)

max_distance = 0.55
user_input = input("enter title:")

closest_dis, closest_bucket, ID, encoding=  em.auto_sort(user_input, max_distance, starting_comparitive_title_buckets, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits=6)

print(closest_bucket, encoding)
