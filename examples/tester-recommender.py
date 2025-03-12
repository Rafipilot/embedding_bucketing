import embedding_bucketing.embedding_model_test as em
from config import openai_key  
em.config(openai_key)

cache_file="cache_genre_rec.json"
start_Genre =  ["Clothes", "Electronics", "Books", "Children Toys", "Jewelry", "Home", "Beauty", "Sports", "Food", "Music", "Movies", "Games", "Art", "Travel", "Pets", "Health", "Fitness", "Tech", "DIY", "Gardening", "Cooking", "Crafts", "Cars", "Outdoors", "Office", "School", "Baby", "Party", "Wedding", "Grooming", "Drama Book", "Dolls", "Purse", "Wallet", "Chocolates"]
cache, Genre = em.init(cache_file, start_Genre)

print("Buckets:", Genre)

max_distance = 1
input_genre = input("input genre: ")

llm_output = em.llm_call(f"what category would this be: {input_genre}")
print("llm_output", llm_output)
closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(cache,llm_output, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)
print("Bucket:", closest_genre, "  Binary encoding", bucket_binary_encoding)
print("closest_distance", closest_distance) 
