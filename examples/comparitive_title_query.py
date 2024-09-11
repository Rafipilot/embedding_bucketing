import embedding_buckting 

from config import openai

embedding_buckting.config(openai)

cache_file="cache_comparititve_title.json"
cache = embedding_buckting.init(cache_file)

starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]

compartive_title_buckets = embedding_buckting.get_cache(cache_file)
if compartive_title_buckets is None:
    embedding_buckting.start_cache(starting_comparitive_title_buckets)
    compartive_title_buckets = embedding_buckting.get_cache(cache_file)

comparitive_title_input = input("Enter comparitive titles: ")
comparitive_title_input = comparitive_title_input + "Using the provided title as input, please identify the closest matching title from the following list: " + str(compartive_title_buckets)

output = embedding_buckting.llm_call(comparitive_title_input)
print(output)