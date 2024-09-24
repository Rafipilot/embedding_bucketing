import embedding_bucketing.embedding_model_test as em

from config import openai

em.config(openai)

starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]

cache_file="cache_comparititve_title.json"
cache, comparitive_title_buckets = em.init(cache_file, starting_comparitive_title_buckets)


comparitive_title_input = input("Enter comparitive titles: ")
comparitive_title_input = comparitive_title_input + "Using the provided title as input, please identify the closest matching title from the following list or if none are that close then say none: " + str(comparitive_title_buckets)

output = em.llm_call(comparitive_title_input)
print(output)