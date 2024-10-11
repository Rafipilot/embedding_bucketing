import embedding_bucketing.embedding_model_test as em

from config import openai

em.config(openai)

starting_comparative_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]

cache_file="cache_comparative_title.json"
cache, comparative_title_buckets = em.init(cache_file, starting_comparative_title_buckets)


comparative_title_input = input("Enter comparative titles: ")
comparative_title_input = comparative_title_input + "Using the provided title as input, please identify the closest matching title from the following list or if none are that close then say none: " + str(comparitive_title_buckets)

output = em.llm_call(comparative_title_input)
print(output)