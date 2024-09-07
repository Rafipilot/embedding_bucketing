import embedding_model_test

from embedding_model_test import nearest_word

cache_file="cache_comparititve_title.json"
cache = embedding_model_test.init(cache_file)

starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]

compartive_title_buckets = embedding_model_test.get_cache(cache_file)
if compartive_title_buckets is None:
    embedding_model_test.start_cache(starting_comparitive_title_buckets)
    compartive_title_buckets = embedding_model_test.get_cache(cache_file)

comparitive_title_input = input("Enter comparitive titles: ")
comparitive_title_input = comparitive_title_input + " from these titles what are the closest 3 titles from the list of titles i will give and if they are not close enough elect to create a new title to match: " + str(compartive_title_buckets)

output = embedding_model_test.llm_call(comparitive_title_input)
print(output)