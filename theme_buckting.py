import embedding_model_test

from embedding_model_test import nearest_word

cache_file="cache_theme.json"
cache = embedding_model_test.init(cache_file)

start_theme = ["Love", "Sacrfice", "Sad", "Death", "Dark"]

Theme = embedding_model_test.get_cache(cache_file) # get the list of buckets from cache
if Theme is None: # if the is no cache file
    embedding_model_test.start_cache(start_theme) # add the starting elements to the cache so we have a base of buckets to start with
    Theme= embedding_model_test.get_cache(cache_file) # get the list of buckets from cache

print(Theme)
EMBEDDING_MODEL = "text-embedding-3-small"

max_distance = 0.7

word2 = input("input genre: ")
Dis_list = []

for theme in Theme:
    print("genre: ", theme)
    distance = nearest_word(theme, word2)
    Dis_list.append((theme, distance))

# Sort by distance
Dis_list.sort(key=lambda x: x[1])

# Print the results
for genre, distance in Dis_list:
    print(f"Genre: {genre}, Distance: {distance}")

# Find the closest genre
closest_distance = Dis_list[0]
print(closest_distance)
closest_genre = Dis_list[0][0]
if closest_distance[1]>max_distance:
    print("make new bucket for :", word2)
    embedding_model_test.new_bucket(word2)
    print("Sucessfully made new bucket for :", word2)
else:
    print(f"The closest genre is: {closest_genre}")
    # we here need to add the genre to the list of closest genres