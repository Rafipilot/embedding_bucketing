import Module.embedding_model_test as embedding_model_test

from Module.embedding_model_test import nearest_word

cache_file="cache_genre.json"
cache = embedding_model_test.init(cache_file)
print(cache)

start_Genre = ["Drama", "Commedy", "Action", "romance", "documentry"]

Genre = embedding_model_test.get_cache(cache_file) # get the list of buckets from cache
if Genre is None: # if the is no cache file
    print("no file")
    embedding_model_test.start_cache(start_Genre) # add the starting elements to the cache so we have a base of buckets to start with
    Genre = embedding_model_test.get_cache(cache_file) # get the list of buckets from cache

print(Genre)
EMBEDDING_MODEL = "text-embedding-3-small"

max_distance = 0.7


word2 = input("input genre: ")
Dis_list = []

for genre_bucket in Genre:
    print("genre: ", genre_bucket)
    distance = embedding_model_test.nearest_word_E_D(genre_bucket, word2)
    Dis_list.append((genre_bucket, distance))

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
    embedding_model_test.adjust(word2, closest_genre)