import embedding_model_test
from embedding_model_test import nearest_word
#Buckets
Genre=["Drama", "Commedy", "Action", "romance", "documentry"]

EMBEDDING_MODEL = "text-embedding-3-small"

max_distance = 0.7


# Call the function to find nearest words
word2 = input("input genre: ")
Dis_list = []

for genre in Genre:
    print("genre: ", genre)
    distance = nearest_word(genre, word2)
    Dis_list.append((genre, distance))

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
    print("make new bucket")

print(f"The closest genre is: {closest_genre}")