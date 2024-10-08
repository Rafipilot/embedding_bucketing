# Bucketing system 

## Overview
This Python program enables users to compare words or phrases using semantic embeddings, powered by OpenAI's language models. It uses vector embeddings to compute distances between words based on their semantic meanings, allowing for tasks like finding the nearest word, averaging word embeddings, and comparing distances between words. The program also incorporates a caching system to store and retrieve word embeddings, making future comparisons faster and more efficient.

The application of this program is ideal in cases where predefined/select outputs are required from variable inputs, such as in situations where a system needs to group or "bucket" words into similar categories based on their semantic similarity. 


## Get set up: 
```bash
pip install git+https://github.com/Rafipilot/embedding_bucketing
```


## Features
### 1. Semantic Similarity:
Uses OpenAI's embeddings to compute the semantic similarity or distance between words.
### 2. Word Bucketing: 
Groups words into predefined categories based on their similarity, useful when a system requires a limited set of outputs from many potential inputs.
### 3. Cache System: 

Efficiently stores embeddings in a JSON cache file, reducing API calls and speeding up repeated computations.
### 4. Support for Multiple Distance Metrics:
Cosine Similarity, Euclidean Distance + more coming
### 5. Averaging of Embeddings:
Allows for blending the meanings of two words by averaging their embeddings and storing the result.

## How to use it
##### Example for recommender systems where llm extracts the genre input

#### 1. Configure the model
Since we are using openai's embedding model encode words into embedding vectors we need to pass our openai api key through the config() function to configure the openai client.

#### 2. Load cache, or make one if we don't have one
To load or create our cache file we need to use the init() function and pass through our file name and our array of starting genre buckets( for if we need to start up the cache). This function returns the cache and the buckets, in this case different genres.

#### 3. We are ready to go!
Use the autosort function to get: 
   1. Closest distance to the closest genre
   2. Closest genre
   3. The genre's unqiue ID
   4. The genre's unique binary encoding

By passing through:
   1. Input genre
   2. Max distance before new bucket creation
   3. Type of distance calculation
   4. The amount of binary digits you want the unique numerical id to be encoded in


## Documentation

#### Note: use ```import embedding_bucketing.embedding_model_test``` to get the model

1.
```bash
   config(apikey)
```
You need to pass your Openai api key through to the config function to setup the openai client.


2.
 ```bash
   init(cache_file_name, starting_bucket_array)
  ```
Initializes the cache system by loading the cache from a file or creating a new one, if one does not exist, by using the "starting_bucket_array" as buckets to use initally.

3.
 ```bash
   get_embedding(input_to_embedding_model)
  ```
Returns the embedding for a given word/phrase using OpenAI's embedding API.

4.
 ```bash
   nearest_word(word1, word2)
  ```
Compares two words using their cosine similarity and returns the semantic distance between them.

5.
 ```bash
   nearest_word_E_D(word1, word2)
  ```
Compares two words using Euclidean distance and returns the semantic distance between their embeddings.

6.
 ```bash
   new_bucket(name)
  ```
Creates a new word bucket by generating the embedding of the word and saving it to the cache and assigns it a unique numerical id.

7.
 ```bash
   llm_call(input_message)
  ```
Makes a call to an LLM (e.g., GPT-3.5) to get a one-word answer based on the input and returns the output.

8.
 ```bash
   adjust(word, word2)
  ```
Averages the embeddings of two words and replaces the old vector of the second word with the new averaged vector in the cache but keeps the name the same.

9 (not yet implemented). 
 ```bash
   averaging_and_compare(word1, word2)
  ```

10
```bash
    get_cache(cache_file_name)
```
Returns array of the existing cache file and returns none if it does not exist.

11.
 ```bash
   start_cache(starting_array)
```
Writes to the cache the starting array of buckets. e.g if you didnt have any buckets to read from the cache you would call this function and pass through a array of buckets to write to the cache.
12
```bash
   auto_sort(input_word, max_distance, bucket_array, type_of_distance_calc, amount_of_binary_digits) 
  ```
Max Distance: max distance between the closest bucket and the input word

Amount_of_binary_digits: the amount of binary digits to return the bucket in 
(Note: if you have an insufficent amount of binary digits an error message will be prinited and the bucket will not be encoded into binary)

Type of Distance calc must be either "EUCLIDEAN DISTANCE" or "COSINE_SIMILARITY" if input is not valid then auto uses euclidean distance

Returns the closest distance, closest bucket, unique bucket ID and the unique bucket binary encoding in the amount of digits specified


## Caching system
The caching feature speeds up the embedding lookup process. By saving previously generated embeddings, the program avoids redundant API calls to OpenAI, which reduces latency and API usage costs. Also the cache stores a unqiue numerical id per bucket so we can easily convert it into the relevent amount of binary digits.

## Example Usage
#### Comparing an input genre to an array of genres and returning the most semantically similar one
```shell
import embedding_buckting.embedding_model_test as em  # importing relevent modules

em.config("Your personal api key") # setting up the module, here you pass your personal Openai api key through
cache_file_name="cache_genre.json"  # name of the cache file to save the embedding and their buckts in

start_Genre = ["Drama", "Commedy", "Action", "romance", "documentry"]  # starting array of buckets, if there are no buckets found then this is the list of buckets that will be used
cache, Genre = em.init(cache_file, start_Genre) # init cache return cache object and the array of buckets, in this case genres

input_genre = input("Input a genre: ") #User input
max_distance = 0.5 # max distance a word can be from the closest bucket before we create a new bucket

closest_distance, closest_genre, bucket_id, bucket_binary_encoding = em.auto_sort(input_genre, max_distance, Genre, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits = 8)  # using autosort 
print(closest_genre) # printing the closest bucket's name
```
Look at our examples file in our repo to get a better idea of how it works in practice!
