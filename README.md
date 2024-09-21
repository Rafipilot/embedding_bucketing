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

## How it Works

### 1. Word Embeddings
The program queries OpenAI's API to generate word embeddings for any given word or phrase. These embeddings are then compared using various metrics such as cosine similarity or Euclidean distance.

### 2. Nearest Word
The nearest_word() function compares two words and finds the semantic distance between them, using cosine similarity. The lower the distance, the more similar the words are.

### 3. New Buckets
New categories or "buckets" can be created with the new_bucket() function. Each bucket represents a word whose embedding is stored in the cache.

### 4. Caching System
Embeddings are saved in a cache file (in JSON format) to prevent redundant API calls. This improves performance by allowing for quick lookups of word embeddings from the cache when they have already been generated.

### 5. Averaging Embeddings
The program supports blending the meaning of buckets by averaging their embeddings using the adjust() function, which can then be stored in the cache for future use.



## Documentation

#### Note: use ```import embedding_buckting.embedding_model_test``` to get the model

1.
```bash
   config(apikey)
```
You need to pass your Openai api key through to the config function to setup the openai client.


2.
 ```bash
   init(cache_file_name)
  ```
Initializes the cache system by loading the cache from a file or creating a new one.

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
Creates a new word bucket by generating the embedding of the word and saving it to the cache.

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
12
```bash
   auto_sort(input_word, max_distance, bucket_array, type_of_distance_calc) 
  ```
Max Distance: max distance between the closest bucket and the input word

Type of Distance calc must be either "EUCLIDEAN DISTANCE" or "COSINE_SIMILARITY" if input is not valid then auto uses euclidean distance

Returns the closest distance and the closest bucket


## Caching system
The caching feature speeds up the embedding lookup process. By saving previously generated embeddings, the program avoids redundant API calls to OpenAI, which reduces latency and API usage costs.

## Example Usage
```shell
import embedding_buckting.embedding_model_test as em  # importing relevent modules

from config import openai_key  # importing your personal openai key you should have your api key in a seperate folder in gitgnore

em.config(openai) # setting up the module, here you pass your personal Openai api key through

cache_file="cache_genre.json"  # name of the cache file to save the embedding and their buckts in
cache = em.init(cache_file) # initializing your cache file with name cache_file

start_Genre = ["Drama", "Commedy", "Action", "romance", "documentry"]  # starting array of bucket, if there are no buckets found then this is the list of buckets that will be used

Genre = em.get_cache(cache_file) # get the list of buckets from cache  
if Genre is None: # if the is no cache file
    print("no file")
    em.start_cache(start_Genre) # add the starting array to the cache so we have a base of buckets to start with Note: this automatically saves the embeddings with the associated word
    Genre = em.get_cache(cache_file) # get the list of buckets from cache now that we have added the starting array of buckets

input_genre = input("Input a genre: ")

max_distance = 0.7 # max distance a word can be from the closest bucket before we create a new bucket
closest_distance, closest_genre = em.auto_sort(input_genre, max_distance, Genre, type_of_distance_calc="EUCLIDEAN DISTANCE") one  # using autosort to get the closest distance and closest bucket
print(closest_genre) # printing the closest bucket's name
```
Look at our examples file in our repo to get a better idea of how it works in practice!
