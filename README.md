# Text Embedding and Bucket Categorization System

## Overview

This Python module provides functionality for generating text embeddings using OpenAI's API, managing a cache of embeddings, and categorizing words into buckets based on distance metrics. The module supports both **cosine similarity** and **Euclidean distance** for comparing embeddings.

---

## Installtion
You can install this library with:

`pip install [git+https://github.com/Rafipilot/embedding_bucketing/edit/main/README.md#get_embeddinginput_to_model](https://github.com/Rafipilot/embedding_bucketing)`

---

## Configuration and Initialization

### `config(apikey)`
Configures the OpenAI API client.

**Parameters:**
- `apikey` (*str*): OpenAI API key.

### `init(cache_file, starting_buckets)`
Initializes the caching system and loads existing buckets.

**Parameters:**
- `cache_file` (*str*): Path to the cache file.
- `starting_buckets` (*list*): Initial words to be categorized into buckets.

**Returns:**
- `cache` (*Cache object*): Cache instance managing embeddings.
- `bucket_array` (*list*): List of buckets.

---

## Embedding Functions

### `get_embedding(input_to_model)`
Fetches the embedding of a given text input from OpenAI's API.

**Parameters:**
- `input_to_model` (*str*): Text input to generate embedding for.

**Returns:**
- `embedding` (*list*): Embedding vector.

### `normalize(embedding)`
Normalizes an embedding vector.

**Parameters:**
- `embedding` (*numpy array*): Embedding vector.

**Returns:**
- `normalized_embedding` (*numpy array*): Normalized vector.

---

## Distance Calculation

### `nearest_word(cache, word1, word2)`
Finds the similarity between two words using **cosine similarity**.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `word1` (*str*): First word.
- `word2` (*str or numpy array*): Second word's embedding.

**Returns:**
- `distance` (*float*): Cosine distance between words.

### `nearest_word_E_D(cache, word1, word2)`
Calculates **Euclidean distance** between two words.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `word1` (*str*): First word.
- `word2` (*str or numpy array*): Second word's embedding.

**Returns:**
- `distance` (*float*): Euclidean distance between words.

---

## Bucket Management

### `new_bucket(cache, name)`
Creates a new bucket and stores its embedding in the cache.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `name` (*str*): Name of the bucket.

### `get_cache(cache_file)`
Retrieves a list of cached buckets.

**Parameters:**
- `cache_file` (*str*): Path to cache file.

**Returns:**
- `array` (*list*): List of bucket names.

### `start_cache(cache, starting_array)`
Populates the cache with initial buckets.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `starting_array` (*list*): Initial words to store as buckets.

---

## Language Model Call

### `llm_call(input_message)`
Calls OpenAI's Chat API to get a **one-word response**.

**Parameters:**
- `input_message` (*str*): Input message.

**Returns:**
- `response` (*str*): LLM-generated one-word response.

---

## Adjusting Embeddings

### `adjust(cache, word, word2)`
Adjusts a word’s embedding by averaging it with another word's embedding.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `word` (*str*): First word.
- `word2` (*str*): Second word.

### `adjusting_vectors(vec1, vec2)`
Averages two embedding vectors.

**Parameters:**
- `vec1` (*numpy array*): First embedding vector.
- `vec2` (*numpy array*): Second embedding vector.

**Returns:**
- `adjusted_embedding` (*numpy array*): Averaged vector.

---

## Auto-Sorting

### `auto_sort(cache, word, max_distance, bucket_array, type_of_distance_calc, amount_of_binary_digits)`
Automatically categorizes a word into the closest bucket or creates a new bucket if necessary.

**Parameters:**
- `cache` (*Cache object*): Cache instance.
- `word` (*str*): Word to categorize.
- `max_distance` (*float*): Threshold distance for creating a new bucket.
- `bucket_array` (*list*): List of existing bucket names.
- `type_of_distance_calc` (*str*): Distance metric (`"EUCLIDEAN DISTANCE"` or `"COSINE SIMILARITY"`).
- `amount_of_binary_digits` (*int*): Limit for binary digit representation of bucket IDs.

**Returns:**
- `closest_distance` (*tuple*): Closest bucket and distance.
- `closest_bucket` (*str*): Name of the closest bucket.
- `bucket_id` (*int*): ID of the closest bucket.
- `bucket_binary` (*numpy array*): Binary representation of bucket ID.

---

## Cache Class

### `Cache(cache_file)`
Manages caching of embeddings and bucket assignments.

**Attributes:**
- `cache` (*dict*): Stores word embeddings and IDs.
- `cache_file` (*str*): Path to cache file.
- `next_id` (*int*): Next available bucket ID.

**Methods:**
- `write_to_cache(key, embedding, assign_id=True)`: Stores embedding in cache.
- `read_from_cache(key)`: Retrieves word ID from cache.
- `get_id(key)`: Gets bucket ID for a word.
- `get_embedding_from_cache(key)`: Retrieves stored embedding.
- `save_cache()`: Saves cache to file.
- `load_cache()`: Loads cache from file.
- `clear_cache()`: Clears all cache data.

---

## Summary

This module facilitates **text embedding, categorization, and automatic sorting** of words into semantic clusters. It employs OpenAI embeddings, **cosine similarity**, and **Euclidean distance** to optimize text classification.

---

## Future Enhancements

- Debugging issues related to Euclidean distance sorting.
- Improving efficiency of cache handling.
- Extending auto-sort logic for larger datasets.
