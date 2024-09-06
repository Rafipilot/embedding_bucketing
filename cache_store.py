import json
import os
import numpy as np

class Cache:
    def __init__(self, cache_file):
        self.cache = {}
        self.cache_file = cache_file
        self.load_cache()

    def write_to_cache(self, key, value):
        """Store data in the cache with a given key."""
        # Convert NumPy array to list if necessary
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.cache[key] = value
        print(f"Added {key}: {value} to cache")
        self.save_cache()

    def read_from_cache(self, key):
        """Retrieve data from the cache."""
        if key in self.cache:
            print(f"Cache hit! {key}: {self.cache[key]}")
            # If the value was originally a NumPy array, convert it back
            if isinstance(self.cache[key], list):
                return np.array(self.cache[key])
            return self.cache[key]
        else:
            print(f"Cache miss! {key} not found.")
            return None

    def save_cache(self):
        """Save the cache to a file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print(f"Cache saved to {self.cache_file}")

    def load_cache(self):
        """Load the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"Cache loaded from {self.cache_file}")
        else:
            print("No cache file found. Starting with an empty cache.")

    def clear_cache(self):
        """Clear the cache in memory and delete the cache file."""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Cache cleared and cache file deleted.")



    def adjusting_vectors(self, key1, key2):
        # Check if the keys exist in the cache
        array1_np = np.array(key1)
        array2_np = np.array(key2)
  
        adjusted_embedding = (array1_np + array2_np) / 2
        return adjusted_embedding



