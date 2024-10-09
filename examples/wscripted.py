#################
# PREPARING BINARY INPUT DATA FOR API.AOLABS.AI

# import embedding bucketing; to install from the repo: pip install git+https://github.com/Rafipilot/embedding_bucketing
import embedding_bucketing.embedding_model_test as em
import numpy as np
# you'll need an OpenAPI key
from config import openai  
em.config(openai)

#Constants
max_distance = 0.55 # max distance a word can be from the closest bucket before we create a new bucket
type_of_distance_calc="COSINE SIMILARITY" # another option to try is "EUCLIDEAN DISTANCE"   
amount_of_binary_digits= 10 #Amount of binary digits for the bucket to be encoded into e.g if it is 5 Romance may be 10010 for example

#Function to get the closest bucket and its binary encoding
def embedding_bucketing_response(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits):
    sort_response = em.auto_sort(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits) 

    closest_distance = sort_response[0]
    closest_bucket   = sort_response[1]  # which bucket the uncategorized_input was placed in
    bucket_id        = sort_response[2]  # the id of the closest_bucket
    bucket_binary    = sort_response[3]  # binary representation of the id for INPUT into api.aolabs.ai
    return closest_bucket, bucket_binary # returning the closest bucket and its binary encoding



starting_genre_buckets= ["Comedy", "Drama", "Action"]
cache_file_name = "cache_genre.json"  # will be saved or loaded from your current working directory
cache, genre_buckets = em.init(cache_file_name, starting_genre_buckets)
uncategorized_genre_input  = "Documentary"
#Call for genre
closest_genre, genre_encoding = embedding_bucketing_response(uncategorized_genre_input, max_distance, genre_buckets, type_of_distance_calc, amount_of_binary_digits)
print("Closest Genre to", uncategorized_genre_input, "is", closest_genre)




cache_file_theme="cache_theme.json"
start_theme = ["Love", "Sacrifice", "Sad", "Death", "Dark"]
cache_theme, theme_buckets = em.init(cache_file_theme, start_theme)
uncategorized_theme_input = "Romance"
# Call for theme
closest_theme, theme_encoding = embedding_bucketing_response(uncategorized_theme_input, max_distance, theme_buckets, type_of_distance_calc, amount_of_binary_digits)
print("closest theme to", uncategorized_theme_input, "is", closest_theme)




cache_file_comp="cache_comparititve_title.json"
starting_comparitive_title_buckets = ["romeo and julliet", "the great gatsby", "harry potter", "oliver twist", "an inspector calls" ]
cache_comp, comparative_title_buckets = em.init(cache_file_comp, starting_comparitive_title_buckets)
uncategorized_comparative_input = ["Beauty and the beast", "The red october", "the big short"]

#Call for comparative title#
comparative_title_encodings = []
closest_comparative_title_buckets= []
for i in range(len(uncategorized_comparative_input)):
    closest_comp_bucket, comparative_title_encoding = embedding_bucketing_response(uncategorized_comparative_input[i], max_distance, comparative_title_buckets, type_of_distance_calc, amount_of_binary_digits)
    comparative_title_encodings.append(comparative_title_encoding)
    closest_comparative_title_buckets.append(closest_comp_bucket)
    print("Encoded: ", uncategorized_comparative_input[i], "into", "Bucket:", closest_comp_bucket, "With binary encoding", comparative_title_encoding)




print("Inputs Closest to:", closest_genre, closest_theme, "Closest comparative titles: ", closest_comparative_title_buckets)



ao_input_binary_array = theme_encoding+genre_encoding+comparative_title_encodings
print("ao binary input:", theme_encoding, genre_encoding, comparative_title_encodings)
print("input to ao:",ao_input_binary_array)

INPUT_AO_api = ''.join(map(str,ao_input_binary_array))  # right now, our API accepts only binary strings as input

FULL_INPUT_AO_api = INPUT_AO_api + INPUT_AO_api + INPUT_AO_api



#################
# UPLOADING AGENT ARCH TO CREATE A KENNEL

# WScripted Agent Architecture 
# https://gist.github.com/mi3law/612f7cc6b9cbe96ba7304bad16931087

import requests

url = "https://api.aolabs.ai/v0dev/kennel"

payload = {
    "kennel_name": "WScripted_Curator",
    "arch_URL": "https://gist.githubusercontent.com/mi3law/612f7cc6b9cbe96ba7304bad16931087/raw/5ea3b07f89a2b5eeacbc7045f841a8d48df0346a/arch__WScripted_Curator.py",
    "description": "WScripted- Personal Curation Agent.  Takes in Genre, Theme, and Comparative Title (10 binary neurons each) for a piece of content to offer the user a recommendation, learning from subsequent individual user interaction.",
    "permissions": "private"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)



#################
# CALLING AGENTS PER USER FROM THE KENNEL

import requests

url = "https://api.aolabs.ai/v0dev/kennel/agent"

payload = {
    "kennel_id": "WScripted_Curator",  # use kennel_name entered above
    "agent_id": "1st_USER_ID",   # enter unique user IDs here, to call a unique agent for each ID
    "INPUT": FULL_INPUT_AO_api,  # pass through the input from embedding_bucketing.auto_sort, adding any other inputs
    "control": {
        "CN": False,             # set as True when the user clicks "Good Recommendation," this will trigger learning to positively reinforce the agent's recommendation
        "CP": False,             # set as True when the user clicks "Bad Recommendation," this will trigger learning to **negatively** reinforce the agent's recommendation
        "US": False,
        "neuron": {
            "DD": True,
            "Hamming": True,
            "Default": True
        }
    }
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)