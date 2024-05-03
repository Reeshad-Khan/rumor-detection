import os
import json
import shutil
import requests
import matplotlib.pyplot as plt

# Constants for the API and directory setup
api_token = "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"  # Your Hugging Face API token
ORIGINAL_DATA_DIR = "/home/rk010/DM/Rumoureval2019/rumoureval-2019-training-data/twitter-english/sydneysiege"
NEW_DATA_DIR = "//home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english/sydneysiege"
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}

#charliehebdo  ebola-essien  ferguson  germanwings-crash  illary  ottawashooting  prince-toronto  putinmissing  sydneysiege

def setup_new_dataset():
    """ Copy the original dataset to a new location for modification. """
    if os.path.exists(NEW_DATA_DIR):
        print(f"Removing existing directory: {NEW_DATA_DIR}")
        shutil.rmtree(NEW_DATA_DIR)
    print(f"Copying from {ORIGINAL_DATA_DIR} to {NEW_DATA_DIR}")
    shutil.copytree(ORIGINAL_DATA_DIR, NEW_DATA_DIR)
    print("Copy complete.")

def query(payload):
    """ Send a request to the Hugging Face API to compute text similarities. """
    print("Sending API request...")
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        print("API request successful.")
    else:
        print(f"API request failed with status code {response.status_code}")
    return response.json()

def read_json_file(file_path):
    """ Utility function to read a JSON file. """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_text(json_data):
    """ Extract the 'text' field from the JSON data. """
    return json_data.get("text", "")

def find_similarities(post_text, replies_texts, reply_files):
    """ Query the API for similarities between the post and each reply. """
    payload = {
        "inputs": {
            "source_sentence": post_text,
            "sentences": replies_texts
        }
    }
    response = query(payload)
    if isinstance(response, list) and all(isinstance(score, float) for score in response):
        return list(zip(reply_files, response))
    else:
        print("Unexpected response format or error:", response)
        return []

def save_similarity_scores(scores, filename):
    """ Save similarity scores to a JSON file. """
    with open(os.path.join(NEW_DATA_DIR, filename), 'w') as f:
        json.dump(scores, f, indent=4)

def generate_histogram(scores, title, filename):
    """ Generate a histogram of similarity scores and save it as an image. """
    plt.figure()
    plt.hist(scores, bins=20, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(NEW_DATA_DIR, filename))
    plt.close()

def process_and_clean_dataset():
    all_scores = []
    processed_scores = []
    similarity_scores = []
    for post_id in os.listdir(NEW_DATA_DIR):
        post_dir = os.path.join(NEW_DATA_DIR, post_id)
        source_tweet_path = os.path.join(post_dir, 'source-tweet', f"{post_id}.json")
        replies_dir = os.path.join(post_dir, 'replies')
        
        print(f"Processing post {post_id}")
        if os.path.exists(source_tweet_path):
            post_data = read_json_file(source_tweet_path)
            post_text = extract_text(post_data)
            
            reply_files = os.listdir(replies_dir)
            replies_texts = [extract_text(read_json_file(os.path.join(replies_dir, rf))) for rf in reply_files]
            
            if replies_texts:
                similarities = find_similarities(post_text, replies_texts, reply_files)
                for reply_file, score in similarities:
                    similarity_scores.append({'post': post_id, 'reply': reply_file, 'score': score})
                    all_scores.append(score)
                    if score >= 0.50:
                        processed_scores.append(score)
                    else:
                        print(f"Removing low similarity reply {reply_file} with score {score}")
                        os.remove(os.path.join(replies_dir, reply_file))
    save_similarity_scores(similarity_scores, "charliehebdo_similarity_scores.json")
    generate_histogram(all_scores, "Initial Distribution of Similarity Scores", "initial_similarity_histogram.png")
    generate_histogram(processed_scores, "Filtered Distribution of Similarity Scores", "filtered_similarity_histogram.png")

# Setup new dataset directory
setup_new_dataset()

# Run the dataset processing and cleaning
process_and_clean_dataset()
