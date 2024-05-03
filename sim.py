import os
import json
import shutil
import requests
import matplotlib.pyplot as plt
import time


# Constants for the API and directory setup
api_token = "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"  # Your Hugging Face API token
ORIGINAL_DATA_DIR = "/home/rk010/DM/Rumoureval2019/rumoureval-2019-training-data/twitter-english"
NEW_DATA_DIR = "/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english"
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}

def setup_new_dataset():
    if os.path.exists(NEW_DATA_DIR):
        print(f"Removing existing directory: {NEW_DATA_DIR}")
        shutil.rmtree(NEW_DATA_DIR)
    print(f"Copying from {ORIGINAL_DATA_DIR} to {NEW_DATA_DIR}")
    shutil.copytree(ORIGINAL_DATA_DIR, NEW_DATA_DIR)
    print("Copy complete.")



def query(payload):
    """Send a request to the Hugging Face API to compute text similarities with handling for rate limits."""
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        print("API request successful.")
        return response.json()
    elif response.status_code == 429:
        print("Rate limit reached. Waiting for 1 hour before retrying...")
        time.sleep(3600)  # Wait for one hour
        return query(payload)  # Retry the same request
    else:
        print(f"API request failed with status code {response.status_code}. No retry.")
        return None


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def extract_text(json_data):
    return json_data.get("text", "")

def find_similarities(post_text, replies_texts, reply_files):
    payload = {"inputs": {"source_sentence": post_text, "sentences": replies_texts}}
    response = query(payload)
    return list(zip(reply_files, response)) if isinstance(response, list) and all(isinstance(score, float) for score in response) else []

def save_similarity_scores(scores, folder_name):
    file_path = os.path.join(NEW_DATA_DIR, f"{folder_name}_similarity_scores.json")
    with open(file_path, 'w') as f:
        json.dump(scores, f, indent=4)

def generate_histogram(scores, title, filename):
    plt.figure()
    plt.hist(scores, bins=20, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(NEW_DATA_DIR, filename))
    plt.close()

def process_and_clean_dataset(folder):
    folder_path = os.path.join(NEW_DATA_DIR, folder)
    for post_id in os.listdir(folder_path):
        post_dir = os.path.join(folder_path, post_id)
        structure_path = os.path.join(post_dir, "structure.json")
        source_tweet_path = os.path.join(post_dir, 'source-tweet', f"{post_id}.json")
        replies_dir = os.path.join(post_dir, 'replies')

        if os.path.exists(source_tweet_path):
            post_data = read_json_file(source_tweet_path)
            post_text = extract_text(post_data)
            reply_files = os.listdir(replies_dir)
            replies_texts = [extract_text(read_json_file(os.path.join(replies_dir, rf))) for rf in reply_files]
            similarities = find_similarities(post_text, replies_texts, reply_files)

            # Update structure.json
            if os.path.exists(structure_path):
                structure_data = read_json_file(structure_path)

            for reply_file, score in similarities:
                if score < 0.20:
                    os.remove(os.path.join(replies_dir, reply_file))
                    # Remove entry from structure.json
                    structure_data[post_id].pop(reply_file.split('.')[0], None)
                else:
                    structure_data[post_id][reply_file.split('.')[0]] = []

            write_json_file(structure_path, structure_data)

    # Save and generate histograms after processing all posts
    save_similarity_scores(similarities, folder)
    all_scores = [score for _, score in similarities]
    processed_scores = [score for _, score in similarities if score >= 0.50]
    generate_histogram(all_scores, f"Initial Distribution of Similarity Scores in {folder}", f"{folder}_initial_similarity_histogram.png")
    generate_histogram(processed_scores, f"Filtered Distribution of Similarity Scores in {folder}", f"{folder}_filtered_similarity_histogram.png")

# Setup new dataset directory
setup_new_dataset()

# Run the dataset processing and cleaning for each folder
for topic_folder in os.listdir(NEW_DATA_DIR):
    process_and_clean_dataset(topic_folder)
