import os
import json
import shutil
import requests
import time

# Constants
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
DATA_DIR = "/home/rk010/DM/Rumoureval2019/rumoureval-2019-training-data/reddit-dev-data"

def send_api_request(payload):
    """ Send a request to the API and handle the response. """
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        try:
            return response.json()  # Assume response is directly the list of scores
        except json.JSONDecodeError:
            print("Failed to decode JSON from response.")
            return None
    else:
        print(f"API request failed: {response.status_code} {response.text}")
        return None

def process_and_filter_replies(dataset_dir):
    """ Process replies based on similarity score and remove those below threshold. """
    source_post_path = os.path.join(dataset_dir, 'source-tweet', os.listdir(os.path.join(dataset_dir, 'source-tweet'))[0])
    replies_dir = os.path.join(dataset_dir, 'replies')

    if not os.path.exists(source_post_path) or not os.path.exists(replies_dir):
        print("Source post or replies directory does not exist.")
        return

    with open(source_post_path, 'r') as file:
        post_data = json.load(file)
    post_text = post_data['data']['children'][0]['data'].get('title', '') + " " + post_data['data']['children'][0]['data'].get('selftext', '')

    replies = []
    reply_files = os.listdir(replies_dir)
    for rf in reply_files:
        with open(os.path.join(replies_dir, rf), 'r') as file:
            reply_data = json.load(file)
            replies.append(reply_data['data'].get('body', ''))

    payload = {"inputs": {"source_sentence": post_text, "sentences": replies}}
    scores = send_api_request(payload)
    if scores:
        for reply_file, score in zip(reply_files, scores):
            if score < 0.50:
                os.remove(os.path.join(replies_dir, reply_file))
                print(f"Removed {reply_file} due to low similarity score: {score}")
    else:
        print("API did not return valid results.")

def main():
    for subdir in os.listdir(DATA_DIR):
        subdir_path = os.path.join(DATA_DIR, subdir)
        print(f"Processing {subdir_path}...")
        process_and_filter_replies(subdir_path)
        time.sleep(10)  # Pause between requests

if __name__ == "__main__":
    main()
