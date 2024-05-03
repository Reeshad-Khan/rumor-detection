import os
import json
import shutil
import torch
from tqdm import tqdm  # for progress bars
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').to(device)
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device

def paraphrase_text(text, model, tokenizer, device):
    try:
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=200, early_stopping=True)
        paraphrased_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return paraphrased_text
    except Exception as e:
        print(f"Error processing text: {text[:60]}... Error: {str(e)}")
        return text  # Return original text if error occurs

def process_json_file(file_path, model, tokenizer, device, pbar):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        original_text = data.get('text', '')
        if original_text:
            paraphrased_text = paraphrase_text(original_text, model, tokenizer, device)
            data['text'] = paraphrased_text
            pbar.set_postfix_str(f"Original: {original_text[:30]}... Paraphrased: {paraphrased_text[:30]}...")
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to process file {file_path}: {str(e)}")

def walk_directory(directory, model, tokenizer, device):
    total_files = sum(len(files) for _, _, files in os.walk(directory) if files)
    with tqdm(total=total_files, desc="Paraphrasing files") as pbar:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    process_json_file(file_path, model, tokenizer, device, pbar)
                    pbar.update(1)

def copy_dataset(original_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(original_dir, target_dir)
    for entry in os.listdir(target_dir):
        full_path = os.path.join(target_dir, entry)
        if os.path.isdir(full_path):
            new_name = full_path + "v2"
            os.rename(full_path, new_name)
    print(f"Dataset copied to: {target_dir} and top-level directories renamed")

if __name__ == "__main__":
    original_directory = "/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english/sydneysiege"
    new_directory = "/home/rk010/DM/new_test"
    copy_dataset(original_directory, new_directory)
    model, tokenizer, device = load_model()
    #new_directory_v2 = new_directory + "v2"
    walk_directory(new_directory, model, tokenizer, device)
