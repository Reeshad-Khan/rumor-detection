# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:21:27 2024

@author: gnkhata
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader
#from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from Read_src_twt_only1 import read_src_twts
from Rumour_src_twt_only_basic import RumorDataset, RumorVerificationModel


# Function to train the model for multiclass
def test_model(model, test_loader, criterion):
    print("Testing model...")
    target_names = ['true 0', 'false 1', 'unverified 2']
    
    # Validation
    #model.eval()
    test_loss = 0.0
    total_test_correct = 0
    total_test_samples = 0
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if L_MODEL == "gpt2":
                token_type_ids = batch["token_type_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                _, predicted_labels = torch.max(logits, 1)
            else:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                _, predicted_labels = torch.max(outputs, dim=1)

            test_loss += loss.item()

            # Calculate validation accuracy            
            total_test_correct += (predicted_labels == labels).sum().item()
            total_test_samples += labels.size(0)
            
            # Store predictions and true labels for F1 calculation
            test_predictions.extend(predicted_labels.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_accuracy = total_test_correct / total_test_samples
    test_loss = test_loss / len(test_loader)
    # Compute F1 score for validation set
    test_macro_f1 = f1_score(test_targets, test_predictions, average='macro')
    test_f1_scores = f1_score(test_targets, test_predictions, average=None)
    
    print(f'Test Loss: {test_loss:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}, '
          f'Test macro F1: {test_macro_f1:.4f}')
    #print("Train f1 scores: ", tr_f1_scores)
    #print("Val f1 scores: ", val_f1_scores)
    #print(classification_report(test_targets, test_predictions, target_names=target_names))
    #print(confusion_matrix(test_targets, test_predictions, labels=np.unique(test_predictions)))
    print(classification_report(test_targets, test_predictions, target_names=target_names))
    print(confusion_matrix(test_targets, test_predictions))
    print()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("L_MODEL",  help="Pre-trained language model to use")
    parser.add_argument("BATCH_SIZE",  help="Supply batch-size")
    parser.add_argument("DATA_MODE",  help="Supply data mode, e.g., src_twt_only, src_twt_all_replies")
    parser.add_argument("SEQ_MODE",  help="Seq length mode, e.g., concat, one")
    args = parser.parse_args()
        
    
    # Define hyperparameters and settings
    L_MODEL = args.L_MODEL
    BATCH_SIZE = int(args.BATCH_SIZE)
    DATA_MODE = args.DATA_MODE
    SEQ_MODE  = args.SEQ_MODE
    NUM_CLASSES = 3  # Change this based on the number of classes in your dataset
    #best_model_pth = '/home/rk010/DM/Models/Best_Model/src_twt_all_replies/roberta/1best_model_100epochs_1e-05lrate_16batch-size.pth'
    #best_model_pth = '/home/rk010/DM/Models/Best_Model/src_twt_all_replies/roberta/1best_model_100epochs_1e-05lrate_16batch-size.pth'
    best_model_pth = '/home/rk010/DM/Models/Best_Model/src_twt_all_replies/bert/1best_model_100epochs_1e-05lrate_16batch-size.pth'
    '''
    best_model_pth = './Best_Model/Src_Twt_Only/best_model_gpt2_52epochs_1e-05lrate_16batch-size.pth'
    best_model_pth = './Best_Model/Src_Twt_Only/best_model_50epochs_3e-05lrate_16batch-size.pth'
    best_model_pth = './Best_Model/Src_Twt_Only/best_model3_52epochs_1e-05lrate_16batch-size.pth'
    '''
    print()
    print("Batch size: ", BATCH_SIZE)
    if DATA_MODE == "src_twt_only":
        MAX_LENGTH = 17
    elif DATA_MODE == "src_twt_all_replies":
        MAX_LENGTH = 512
        
    # Initialize the model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if L_MODEL == "bert":    
        # Load  tokenizer and bert model
        print("Testing BERT model:")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lang_model = BertModel.from_pretrained('bert-base-uncased')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    
    elif L_MODEL == "roberta":
        # Load pre-trained RoBERTa model and tokenizer
        print("Testing RoBERTa model:")
        lang_model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    elif L_MODEL == "gpt2":
        # Load pre-trained gpt2 model and tokenizer
        print("Testing gpt2 model:")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lang_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
        lang_model.config.pad_token_id = lang_model.config.eos_token_id
        model = lang_model.to(device)
    
    # Lodaing test texts and labels
    flag = "test"
    test_texts, test_labels = read_src_twts(flag, DATA_MODE)
    
    test_dataset = RumorDataset(test_texts, test_labels, tokenizer, MAX_LENGTH,L_MODEL,SEQ_MODE,lang_model)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    model.load_state_dict(torch.load(best_model_pth))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Test the model
    test_model(model, test_loader, criterion)
    
    
