# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:18:03 2024

@author: gnkhata
"""

# Import necessary libraries
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

from Read_src_twt_only1 import read_src_twts
from torch.utils.tensorboard import SummaryWriter

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

writer = SummaryWriter('/home/rk010/DM/log')

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionHead, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # features => (batch_size, seq_length, hidden_dim)
        attention_scores = self.V(torch.tanh(self.W(features)))  # (batch_size, seq_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_features = attention_weights * features  # apply attention weights
        output = weighted_features.sum(axis=1)  # sum over the sequence dimension
        return output, attention_weights


class RumorVerificationModel(nn.Module):
    def __init__(self, lang_model, num_classes, dropout_rate=0.5, nhead=4, num_encoder_layers=2):
        super(RumorVerificationModel, self).__init__()
        self.l_model = lang_model
        self.attention_head = AttentionHead(lang_model.config.hidden_size)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=lang_model.config.hidden_size, nhead=nhead, dropout=dropout_rate),
            num_encoder_layers
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lang_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.l_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        attention_output, attention_weights = self.attention_head(sequence_output)
        encoded = self.transformer_encoder(attention_output.unsqueeze(1))
        dropped_out = self.dropout(encoded.squeeze(1))
        logits = self.fc(dropped_out)
        return logits

# Define custom dataset for multiclass
class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, l_model, seq_mode, lang_model):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.l_model = l_model
        self.seq_mode = seq_mode
        self.lang_model = lang_model.to(device)
        if self.seq_mode == "concat":
            print("Initialising Input seq in concat mode /n")
            print()
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.seq_mode == "one":
            if self.l_model == "gpt2":
                encoding = self.tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True
                )
    
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                token_type_ids = encoding["token_type_ids"]
    
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long),
                }
            else:
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'label': torch.tensor(label, dtype=torch.long)
                }
            
        elif self.seq_mode == "concat":
            '''
            sub_sequences = [text[i:i+self.max_length] for i in range(0, len(text), self.max_length)]
            # Encode each sub-sequence separately
            encoded_sub_sequences = []
            for sub_sequence in sub_sequences:
                encoded_input = self.tokenizer(sub_sequence, return_tensors='pt', padding=True, truncation=True)
                encoded_input = encoded_input.to(device)
                with torch.no_grad():
                    outputs = self.lang_model(**encoded_input)
                    encoded_sub_sequences.append(outputs.last_hidden_state)
            # Concatenate encoded sub-sequences
            concatenated_representation = torch.cat(encoded_sub_sequences, dim=1)
            return {
                'input_representation': concatenated_representation,
                'label': torch.tensor(label, dtype=torch.long)
            }
            '''
            sub_sequences = [text[i:i+self.max_length] for i in range(0, len(text), self.max_length)]
            # Encode each sub-sequence separately
            encoded_sub_sequences = []
            max_sequence_length = 0  # Track the maximum sequence length for padding
            for sub_sequence in sub_sequences:
                encoded_input = self.tokenizer(sub_sequence, return_tensors='pt', padding=True, truncation=True)
                encoded_input = encoded_input.to(device)
                with torch.no_grad():
                    outputs = self.lang_model(**encoded_input)
                    encoded_sub_sequences.append(outputs.last_hidden_state)
                    # Update the maximum sequence length
                    max_sequence_length = max(max_sequence_length, outputs.last_hidden_state.shape[1])
    
            # Pad the encoded sub-sequences to the maximum sequence length
            padded_encoded_sub_sequences = [torch.nn.functional.pad(seq, (0, max_sequence_length - seq.shape[1]), 'constant', 0) for seq in encoded_sub_sequences]
            print(padded_encoded_sub_sequences[1])
            # Concatenate padded encoded sub-sequences
            concatenated_representation = torch.cat(padded_encoded_sub_sequences, dim=1)
    
            return {
            'input_representation': concatenated_representation,
            'label': torch.tensor(label, dtype=torch.long)
            }
        
           

# Function to train the model for multiclass
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, LR_RATE, BATCH_SIZE, L_MODEL, DATA_MODE, SEQ_MODE, device):
    writer = SummaryWriter(f'runs/{DATA_MODE}_{L_MODEL}_{SEQ_MODE}')
    target_names = ['true 0', 'false 1', 'unverified 2']
    best_model = '/home/rk010/DM/Models/Best_Model/'+DATA_MODE+'/'+L_MODEL+'/1best_model_'+str(num_epochs)+'epochs_'+str(LR_RATE)+'lrate_'+str(BATCH_SIZE)+'batch-size.pth'
    # Define variables to track the best validation loss and corresponding model state
    
    if not os.path.exists('/home/rk010/DM/Models/Best_Model/'+DATA_MODE+'/'+L_MODEL):
        os.makedirs('/home/rk010/DM/Models/Best_Model/'+DATA_MODE+'/'+L_MODEL)  # Create the directory and any necessary parent directories

    best_val_macro_f1 = float('-inf')
    best_model_state = None
    if SEQ_MODE == "concat":
        print("Training in concat mode \n")
        print()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        train_predictions = []
        train_targets = []
        for batch in train_loader:
            if SEQ_MODE == "one":
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
    
                optimizer.zero_grad()
                
                if L_MODEL == "gpt2":
                    token_type_ids = batch["token_type_ids"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                    logits = outputs.logits
                    _, predicted_labels = torch.max(logits, 1)
                    loss = outputs.loss
                else:
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    _, predicted_labels = torch.max(outputs, dim=1)
            elif SEQ_MODE == "concat":
                input_rep = batch['input_representation'].to(device)
                labels = batch['label'].to(device)
    
                optimizer.zero_grad()
                
                outputs = model(input_rep)
                loss = criterion(outputs, labels)
                _, predicted_labels = torch.max(outputs, dim=1)
                
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            total_train_correct += (predicted_labels == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # Store predictions and true labels for F1 calculation
            train_predictions.extend(predicted_labels.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        train_accuracy = total_train_correct / total_train_samples
        train_loss = train_loss / len(train_loader)
        # Compute F1 score for training set
        train_f1 = f1_score(train_targets, train_predictions, average='macro')
        tr_f1_scores = f1_score(train_targets, train_predictions, average=None)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('F1_Score/Train', train_f1, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                if SEQ_MODE == "one":
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
                elif SEQ_MODE == "concat":
                    input_rep = batch['input_representation'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_rep)
                    loss = criterion(outputs, labels)
                    _, predicted_labels = torch.max(outputs, dim=1)
                    
                val_loss += loss.item()

                # Calculate validation accuracy
                total_val_correct += (predicted_labels == labels).sum().item()
                total_val_samples += labels.size(0)
                
                # Store predictions and true labels for F1 calculation
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_accuracy = total_val_correct / total_val_samples
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        # Compute F1 score for validation set
        val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
        val_f1_scores = f1_score(val_targets, val_predictions, average=None)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('F1_Score/Validation', val_macro_f1, epoch)
        
        # If the current model has a higher validation macro avg f1 than the best one so far, save it
        if best_val_macro_f1 < val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = model.state_dict()
            torch.save(best_model_state, best_model)
            print()
            print("Saving at epoch: ", epoch, "Macro avg f1: ", best_val_macro_f1)
            print()
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, '
              f'Train macro F1: {train_f1:.4f}, '
              f'Val macro F1: {val_macro_f1:.4f}')
        #print("Train f1 scores: ", tr_f1_scores)
        #print("Val f1 scores: ", val_f1_scores)
        
        print(classification_report(val_targets, val_predictions, target_names=target_names, labels=np.unique(val_predictions)))
        print(confusion_matrix(val_targets, val_predictions, labels=np.unique(val_predictions)))
        print()
    writer.close()
    print("Training ended, best model stored in: ", best_model)

def load_model(l_model, device, num_classes):
    if L_MODEL == "bert":
        print("Training BERT model:")
        # Load tokenizer and BERT model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lang_model = BertModel.from_pretrained('bert-base-uncased')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    
    elif L_MODEL == "roberta":
        # Load pre-trained RoBERTa model and tokenizer
        print("Training RoBERTa model:")
        lang_model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    elif L_MODEL == "gpt2":
        # Load pre-trained GPT2 model and tokenizer
        print("Training GPT2 model:")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lang_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
        lang_model.config.pad_token_id = lang_model.config.eos_token_id
        model = lang_model.to(device)
    return tokenizer, lang_model, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("L_MODEL", help="Pre-trained language model to use")
    parser.add_argument("BATCH_SIZE", type=int, help="Supply batch-size")  # Ensure BATCH_SIZE is an integer
    parser.add_argument("LR_RATE", type=float, help="Supply desired learning rate")  # Ensure LR_RATE is a float
    parser.add_argument("NUM_EPOCHS", type=int, help="Supply number of training epochs")  # Ensure NUM_EPOCHS is an integer
    parser.add_argument("DATA_MODE", help="Supply data mode, e.g., src_twt_only, src_twt_all_replies")
    parser.add_argument("SEQ_MODE", help="Seq length mode, e.g., concat, one")
    args = parser.parse_args()
    
    # Now BATCH_SIZE, LR_RATE, and NUM_EPOCHS are correctly typed
    print(f"Training with batch size {args.BATCH_SIZE}, learning rate {args.LR_RATE}, for {args.NUM_EPOCHS} epochs.")
        
    # Define hyperparameters and settings
    L_MODEL = args.L_MODEL
    BATCH_SIZE = int(args.BATCH_SIZE)
    LR_RATE = float(args.LR_RATE)
    NUM_EPOCHS = int(args.NUM_EPOCHS)
    DATA_MODE = args.DATA_MODE
    SEQ_MODE  = args.SEQ_MODE
    if DATA_MODE == "src_twt_only":
        MAX_LENGTH = 17
    elif DATA_MODE == "src_twt_all_replies":
        MAX_LENGTH = 361
    NUM_CLASSES = 3  # Change this based on the number of classes in your dataset
    flag = "train"
    print("Training starts with :")
    print("Batch size: ", BATCH_SIZE, "\n Lr_rate: ", LR_RATE, "\n Num epochs:", NUM_EPOCHS)
    # Initialize the model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer based on selected L_MODEL
    tokenizer, lang_model, model = load_model(args.L_MODEL, device, NUM_CLASSES)

        # Read all data
    train_txt, val_txt, train_label, val_label = read_src_twts("train", args.DATA_MODE)

    # Convert to lists if they are not already (assumed to be pandas Series)
    train_txt = train_txt.tolist()
    val_txt = val_txt.tolist()
    train_label = train_label.tolist()
    val_label = val_label.tolist()

    # Combine train and validation sets for cross-validation
    texts = train_txt + val_txt
    labels = train_label + val_label

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Using 5 folds for cross-validation
    fold = 0
    for train_index, val_index in kf.split(texts):
        fold += 1
        print(f"Training fold {fold}/5")
        
        # Split data into train and validation for the current fold
        train_texts, val_texts = [texts[i] for i in train_index], [texts[i] for i in val_index]
        train_labels, val_labels = [labels[i] for i in train_index], [labels[i] for i in val_index]
        
        # Initialize tokenizer and model based on selected language model
        if args.L_MODEL == "bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            lang_model = BertModel.from_pretrained('bert-base-uncased')
        elif args.L_MODEL == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            lang_model = RobertaModel.from_pretrained('roberta-base')
        elif args.L_MODEL == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            lang_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=NUM_CLASSES)
            tokenizer.pad_token = tokenizer.eos_token
            lang_model.config.pad_token_id = tokenizer.eos_token_id
        
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)

        # Create datasets and dataloaders for training and validation sets
        train_dataset = RumorDataset(train_texts, train_labels, tokenizer, MAX_LENGTH, args.L_MODEL, args.SEQ_MODE, lang_model)
        val_dataset = RumorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH, args.L_MODEL, args.SEQ_MODE, lang_model)
        train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
        
        # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        # Setup optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR_RATE, weight_decay=1e-5)

        # Setup learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.NUM_EPOCHS, args.LR_RATE, args.BATCH_SIZE, args.L_MODEL, args.DATA_MODE, args.SEQ_MODE, device)
