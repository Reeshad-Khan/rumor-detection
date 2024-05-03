# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:47:25 2024

@author: gnkhata
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
#from transformers import BertModel, BertTokenizer,AdamW
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
#from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
#from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from Read_src_twt_only1 import read_src_twts
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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

# Function to train the model for multiclass
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,LR_RATE, BATCH_SIZE):
    target_names = ['true 0', 'false 1', 'unverified 2']
    best_model = './Best_Model/Src_Twt_Only/best_model3_'+str(num_epochs)+'epochs_'+str(LR_RATE)+'lrate_'+str(BATCH_SIZE)+'batch-size.pth'
    # Define variables to track the best validation loss and corresponding model state
    best_val_macro_f1 = float('-inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        train_predictions = []
        train_targets = []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted_labels = torch.max(outputs, dim=1)
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

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted_labels = torch.max(outputs, dim=1)
                total_val_correct += (predicted_labels == labels).sum().item()
                total_val_samples += labels.size(0)
                
                # Store predictions and true labels for F1 calculation
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_accuracy = total_val_correct / total_val_samples
        val_loss = val_loss / len(val_loader)
        # Compute F1 score for validation set
        val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
        val_f1_scores = f1_score(val_targets, val_predictions, average=None)
        
        # If the current model has a higher validation macro avg f1 than the best one so far, save it
        if best_val_macro_f1 < val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = model.state_dict()
            torch.save(best_model_state, best_model)
            print("Saving at epoch: ", epoch)

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
    print("Training ended, best model stored in: ", best_model)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("BATCH_SIZE",  help="Supply batch-size")
    parser.add_argument("LR_RATE",  help="Supply desired learning rate")
    parser.add_argument("NUM_EPOCHS",  help="Supply number of training epochs")
    args = parser.parse_args()
        
    
    # Define hyperparameters and settings
    MAX_LENGTH = 17
    BATCH_SIZE = int(args.BATCH_SIZE)
    LR_RATE = float(args.LR_RATE)
    NUM_EPOCHS = int(args.NUM_EPOCHS)
    NUM_CLASSES = 3  # Change this based on the number of classes in your dataset
    flag = "train"
    print("Training starts with :")
    print("Batch size: ", BATCH_SIZE, "\n Lr_rate: ", LR_RATE, "\n Num epochs:", NUM_EPOCHS)
    
    # Load pre-trained RoBERTa model and tokenizer
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Assuming you have loaded your dataset into texts and labels
    train_texts, val_texts, train_labels, val_labels = read_src_twts(flag)
    
    train_dataset = RumorDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = RumorDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize the model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RumorVerificationModel(roberta_model, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=LR_RATE)
    optimizer = AdamW(model.parameters(), lr=LR_RATE)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=LR_RATE)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=LR_RATE)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,LR_RATE, BATCH_SIZE)
    
    
    '''
    # Get predicted class indices
    predicted_labels = torch.argmax(predicted_probs, dim=1)
    
    # Compute F1 score for each class
    f1_scores = f1_score(true_labels, predicted_labels, average=None)
    
    # Compute macro-average F1 score
    macro_f1 = torch.tensor(f1_scores).mean().item()
    '''
