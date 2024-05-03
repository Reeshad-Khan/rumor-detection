# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:00 2024

@author: gnkhata
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:30:08 2023

@author: gnkhata
"""
import pandas as pd
import os
import json

def concatenate_samples(twt_bfs_threads_train, redt_bfs_threads_train, twt_bfs_threads_val, redt_bfs_threads_val):
    train_data  = pd.concat([twt_bfs_threads_train, redt_bfs_threads_train], ignore_index=True)
    val_data    = pd.concat([twt_bfs_threads_val, redt_bfs_threads_val], ignore_index=True)
    train_data  = train_data.dropna()#drop null entries
    train_data  = train_data.sample(frac = 1)#shuffle tuples
    val_data  = val_data.dropna()#drop null entries
    val_data  = val_data.sample(frac = 1)#shuffle tuples
    train_txt   = train_data["Text"]
    train_label = train_data["Veracity_int"]
    val_txt   = val_data["Text"]
    val_label = val_data["Veracity_int"]
    val_data['Text_length'] = val_data['Text'].apply(lambda x: len(x.split()))
    
    return train_txt,  val_txt, train_label, val_label
    

def find_levels(data, level=0, result=None):
    if result is None:
        result = list()
    if isinstance(data, dict):
        for key, value in data.items():
            result.append((key, level))
            find_levels(value, level + 1, result)
    elif isinstance(data, list):
        for item in data:
            find_levels(item, level, result)
    else:
        print(f"Level {level}: {type(data).__name__}")
    return result

def add_spec_tokens(src_post, replies):
    src_post = "[CLS] "+src_post+ " [SEP] "
    replies = [" [CLS] " + tokens + " [SEP] " for tokens in replies]
    return src_post, replies
# Example usage:
def merge_thread(src_post, replies, reply_ids, struct_data, d_type):
    levels = find_levels(struct_data)
    sorted_levels = sorted(levels, key=lambda x: x[1])
    bfs_fuse = src_post
    for key, level in sorted_levels:
        # Use the key directly as a string, no conversion needed
        if key in reply_ids:
            # Access the reply using the index of the key directly
            bfs_fuse += replies[reply_ids.index(key)]
    return bfs_fuse
        
def read_twitr_src_twt(directory, labels, data_mode):
    if data_mode == "src_twt_only":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        count = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            if os.path.basename(dirpath) == 'source-tweet':
                for file in filenames:
                    if file == "structure.json" or file == "dev-key.json" or file == "train-key.json":
                        continue
                    if file.endswith(".json"):
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, 'r') as json_file:
                            try:
                                data = json.load(json_file)
                                
                                if str(data['id']) in labels['subtaskbenglish'].keys():
                                    df.loc[len(df.index)] = [data['text'], labels['subtaskbenglish'][str(data['id'])]]
                                    count = count+1                        
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file {file_path}: {e}")
        zero_numbering = {'true':0, 'false':1, 'unverified':2}
        df['Veracity_int'] = df['Veracity'].apply(lambda x: zero_numbering[x])
        df['Veracity_int'].unique()
        return df
    
    elif data_mode == "src_twt_all_replies":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        p_count   = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            break
        for subdir in dirnames:
            path =  os.path.join(dirpath, subdir)
            p_count = p_count+1
            r_label = None
            replies = list()
            reply_ids     = list()
            for dirpat, dirs, files in os.walk(path):
                r_count = 0
                if "structure.json" in files:
                    with open( os.path.join(dirpat, "structure.json"), 'r') as json_file:
                        struct_data = json.load(json_file)
                if os.path.basename(dirpat) == 'source-tweet':
                    for file in files:
                        if  file == "dev-key.json" or file == "train-key.json":
                            continue

                        if file.endswith(".json"):
                            file_path = os.path.join(dirpat, file)
                            with open(file_path, 'r') as json_file:
                                try:
                                    data = json.load(json_file)
                                    src_post = data['text']
                                    if str(data['id']) in labels['subtaskbenglish'].keys():
                                        r_label = labels['subtaskbenglish'][str(data['id'])]
                                    else:
                                        continue

                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")             
                elif os.path.basename(dirpat) == 'replies':
                    for file in files:
                        if file.endswith(".json"):
                            file_path = os.path.join(dirpat, file)
                            with open(file_path, 'r') as json_file:
                                try:
                                    data = json.load(json_file)
                                    replies.append(data['text'])
                                    reply_ids.append(data['id'])
                                    r_count = r_count+1
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")
            if r_label == None:
                continue
            '''
            print("Source post: ", src_post)
            print()
            print("Replies: ", replies)
            '''
            src_post, replies = add_spec_tokens(src_post, replies)
            '''
            print()
            print()
            
            print("Source post: ", src_post)
            print()
            print("Replies: ", replies)
            '''
            merged_thread_bfs = merge_thread(src_post, replies, reply_ids, struct_data, d_type="twitter")
            df.loc[len(df.index)] = [merged_thread_bfs, r_label]
            '''
            print()
            print("Merged: ", merged_thread_bfs)
            '''
          
        zero_numbering = {'true':0, 'false':1, 'unverified':2}
        df['Veracity_int'] = df['Veracity'].apply(lambda x: zero_numbering[x])
        df['Veracity_int'].unique()
        return df

def read_reddit_src_twt(directory, labels, data_mode):
    if data_mode == "src_twt_only":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        count = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            if os.path.basename(dirpath) == 'source-tweet':
                for file in filenames:
                    if file == "structure.json" or file == "dev-key.json" or file == "train-key.json" or file == "raw.json":
                        continue
                    if file.endswith(".json"):
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, 'r') as json_file:
                            try:
                                data = json.load(json_file)
                                if str(data['data']['children'][0]['data']['id']) in labels['subtaskaenglish'].keys():
                                    df.loc[len(df.index)] = [data['data']['children'][0]['data']['title'], labels['subtaskbenglish'][str(data['data']['children'][0]['data']['id'])]]
                                    count = count+1 
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file {file_path}: {e}")
                                
        zero_numbering = {'true':0, 'false':1, 'unverified':2}
        df['Veracity_int'] = df['Veracity'].apply(lambda x: zero_numbering[x])
        df['Veracity_int'].unique()
        return df
    
    elif data_mode == "src_twt_all_replies":
        df = pd.DataFrame(columns=['Text', 'Veracity'])
        p_count   = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            break
        for subdir in dirnames:
            path =  os.path.join(dirpath, subdir)
            p_count = p_count+1
            r_label = None
            r_count = 0
            replies = list()
            reply_ids     = list()
            for dirpat, dirs, files in os.walk(path):
                if "structure.json" in files:
                    with open( os.path.join(dirpat, "structure.json"), 'r') as json_file:
                        struct_data = json.load(json_file)
                if os.path.basename(dirpat) == 'source-tweet':
                    for file in files:
                        if  file == "dev-key.json" or file == "train-key.json" or file == "raw.json":
                            continue
                        if file.endswith(".json"):
                            file_path = os.path.join(dirpat, file)
                            with open(file_path, 'r') as json_file:
                                try:
                                    data = json.load(json_file)
                                    src_post = data['data']['children'][0]['data']['title']
                                    if str(data['data']['children'][0]['data']['id']) in labels['subtaskaenglish'].keys():
                                        r_label = labels['subtaskbenglish'][str(data['data']['children'][0]['data']['id'])]
                                    else:
                                        continue
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")
    
                elif os.path.basename(dirpat) == 'replies':
                    for file in files:
                        if file.endswith(".json"):
                            file_path = os.path.join(dirpat, file)
                            with open(file_path, 'r') as json_file:
                                try:
                                    data = json.load(json_file)
                                    try:
                                        replies.append(data['data']['body'])
                                        reply_ids.append(data['data']['id'])
                                        r_count = r_count+1
                                    except KeyError:
                                        continue
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")
            if r_label == None:
                continue
            merged_thread_bfs = merge_thread(src_post, replies, reply_ids, struct_data, d_type="reddit")
            df.loc[len(df.index)] = [merged_thread_bfs, r_label]
            #break
        zero_numbering = {'true':0, 'false':1, 'unverified':2}
        df['Veracity_int'] = df['Veracity'].apply(lambda x: zero_numbering[x])
        df['Veracity_int'].unique()
        return df

# Read tweet train data
def read_src_twts(flag, data_mode):
    if flag == "train":
        if data_mode == "src_twt_only":
            
            #Read tweet train data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english'
            label_path = './Rumoureval2019/rumoureval-2019-training-data/train-key.json'
            with open(label_path, 'r') as file:
                labels = json.load(file)
            twt_data_train = read_twitr_src_twt(directory_path, labels, data_mode)
            
            # Read reddit train data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/reddit-training-data'
            redt_data_train = read_reddit_src_twt(directory_path, labels, data_mode)
            
            # Read reddit dev data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/reddit-dev-data'
            label_path = './Rumoureval2019/rumoureval-2019-training-data/dev-key.json'
            with open(label_path, 'r') as file:
                labels = json.load(file)
            redt_data_dev = read_reddit_src_twt(directory_path, labels, data_mode)
            # Read tweet val data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english'
            twt_data_dev = read_twitr_src_twt(directory_path, labels, data_mode)
            
            train_txt,  val_txt, train_label, val_label = concatenate_samples(twt_data_train, redt_data_train, 
                                                                              twt_data_dev, redt_data_dev)
            return train_txt,  val_txt, train_label, val_label
        
        elif data_mode == "src_twt_all_replies":
            twt_bfs_threads_train = pd.DataFrame()
            twt_bfs_threads_val = pd.DataFrame()
            topics = ['charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash', 'illary', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege']
            
            label_path = './Rumoureval2019/rumoureval-2019-training-data/train-key.json'
            with open(label_path, 'r') as file:
                labels_train = json.load(file)
                
            label_path = './Rumoureval2019/rumoureval-2019-training-data/dev-key.json'
            with open(label_path, 'r') as file:
                labels_val = json.load(file)
    
            #Read Twitter train data
            for topic in topics:
                directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english/'+topic
                topic_threads = read_twitr_src_twt(directory_path, labels_train, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_bfs_threads_train = pd.concat([twt_bfs_threads_train, topic_threads], ignore_index=True)
                
            
            
            #Read Twitter validation
            for topic in topics:
                directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english/'+topic
                topic_threads = read_twitr_src_twt(directory_path, labels_val, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_bfs_threads_val = pd.concat([twt_bfs_threads_val, topic_threads], ignore_index=True)
            
            #Read Reddit train data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/reddit-training-data'
            redt_bfs_threads_train = read_reddit_src_twt(directory_path, labels_train, data_mode)
                
            #Reddit val data
            directory_path = '/home/rk010/DM/NewDataset/rumoureval-2019-training-data/reddit-dev-data'
            redt_bfs_threads_val  = read_reddit_src_twt(directory_path, labels_val, data_mode)
            
            train_txt,  val_txt, train_label, val_label = concatenate_samples(twt_bfs_threads_train, redt_bfs_threads_train, 
                                                                              twt_bfs_threads_val, redt_bfs_threads_val)
            #return train_txt,  val_txt, train_label, val_label, twt_bfs_threads_train
            
            return train_txt,  val_txt, train_label, val_label
            
            
    else:
        #Read twitter test data
        twt_dir_path = './Rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data'
        redt_dir_path = './Rumoureval2019/rumoureval-2019-test-data/reddit-test-data'
        label_path = './Rumoureval2019/final-eval-key.json'
        with open(label_path, 'r') as file:
            labels_test = json.load(file)
        if data_mode == "src_twt_only":
            twt_data_test = read_twitr_src_twt(twt_dir_path, labels_test, data_mode)
            # Read reddit dev data
            redt_data_test = read_reddit_src_twt(redt_dir_path, labels_test, data_mode)
            
        elif data_mode == "src_twt_all_replies":
            #read twitter test data
            for dirpath, dirnames, filenames in os.walk(twt_dir_path):
                #print("Dir names: \n", dirnames)
                break
            twt_data_test = pd.DataFrame()
            count = 0
            for topic in dirnames:
                count += 1
                directory_path = './Rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data/'+topic
                topic_threads = read_twitr_src_twt(directory_path, labels_test, data_mode)
                #result_df = pd.concat([result_df, df], ignore_index=True)
                twt_data_test = pd.concat([twt_data_test, topic_threads], ignore_index=True)
            
            #read reddit test data
            redt_data_test  = read_reddit_src_twt(redt_dir_path, labels_test, data_mode)
                        
        test_data    = pd.concat([redt_data_test, twt_data_test], ignore_index=True)
        
        test_data  = test_data.dropna()#drop null entries
        test_data['Text_length'] = test_data['Text'].apply(lambda x: len(x.split()))
        
        test_data  = test_data.sample(frac = 1)#shuffle tuples
        
        test_txt   = test_data["Text"]
        test_label = test_data["Veracity_int"]
        #return test_data, test_txt, test_label
        return test_txt, test_label
            
#test_txt, test_label = read_src_twts(flag="test", data_mode="src_twt_all_replies")
read_src_twts(flag="train", data_mode="src_twt_all_replies")
#Include break statement somewhere

