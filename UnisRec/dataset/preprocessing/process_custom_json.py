# process_custom_json.py
# Purpose: Preprocess the custom JSON dataset format for UniSRec.
# CORRECTED FOR YOUR FILE STRUCTURE AND NAMES

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- Configuration ---
# Corrected path to your raw data, relative to dataset/preprocessing/
RAW_DATA_PATH = '../raw/MyCustomDataset/' 

OUTPUT_DATASET_NAME = 'MyCustomPretrain' # Name for the processed dataset
OUTPUT_PATH = f'../pretrain/{OUTPUT_DATASET_NAME}/' 

RECHOLE_INTER_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
ITEM_ID_PREFIX_FOR_PRETRAIN = '0' 

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Part 1: Load Metadata and Create Item-Level Files ---
print("Step 1: Loading item metadata...")
# Corrected filename for metadata
METADATA_FILENAME = 'mind_data_recformer_small_meta_data.json'
TRAIN_INTERACTION_FILENAME = 'mind_data_recformer_small_train.json'
DEV_INTERACTION_FILENAME = 'mind_data_recformer_small_dev.json'

with open(os.path.join(RAW_DATA_PATH, METADATA_FILENAME), 'r') as f:
    item_metadata_raw = json.load(f)

all_item_original_ids = list(item_metadata_raw.keys())
global_item_map = {orig_id: i for i, orig_id in enumerate(all_item_original_ids)}
num_total_unique_items = len(global_item_map)
print(f"Found {num_total_unique_items} unique items from {METADATA_FILENAME}.")

pd.DataFrame(global_item_map.items(), columns=['original_id', 'new_id']).to_csv(
    os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.item2index'), sep='\t', header=False, index=False)
print(f"Saved {OUTPUT_DATASET_NAME}.item2index")

item_texts_for_embedding = []
item_texts_for_file = [] 
for orig_id in all_item_original_ids: 
    meta = item_metadata_raw[orig_id]
    title = meta.get('title', '')
    category = meta.get('category', '')
    text = f"{title}. {category}".strip()
    item_texts_for_embedding.append(text)
    item_texts_for_file.append({'news_id_orig': orig_id, 'text': text})

pd.DataFrame(item_texts_for_file)[['news_id_orig', 'text']].to_csv(
    os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.text'), sep='\t', header=False, index=False)
print(f"Saved {OUTPUT_DATASET_NAME}.text")

print("Step 2: Generating text embeddings...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to('cuda')
all_item_embeddings_list = []
for text in tqdm(item_texts_for_embedding, desc="Embedding Items"):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
    with torch.no_grad():
        output = embedding_model(**inputs)
    all_item_embeddings_list.append(output.last_hidden_state[:, 0, :].cpu().numpy())
global_final_embeddings = np.vstack(all_item_embeddings_list)

if global_final_embeddings.shape[0] != num_total_unique_items:
    print(f"CRITICAL ERROR: Embedding rows ({global_final_embeddings.shape[0]}) != unique items ({num_total_unique_items})")
    exit()

output_feat_path = os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.feat1CLS')
global_final_embeddings.astype(np.float32).tofile(output_feat_path)
print(f"Saved embeddings to: {output_feat_path} with shape {global_final_embeddings.shape}")

pt_datasets_content = OUTPUT_DATASET_NAME
pt_file_path = os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.pt_datasets')
with open(pt_file_path, 'w') as f: f.write(pt_datasets_content + '\n')
print(f"Created {pt_file_path}")

def process_interaction_file(interaction_filename, global_item_map_ref, user_id_offset=0, is_training_data=True):
    filepath = os.path.join(RAW_DATA_PATH, interaction_filename)
    print(f"Processing interaction file: {filepath}...")
    with open(filepath, 'r') as f:
        sessions_raw = json.load(f)

    all_sequences_for_file = []
    user_id_counter = user_id_offset
    original_user_ids_in_this_file = [] 

    for session_orig_ids in tqdm(sessions_raw, desc=f"Processing sessions in {interaction_filename}"):
        current_user_id = user_id_counter
        # For user2index, we use the assigned integer user ID as both original and new if we're just creating a simple map
        original_user_ids_in_this_file.append(str(current_user_id)) 

        session_global_int_ids = []
        for orig_item_id in session_orig_ids:
            global_int_id = global_item_map_ref.get(orig_item_id)
            if global_int_id is not None:
                session_global_int_ids.append(global_int_id)
        
        if len(session_global_int_ids) < 2:
            user_id_counter +=1
            continue

        if is_training_data:
            for i in range(1, len(session_global_int_ids)):
                history_global_ids = session_global_int_ids[:i]
                target_global_id = session_global_int_ids[i]
                history_for_file = [f"{ITEM_ID_PREFIX_FOR_PRETRAIN}-{gid}" for gid in history_global_ids]
                target_for_file = f"{ITEM_ID_PREFIX_FOR_PRETRAIN}-{target_global_id}"
                all_sequences_for_file.append({
                    'user_id': current_user_id,
                    'item_id_list': ' '.join(history_for_file),
                    'item_id': target_for_file
                })
        else: 
            history_global_ids = session_global_int_ids[:-1]
            target_global_id = session_global_int_ids[-1]
            if history_global_ids:
                history_for_file = [f"{ITEM_ID_PREFIX_FOR_PRETRAIN}-{gid}" for gid in history_global_ids]
                target_for_file = f"{ITEM_ID_PREFIX_FOR_PRETRAIN}-{target_global_id}"
                all_sequences_for_file.append({
                    'user_id': current_user_id,
                    'item_id_list': ' '.join(history_for_file),
                    'item_id': target_for_file
                })
        user_id_counter += 1
    
    return pd.DataFrame(all_sequences_for_file), list(set(original_user_ids_in_this_file)), user_id_counter

train_sequences_df, train_orig_user_ids, last_train_user_id = process_interaction_file(TRAIN_INTERACTION_FILENAME, global_item_map, 0, True)
output_file_path = os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.train.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not train_sequences_df.empty: train_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')
print(f"Saved {OUTPUT_DATASET_NAME}.train.inter")

# Create user2index based on users present in the training data
# The 'original_id' will be our generated integer IDs (0, 1, 2...)
# The 'new_id' will be the same, as RecBole just needs a map.
user_map_data = [{'original_id': str(i), 'new_id': i} for i in range(len(train_orig_user_ids))]
user_map_df_train = pd.DataFrame(user_map_data)
user_map_df_train.to_csv(os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.user2index'), sep='\t', header=False, index=False)
print(f"Saved {OUTPUT_DATASET_NAME}.user2index with {len(train_orig_user_ids)} users.")

dev_sequences_df, _, _ = process_interaction_file(DEV_INTERACTION_FILENAME, global_item_map, last_train_user_id + 1, False)
output_file_path = os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.valid.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not dev_sequences_df.empty: dev_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')
print(f"Saved {OUTPUT_DATASET_NAME}.valid.inter")

with open(os.path.join(OUTPUT_PATH, f'{OUTPUT_DATASET_NAME}.test.inter'), 'w') as f:
    f.write(RECHOLE_INTER_HEADER)
print(f"Created empty {OUTPUT_DATASET_NAME}.test.inter")

print(f"\nPreprocessing for {OUTPUT_DATASET_NAME} is complete!")
print(f"Processed files are located in: {OUTPUT_PATH}")