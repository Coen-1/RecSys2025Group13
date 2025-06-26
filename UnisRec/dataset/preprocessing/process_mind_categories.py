# process_mind_categories.py
# Purpose: Preprocess MIND-small, splitting by category for pre-training and fine-tuning.
# CORRECTED VERSION 6: Extensive debugging prints for ID consistency.

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- Configuration ---
RAW_DATA_PATH = '../raw/MIND-small/'

PRETRAIN_DATASET_NAME = 'MIND-small-Pretrain'
FINETUNE_TV_DATASET_NAME = 'MIND-small-TV'
FINETUNE_MUSIC_DATASET_NAME = 'MIND-small-Music'

PRETRAIN_OUTPUT_PATH = f'../pretrain/{PRETRAIN_DATASET_NAME}/'
FINETUNE_TV_OUTPUT_PATH = f'../downstream/{FINETUNE_TV_DATASET_NAME}/'
FINETUNE_MUSIC_OUTPUT_PATH = f'../downstream/{FINETUNE_MUSIC_DATASET_NAME}/'

PRETRAIN_CATEGORIES = ['autos', 'health', 'finance', 'foodanddrink', 'lifestyle', 'travel', 'video', 'weather']
FINETUNE_TV_CATEGORIES = ['tv']
FINETUNE_MUSIC_CATEGORIES = ['music']

RECHOLE_INTER_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"

os.makedirs(PRETRAIN_OUTPUT_PATH, exist_ok=True)
os.makedirs(FINETUNE_TV_OUTPUT_PATH, exist_ok=True)
os.makedirs(FINETUNE_MUSIC_OUTPUT_PATH, exist_ok=True)

# --- Part 1: Load and Prepare the Full News Catalog ---
print("Step 1: Loading and preparing the full news catalog...")
train_news_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train', 'news.tsv'), sep='\t', header=None, names=['news_id_orig', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
valid_news_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'dev', 'news.tsv'), sep='\t', header=None, names=['news_id_orig', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
all_news_df = pd.concat([train_news_df, valid_news_df]).drop_duplicates(subset=['news_id_orig']).reset_index(drop=True)
all_news_df['category'] = all_news_df['category'].str.lower()
all_news_df['text'] = all_news_df['title'].fillna('') + '. ' + all_news_df['category'].fillna('') + '. ' + all_news_df['abstract'].fillna('')
print(f"Shape of all_news_df after loading and concatenating: {all_news_df.shape}")


# --- Part 2: Create Mappings and Embeddings for the *Entire* Catalog ---
print("Step 2: Creating unified item map and text embeddings for all items...")
global_item_map = {news_id_orig: i for i, news_id_orig in enumerate(all_news_df['news_id_orig'])}
num_total_unique_items = len(global_item_map)
print(f"Total unique items found (size of global_item_map): {num_total_unique_items}")

# Save .item2index and .text files
for ds_name, ds_path in [
    (PRETRAIN_DATASET_NAME, PRETRAIN_OUTPUT_PATH),
    (FINETUNE_TV_DATASET_NAME, FINETUNE_TV_OUTPUT_PATH),
    (FINETUNE_MUSIC_DATASET_NAME, FINETUNE_MUSIC_OUTPUT_PATH)
]:
    pd.DataFrame(global_item_map.items(), columns=['original_id', 'new_id']).to_csv(os.path.join(ds_path, f'{ds_name}.item2index'), sep='\t', header=False, index=False)
    all_news_df[['news_id_orig', 'text']].to_csv(os.path.join(ds_path, f'{ds_name}.text'), sep='\t', header=False, index=False)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to('cuda')
all_item_embeddings_list = []
ordered_news_df_for_embedding = all_news_df.copy()
ordered_news_df_for_embedding['temp_item_id_int'] = ordered_news_df_for_embedding['news_id_orig'].map(global_item_map)
ordered_news_df_for_embedding.sort_values('temp_item_id_int', inplace=True)
print(f"Shape of ordered_news_df_for_embedding: {ordered_news_df_for_embedding.shape}")
if not ordered_news_df_for_embedding['temp_item_id_int'].is_monotonic_increasing or not ordered_news_df_for_embedding['temp_item_id_int'].iloc[0] == 0:
    print("CRITICAL DEBUG: ordered_news_df_for_embedding is not sorted correctly or does not start from 0.")
if ordered_news_df_for_embedding.shape[0] != num_total_unique_items:
     print(f"CRITICAL DEBUG: Mismatch! ordered_news_df shape {ordered_news_df_for_embedding.shape[0]} vs num_total_unique_items {num_total_unique_items}")


for text in tqdm(ordered_news_df_for_embedding['text'], desc="Embedding All News"):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
    with torch.no_grad():
        output = embedding_model(**inputs)
    all_item_embeddings_list.append(output.last_hidden_state[:, 0, :].cpu().numpy())
global_final_embeddings = np.vstack(all_item_embeddings_list)

print(f"Final check before saving .feat1CLS files:")
print(f"  Number of unique items in global_item_map: {len(global_item_map)}")
print(f"  Shape of all_news_df used for global_item_map: {all_news_df.shape}")
print(f"  Shape of ordered_news_df_for_embedding: {ordered_news_df_for_embedding.shape}")
if not ordered_news_df_for_embedding.empty:
    print(f"  Max 'temp_item_id_int' in ordered_news_df_for_embedding: {ordered_news_df_for_embedding['temp_item_id_int'].max()}")
else:
    print("  ordered_news_df_for_embedding is empty!")
print(f"  Number of rows to be saved in .feat1CLS (shape[0] of global_final_embeddings): {global_final_embeddings.shape[0]}")
if global_final_embeddings.shape[0] != num_total_unique_items:
    print(f"CRITICAL ERROR: Mismatch in embedding rows ({global_final_embeddings.shape[0]}) and unique items ({num_total_unique_items}) for global_final_embeddings")
    # exit() # Optionally exit if critical error

# Save .feat1CLS files
for ds_name, ds_path in [
    (PRETRAIN_DATASET_NAME, PRETRAIN_OUTPUT_PATH),
    (FINETUNE_TV_DATASET_NAME, FINETUNE_TV_OUTPUT_PATH),
    (FINETUNE_MUSIC_DATASET_NAME, FINETUNE_MUSIC_OUTPUT_PATH)
]:
    output_feat_path = os.path.join(ds_path, f'{ds_name}.feat1CLS')
    global_final_embeddings.astype(np.float32).tofile(output_feat_path)
    print(f"Saved raw binary embeddings to: {output_feat_path}")

# Create .pt_datasets file for PRETRAIN
print(f"Creating .pt_datasets file for {PRETRAIN_DATASET_NAME}...")
pt_datasets_content = PRETRAIN_DATASET_NAME 
pt_file_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DATASET_NAME}.pt_datasets')
with open(pt_file_path, 'w') as f: f.write(pt_datasets_content + '\n')
print(f"Successfully created: {pt_file_path}")


# --- Part 3: Helper Function to Process Behaviors ---
def process_specific_behaviors(behaviors_df, news_catalog_df, target_categories, user_map_for_output, global_item_map_ref, dataset_name_for_file, output_base_path, is_train_set_for_usermap_creation=False, apply_item_id_prefix=False, prefix_id='0'):
    print(f"Processing behaviors for categories: {target_categories} for dataset: {dataset_name_for_file}...")
    num_items_in_global_map = len(global_item_map_ref)
    print(f"  Inside process_specific_behaviors for {dataset_name_for_file}: Number of items in global_item_map_ref: {num_items_in_global_map}")
    
    relevant_news_df = news_catalog_df[news_catalog_df['category'].isin(target_categories)]
    relevant_news_id_orig_set = set(relevant_news_df['news_id_orig'])
    print(f"  Found {len(relevant_news_id_orig_set)} relevant news articles for categories: {target_categories}")

    current_user_map_to_use = user_map_for_output
    if is_train_set_for_usermap_creation:
        temp_user_map = {}
        user_counter = 0
        for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc=f"Building User Map for {dataset_name_for_file}"):
            if row['user_id'] in temp_user_map:
                continue
            if isinstance(row['impressions'], str):
                for impression in row['impressions'].split():
                    if impression.endswith('-1'):
                        news_id_orig_clicked = impression[:-2]
                        if news_id_orig_clicked in relevant_news_id_orig_set:
                            temp_user_map[row['user_id']] = user_counter
                            user_counter += 1
                            break 
        current_user_map_to_use = temp_user_map
        pd.DataFrame(current_user_map_to_use.items(), columns=['original_id', 'new_id']).to_csv(os.path.join(output_base_path, f'{dataset_name_for_file}.user2index'), sep='\t', header=False, index=False)
        print(f"  Created user map for {dataset_name_for_file} with {len(current_user_map_to_use)} users.")

    all_sequences = []
    max_item_id_encountered_in_this_split = -1

    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc=f"Creating Sequences for {dataset_name_for_file}"):
        user_int_id = current_user_map_to_use.get(row['user_id'])
        if user_int_id is None:
            continue
            
        current_session_history_global_ids = []
        if isinstance(row['history'], str):
            for news_id_orig_hist in row['history'].split():
                if news_id_orig_hist in relevant_news_id_orig_set:
                    global_int_id = global_item_map_ref.get(news_id_orig_hist)
                    if global_int_id is not None:
                        if global_int_id >= num_items_in_global_map: # Debug check
                            print(f"WARNING: History item ID {global_int_id} (from {news_id_orig_hist}) is out of bounds for global_item_map_ref size {num_items_in_global_map}")
                            continue
                        current_session_history_global_ids.append(global_int_id)
                        max_item_id_encountered_in_this_split = max(max_item_id_encountered_in_this_split, global_int_id)
        
        if isinstance(row['impressions'], str):
            clicked_impressions = [imp for imp in row['impressions'].split() if imp.endswith('-1')]
            for impression in clicked_impressions:
                news_id_orig_target = impression[:-2]
                if news_id_orig_target in relevant_news_id_orig_set:
                    global_int_id_target = global_item_map_ref.get(news_id_orig_target)
                    if global_int_id_target is not None:
                        if global_int_id_target >= num_items_in_global_map: # Debug check
                             print(f"WARNING: Target item ID {global_int_id_target} (from {news_id_orig_target}) is out of bounds for global_item_map_ref size {num_items_in_global_map}")
                             continue
                        if current_session_history_global_ids:
                            if apply_item_id_prefix:
                                history_for_file = [f"{prefix_id}-{gid}" for gid in current_session_history_global_ids]
                                target_for_file = f"{prefix_id}-{global_int_id_target}"
                            else:
                                history_for_file = [str(gid) for gid in current_session_history_global_ids]
                                target_for_file = str(global_int_id_target)
                            
                            all_sequences.append({
                                'user_id': user_int_id,
                                'item_id_list': ' '.join(history_for_file),
                                'item_id': target_for_file
                            })
                        current_session_history_global_ids.append(global_int_id_target)
                        max_item_id_encountered_in_this_split = max(max_item_id_encountered_in_this_split, global_int_id_target)
    
    print(f"  For {dataset_name_for_file}, max unprefixed item ID written to .inter file: {max_item_id_encountered_in_this_split}")
    if num_items_in_global_map > 0 and max_item_id_encountered_in_this_split >= num_items_in_global_map:
         print(f"  CRITICAL ID MISMATCH for {dataset_name_for_file}: Max item ID {max_item_id_encountered_in_this_split} is >= embedding table size {num_items_in_global_map}")
    elif num_items_in_global_map == 0 and max_item_id_encountered_in_this_split != -1:
         print(f"  CRITICAL: global_item_map seems empty but items were processed for {dataset_name_for_file}")


    return pd.DataFrame(all_sequences)

# --- Part 4: Process and Save Datasets ---
train_behaviors_raw_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train', 'behaviors.tsv'), sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
valid_behaviors_raw_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'dev', 'behaviors.tsv'), sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])

# --- Pre-training Dataset ---
print("--- Generating Pre-training Dataset ---")
pretrain_train_sequences_df = process_specific_behaviors(train_behaviors_raw_df, all_news_df, PRETRAIN_CATEGORIES, {}, global_item_map, PRETRAIN_DATASET_NAME, PRETRAIN_OUTPUT_PATH, is_train_set_for_usermap_creation=True, apply_item_id_prefix=True, prefix_id='0')
output_file_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DATASET_NAME}.train.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not pretrain_train_sequences_df.empty: pretrain_train_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

pretrain_user_map = pd.read_csv(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DATASET_NAME}.user2index'), sep='\t', header=None, names=['original_id', 'new_id'], index_col=0)['new_id'].to_dict()
pretrain_valid_sequences_df = process_specific_behaviors(valid_behaviors_raw_df, all_news_df, PRETRAIN_CATEGORIES, pretrain_user_map, global_item_map, PRETRAIN_DATASET_NAME, PRETRAIN_OUTPUT_PATH, apply_item_id_prefix=True, prefix_id='0')
output_file_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DATASET_NAME}.valid.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not pretrain_valid_sequences_df.empty: pretrain_valid_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

with open(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DATASET_NAME}.test.inter'), 'w') as f: f.write(RECHOLE_INTER_HEADER)

# --- Fine-tuning TV Dataset ---
print("--- Generating Fine-tuning TV Dataset ---")
tv_train_sequences_df = process_specific_behaviors(train_behaviors_raw_df, all_news_df, FINETUNE_TV_CATEGORIES, {}, global_item_map, FINETUNE_TV_DATASET_NAME, FINETUNE_TV_OUTPUT_PATH, is_train_set_for_usermap_creation=True, apply_item_id_prefix=False)
output_file_path = os.path.join(FINETUNE_TV_OUTPUT_PATH, f'{FINETUNE_TV_DATASET_NAME}.train.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not tv_train_sequences_df.empty: tv_train_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

tv_user_map = pd.read_csv(os.path.join(FINETUNE_TV_OUTPUT_PATH, f'{FINETUNE_TV_DATASET_NAME}.user2index'), sep='\t', header=None, names=['original_id', 'new_id'], index_col=0)['new_id'].to_dict()
tv_valid_sequences_df = process_specific_behaviors(valid_behaviors_raw_df, all_news_df, FINETUNE_TV_CATEGORIES, tv_user_map, global_item_map, FINETUNE_TV_DATASET_NAME, FINETUNE_TV_OUTPUT_PATH, apply_item_id_prefix=False)
output_file_path = os.path.join(FINETUNE_TV_OUTPUT_PATH, f'{FINETUNE_TV_DATASET_NAME}.valid.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not tv_valid_sequences_df.empty: tv_valid_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

with open(os.path.join(FINETUNE_TV_OUTPUT_PATH, f'{FINETUNE_TV_DATASET_NAME}.test.inter'), 'w') as f: f.write(RECHOLE_INTER_HEADER)

# --- Fine-tuning Music Dataset ---
print("--- Generating Fine-tuning Music Dataset ---")
music_train_sequences_df = process_specific_behaviors(train_behaviors_raw_df, all_news_df, FINETUNE_MUSIC_CATEGORIES, {}, global_item_map, FINETUNE_MUSIC_DATASET_NAME, FINETUNE_MUSIC_OUTPUT_PATH, is_train_set_for_usermap_creation=True, apply_item_id_prefix=False)
output_file_path = os.path.join(FINETUNE_MUSIC_OUTPUT_PATH, f'{FINETUNE_MUSIC_DATASET_NAME}.train.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not music_train_sequences_df.empty: music_train_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

music_user_map = pd.read_csv(os.path.join(FINETUNE_MUSIC_OUTPUT_PATH, f'{FINETUNE_MUSIC_DATASET_NAME}.user2index'), sep='\t', header=None, names=['original_id', 'new_id'], index_col=0)['new_id'].to_dict()
music_valid_sequences_df = process_specific_behaviors(valid_behaviors_raw_df, all_news_df, FINETUNE_MUSIC_CATEGORIES, music_user_map, global_item_map, FINETUNE_MUSIC_DATASET_NAME, FINETUNE_MUSIC_OUTPUT_PATH, apply_item_id_prefix=False)
output_file_path = os.path.join(FINETUNE_MUSIC_OUTPUT_PATH, f'{FINETUNE_MUSIC_DATASET_NAME}.valid.inter')
with open(output_file_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
if not music_valid_sequences_df.empty: music_valid_sequences_df.to_csv(output_file_path, sep='\t', index=False, header=False, mode='a')

with open(os.path.join(FINETUNE_MUSIC_OUTPUT_PATH, f'{FINETUNE_MUSIC_DATASET_NAME}.test.inter'), 'w') as f: f.write(RECHOLE_INTER_HEADER)

print("\nPreprocessing for MIND-small with category splits is complete!")