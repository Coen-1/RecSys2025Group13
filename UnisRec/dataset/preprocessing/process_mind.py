# process_mind.py
# V5: The definitive and complete version with all fixes.

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- Configuration ---
PRETRAIN_CATEGORIES = {'autos', 'health', 'finance', 'foodanddrink', 'lifestyle', 'travel', 'video', 'weather'}
FINETUNE_CATEGORIES = {'tv', 'music'}
PRETRAIN_DOMAIN_NAME = 'MIND-pretrain'
FINETUNE_DOMAIN_NAME = 'MIND-finetune'
RAW_DATA_PATH = '../raw/MIND-small/'
PRETRAIN_OUTPUT_PATH = f'../pretrain/{PRETRAIN_DOMAIN_NAME}/'
FINETUNE_OUTPUT_PATH = f'../downstream/{FINETUNE_DOMAIN_NAME}/'

os.makedirs(PRETRAIN_OUTPUT_PATH, exist_ok=True)
os.makedirs(FINETUNE_OUTPUT_PATH, exist_ok=True)

# --- Part 1: Unify Item Catalog ---
print("Step 1: Unifying the item catalog...")
train_news_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train', 'news.tsv'), sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
valid_news_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'dev', 'news.tsv'), sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
all_news_df = pd.concat([train_news_df, valid_news_df]).drop_duplicates(subset=['news_id']).reset_index(drop=True)
master_item_map = {news_id: i for i, news_id in enumerate(all_news_df['news_id'])}
train_behaviors_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train', 'behaviors.tsv'), sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
unique_users_train = train_behaviors_df['user_id'].unique()
master_user_map = {user_id: i for i, user_id in enumerate(unique_users_train)}
all_news_df['item_id_int'] = all_news_df['news_id'].map(master_item_map)
item_id_to_category = pd.Series(all_news_df.category.str.lower().values, index=all_news_df.item_id_int).to_dict()
all_news_df['text'] = all_news_df['title'].fillna('') + '. ' + all_news_df['category'].fillna('') + '. ' + all_news_df['abstract'].fillna('')

# --- Part 2: Generate Text Embeddings (Corrected) ---
print("Step 2: Generating text embeddings...")
master_feat_path = os.path.join('../pretrain/', 'MIND-small.feat1CLS') # Use raw binary extension
if not os.path.exists(master_feat_path):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased').to('cuda')
    all_item_embeddings = []
    for text in tqdm(all_news_df['text'], desc="Embedding All News"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
        with torch.no_grad():
            output = model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_item_embeddings.append(cls_embedding)
    final_embeddings = np.vstack(all_item_embeddings).astype(np.float32)
    final_embeddings.tofile(master_feat_path) # Use .tofile() for raw binary
else:
    print("Master .feat1CLS file already exists. Skipping.")

# --- Part 3: Helper function ---
def process_behaviors_to_sequences(df, user_map, item_map, item_id_to_category, allowed_categories):
    all_sequences = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Filtering for categories"):
        user_int_id = user_map.get(row['user_id'])
        if user_int_id is None: continue
        history_ids = []
        if isinstance(row['history'], str):
            history_ids = [item_map.get(nid) for nid in row['history'].split() if item_map.get(nid) is not None and item_id_to_category.get(item_map.get(nid)) in allowed_categories]
        if isinstance(row['impressions'], str):
            clicked_impressions = [imp for imp in row['impressions'].split() if imp.endswith('-1')]
            for impression in clicked_impressions:
                news_id = impression[:-2]
                item_int_id = item_map.get(news_id)
                if item_int_id is not None and item_id_to_category.get(item_int_id) in allowed_categories:
                    if history_ids:
                        all_sequences.append({'user_id': user_int_id, 'item_id_list': ' '.join(map(str, history_ids)), 'item_id': item_int_id})
                    history_ids.append(item_int_id)
    return pd.DataFrame(all_sequences)

# --- Part 4: Process and save datasets with correct headers ---
valid_behaviors_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'dev', 'behaviors.tsv'), sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
CORRECT_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"

print(f"\n----- Creating dataset for Pre-training: {PRETRAIN_DOMAIN_NAME} -----")
pretrain_train_seq = process_behaviors_to_sequences(train_behaviors_df, master_user_map, master_item_map, item_id_to_category, PRETRAIN_CATEGORIES)
pretrain_valid_seq = process_behaviors_to_sequences(valid_behaviors_df, master_user_map, master_item_map, item_id_to_category, PRETRAIN_CATEGORIES)
pretrain_train_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.train.inter')
with open(pretrain_train_path, 'w') as f: f.write(CORRECT_HEADER)
pretrain_train_seq.to_csv(pretrain_train_path, sep='\t', index=False, header=False, mode='a')
pretrain_valid_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.valid.inter')
with open(pretrain_valid_path, 'w') as f: f.write(CORRECT_HEADER)
pretrain_valid_seq.to_csv(pretrain_valid_path, sep='\t', index=False, header=False, mode='a')

print(f"\n----- Creating dataset for Fine-tuning: {FINETUNE_DOMAIN_NAME} -----")
finetune_train_seq = process_behaviors_to_sequences(train_behaviors_df, master_user_map, master_item_map, item_id_to_category, FINETUNE_CATEGORIES)
finetune_valid_seq = process_behaviors_to_sequences(valid_behaviors_df, master_user_map, master_item_map, item_id_to_category, FINETUNE_CATEGORIES)
finetune_train_path = os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.train.inter')
with open(finetune_train_path, 'w') as f: f.write(CORRECT_HEADER)
finetune_train_seq.to_csv(finetune_train_path, sep='\t', index=False, header=False, mode='a')
finetune_valid_path = os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.valid.inter')
with open(finetune_valid_path, 'w') as f: f.write(CORRECT_HEADER)
finetune_valid_seq.to_csv(finetune_valid_path, sep='\t', index=False, header=False, mode='a')

# --- Part 5: Save shared files and create symbolic links ---
print("\n----- Saving shared files and creating symbolic links -----")
master_user_map_df = pd.DataFrame(master_user_map.items(), columns=['original_id', 'new_id'])
master_item_map_df = pd.DataFrame(master_item_map.items(), columns=['original_id', 'new_id'])
master_text_df = all_news_df[['item_id_int', 'text']]

master_user_map_df.to_csv(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.user2index'), sep='\t', header=False, index=False)
master_item_map_df.to_csv(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.item2index'), sep='\t', header=False, index=False)
master_text_df.to_csv(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.text'), sep='\t', header=False, index=False)
os.symlink(os.path.relpath(master_feat_path, start=PRETRAIN_OUTPUT_PATH), os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.feat1CLS'))

os.symlink(f'../../pretrain/{PRETRAIN_DOMAIN_NAME}/{PRETRAIN_DOMAIN_NAME}.user2index', os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.user2index'))
os.symlink(f'../../pretrain/{PRETRAIN_DOMAIN_NAME}/{PRETRAIN_DOMAIN_NAME}.item2index', os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.item2index'))
os.symlink(f'../../pretrain/{PRETRAIN_DOMAIN_NAME}/{PRETRAIN_DOMAIN_NAME}.text', os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.text'))
os.symlink(f'../../pretrain/MIND-small.feat1CLS', os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.feat1CLS')) # Corrected symlink path

with open(os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.test.inter'), 'w') as f: f.write(CORRECT_HEADER)
with open(os.path.join(FINETUNE_OUTPUT_PATH, f'{FINETUNE_DOMAIN_NAME}.test.inter'), 'w') as f: f.write(CORRECT_HEADER)

# --- Part 6: Create the required .pt_datasets file ---
print("\n----- Creating .pt_datasets file for pre-training -----")
pt_datasets_path = os.path.join(PRETRAIN_OUTPUT_PATH, f'{PRETRAIN_DOMAIN_NAME}.pt_datasets')
with open(pt_datasets_path, 'w') as f:
    f.write(f'{PRETRAIN_DOMAIN_NAME}\n')

print("\n\nAll preprocessing tasks are complete!")