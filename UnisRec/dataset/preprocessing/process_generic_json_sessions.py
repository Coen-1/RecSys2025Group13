# process_generic_json_sessions.py
# Purpose: Converts a generic JSON session dataset, using an item ID translation map.
# VERSION 3: Adds use of an item ID translation map.

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os
import argparse

RECHOLE_INTER_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"

def main(args):
    print(f"Starting conversion for dataset: {args.dataset_name}")
    print(f"Input JSON directory: {args.input_json_dir}")
    print(f"Output UniSRec directory: {args.output_unisrec_dir}")
    print(f"Apply item ID prefix ('{args.item_id_prefix}-'): {args.apply_prefix}")

    os.makedirs(args.output_unisrec_dir, exist_ok=True)

    # --- Load Item ID Translation Map (if provided) ---
    item_id_translation_map = {}
    if args.item_translation_map_filename:
        translation_map_filepath = os.path.join(args.input_json_dir, args.item_translation_map_filename)
        if os.path.exists(translation_map_filepath):
            with open(translation_map_filepath, 'r', encoding='utf-8') as f:
                item_id_translation_map = json.load(f)
            print(f"Loaded item ID translation map from: {translation_map_filepath} with {len(item_id_translation_map)} entries.")
        else:
            print(f"WARNING: Item ID translation map file not found: {translation_map_filepath}. Will attempt direct mapping.")

    # --- Part 1: Load Metadata and Create Item-Level Files ---
    print("Step 1: Loading item metadata (meta_data.json)...")
    metadata_filepath = os.path.join(args.input_json_dir, 'meta_data.json')
    
    if not os.path.exists(metadata_filepath):
        print(f"ERROR: Metadata file not found: {metadata_filepath}")
        return

    with open(metadata_filepath, 'r', encoding='utf-8') as f:
        item_metadata_raw = json.load(f) # Keys are original string IDs like "Nxxxxx"

    all_item_original_ids_from_meta = list(item_metadata_raw.keys())
    # This item_map_for_this_dataset maps ORIGINAL META IDs ("Nxxxxx") to new local integers (0, 1, 2...)
    item_map_for_this_dataset = {orig_id: i for i, orig_id in enumerate(all_item_original_ids_from_meta)}
    num_items_in_this_dataset = len(item_map_for_this_dataset)
    print(f"Found {num_items_in_this_dataset} unique items in meta_data.json.")

    pd.DataFrame(item_map_for_this_dataset.items(), columns=['original_id', 'new_id']).to_csv(
        os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.item2index'), sep='\t', header=False, index=False)
    print(f"Saved {args.dataset_name}.item2index")

    item_texts_for_embedding = []
    item_texts_for_file_df = [] 
    for orig_id_meta in all_item_original_ids_from_meta: 
        meta = item_metadata_raw[orig_id_meta]
        title = meta.get('title', '')
        category = meta.get('category', '') 
        text = f"{title}. {category}".strip()
        item_texts_for_embedding.append(text)
        # Use the new local integer ID for the .text file
        item_texts_for_file_df.append({'item_id_int': item_map_for_this_dataset[orig_id_meta], 'text': text}) 

    pd.DataFrame(item_texts_for_file_df).sort_values('item_id_int')[['item_id_int', 'text']].to_csv(
        os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.text'), sep='\t', header=False, index=False)
    print(f"Saved {args.dataset_name}.text")

    print("Step 2: Generating text embeddings...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to('cuda')
    
    embeddings_for_this_dataset_list = []
    for text in tqdm(item_texts_for_embedding, desc=f"Embedding items for {args.dataset_name}"): # Embed in order of item_map_for_this_dataset
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
        with torch.no_grad(): output = embedding_model(**inputs)
        embeddings_for_this_dataset_list.append(output.last_hidden_state[:, 0, :].cpu().numpy())
    
    if not embeddings_for_this_dataset_list:
        final_embeddings_for_this_dataset = np.array([]).reshape(0, 768)
    else:
        final_embeddings_for_this_dataset = np.vstack(embeddings_for_this_dataset_list)

    if final_embeddings_for_this_dataset.shape[0] != num_items_in_this_dataset:
        print(f"CRITICAL ERROR: Embedding rows != items in metadata for {args.dataset_name}")
    
    output_feat_path = os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.feat1CLS')
    final_embeddings_for_this_dataset.astype(np.float32).tofile(output_feat_path)
    print(f"Saved embeddings to: {output_feat_path} with shape {final_embeddings_for_this_dataset.shape}")

    if args.apply_prefix:
        pt_content = args.dataset_name 
        pt_path = os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.pt_datasets')
        with open(pt_path, 'w') as f: f.write(pt_content + '\n')
        print(f"Created {pt_path}")

    print("User map will be built from train.json. umap.json (if present) will be ignored for user ID generation.")
    user_map_for_processing = {}


    def create_sequences_from_json_sessions(interaction_json_filename, 
                                            # item_map_for_this_dataset maps META IDs ("N...") to local integers (0...)
                                            item_map_for_this_dataset_ref, 
                                            # item_id_translation_map_ref maps SESSION IDs ("8") to META IDs ("N...")
                                            item_id_translation_map_ref, 
                                            user_map_to_use, 
                                            is_training_data=True, 
                                            build_user_map_from_this_file=False):
        filepath = os.path.join(args.input_json_dir, interaction_json_filename)
        print(f"Processing interaction file: {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f: sessions_raw = json.load(f)
        output_sequences = []
        current_user_map = user_map_to_use.copy()

        if build_user_map_from_this_file:
            print(f"Building user map from {interaction_json_filename}...")
            user_id_counter = 0
            for session_with_user in sessions_raw:
                if not session_with_user: continue
                user_orig_id_str = str(session_with_user[0])
                if user_orig_id_str not in current_user_map:
                    current_user_map[user_orig_id_str] = user_id_counter
                    user_id_counter += 1
            print(f"Built user map with {len(current_user_map)} users.")
        
        processed_sessions_count = 0
        sequences_generated_count = 0

        for session_idx, session_with_user in enumerate(tqdm(sessions_raw, desc=f"Processing sessions in {interaction_json_filename}")):
            if not session_with_user or len(session_with_user) < 2: continue
            user_orig_id_str = str(session_with_user[0])
            item_ids_from_session_file = [str(item) for item in session_with_user[1:]] # e.g., ["8", "9", "0"]

            user_int_id = current_user_map.get(user_orig_id_str)
            if user_int_id is None: continue

            session_final_local_int_ids = [] # These will be the 0-based IDs for items in item_map_for_this_dataset
            unmapped_count = 0
            for item_id_in_session in item_ids_from_session_file:
                # Step A: Translate session ID (e.g., "8") to meta ID (e.g., "N12345")
                meta_id_for_item = item_id_translation_map_ref.get(item_id_in_session)
                if not meta_id_for_item and not item_id_translation_map_ref: # If no map, assume session ID is meta ID
                    meta_id_for_item = item_id_in_session 
                
                if meta_id_for_item:
                    # Step B: Translate meta ID (e.g., "N12345") to final local integer ID (e.g., 101)
                    final_local_int_id = item_map_for_this_dataset_ref.get(meta_id_for_item)
                    if final_local_int_id is not None:
                        session_final_local_int_ids.append(final_local_int_id)
                    else:
                        unmapped_count +=1
                else: # item_id_in_session not in translation_map
                    unmapped_count +=1
            
            if len(item_ids_from_session_file) > 0 and len(session_final_local_int_ids) < (1 if is_training_data else 2) :
                print(f"DEBUG (session {session_idx}, user {user_orig_id_str}): Original items in session: {item_ids_from_session_file}")
                print(f"DEBUG: Mapped local int IDs: {session_final_local_int_ids}. Not enough valid items. Unmapped: {unmapped_count}")
            
            if len(session_final_local_int_ids) < 2: continue # Need at least 1 history + 1 target
            processed_sessions_count += 1

            # Sequence generation logic (sliding window or leave-one-out)
            # Uses session_final_local_int_ids which are now correct indices for your .feat1CLS
            if is_training_data: 
                for i in range(1, len(session_final_local_int_ids)):
                    history_ids = session_final_local_int_ids[:i]
                    target_id = session_final_local_int_ids[i]
                    # Apply prefix only if specified (for pre-training datasets)
                    history_str = [f"{args.item_id_prefix}-{gid}" if args.apply_prefix else str(gid) for gid in history_ids]
                    target_str = f"{args.item_id_prefix}-{target_id}" if args.apply_prefix else str(target_id)
                    output_sequences.append({'user_id': user_int_id, 'item_id_list': ' '.join(history_str), 'item_id': target_str})
                    sequences_generated_count +=1
            else: # Validation or Test
                history_ids = session_final_local_int_ids[:-1]
                target_id = session_final_local_int_ids[-1]
                if history_ids:
                    history_str = [f"{args.item_id_prefix}-{gid}" if args.apply_prefix else str(gid) for gid in history_ids]
                    target_str = f"{args.item_id_prefix}-{target_id}" if args.apply_prefix else str(target_id)
                    output_sequences.append({'user_id': user_int_id, 'item_id_list': ' '.join(history_str), 'item_id': target_str})
                    sequences_generated_count += 1
        
        print(f"Finished {interaction_json_filename}: Raw sessions: {len(sessions_raw)}, Processed sessions: {processed_sessions_count}, Output sequences: {sequences_generated_count}")
        return pd.DataFrame(output_sequences), current_user_map 
    
    train_sequences_df, final_user_map = create_sequences_from_json_sessions(
        'train.json', item_map_for_this_dataset, item_id_translation_map, user_map_for_processing, True, True
    )
    output_inter_path = os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.train.inter')
    with open(output_inter_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
    if not train_sequences_df.empty: train_sequences_df.to_csv(output_inter_path, sep='\t', index=False, header=False, mode='a')
    print(f"Saved {args.dataset_name}.train.inter")

    pd.DataFrame(final_user_map.items(), columns=['original_id', 'new_id']).to_csv(
        os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.user2index'), sep='\t', header=False, index=False)
    print(f"Saved {args.dataset_name}.user2index with {len(final_user_map)} users.")

    dev_sequences_df, _ = create_sequences_from_json_sessions('val.json', item_map_for_this_dataset, item_id_translation_map, final_user_map, False)
    output_inter_path = os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.valid.inter')
    with open(output_inter_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
    if not dev_sequences_df.empty: dev_sequences_df.to_csv(output_inter_path, sep='\t', index=False, header=False, mode='a')
    print(f"Saved {args.dataset_name}.valid.inter")

    test_json_path = os.path.join(args.input_json_dir, 'test.json')
    if os.path.exists(test_json_path):
        test_sequences_df, _ = create_sequences_from_json_sessions('test.json', item_map_for_this_dataset, item_id_translation_map, final_user_map, False)
        output_inter_path = os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.test.inter')
        with open(output_inter_path, 'w') as f: f.write(RECHOLE_INTER_HEADER)
        if not test_sequences_df.empty: test_sequences_df.to_csv(output_inter_path, sep='\t', index=False, header=False, mode='a')
        print(f"Saved {args.dataset_name}.test.inter")
    else:
        with open(os.path.join(args.output_unisrec_dir, f'{args.dataset_name}.test.inter'), 'w') as f: f.write(RECHOLE_INTER_HEADER)
        print(f"Created empty {args.dataset_name}.test.inter as test.json was not found.")

    print(f"\nPreprocessing for {args.dataset_name} is complete! Files in {args.output_unisrec_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert generic JSON session dataset to UniSRec format.")
    parser.add_argument('--input_json_dir', type=str, required=True, help="Directory containing meta_data.json, train.json, val.json, [test.json].")
    parser.add_argument('--item_translation_map_filename', type=str, default=None, help="Optional: Filename of the JSON item ID translation map within input_json_dir (e.g., item_id_internal_to_original.json).")
    parser.add_argument('--output_unisrec_dir', type=str, required=True, help="Output directory for UniSRec files.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Base name for output files.")
    parser.add_argument('--apply_prefix', action='store_true', help="Apply item ID prefix for pre-training.")
    parser.add_argument('--item_id_prefix', type=str, default='0', help="Prefix if --apply_prefix is set.")
    
    cli_args = parser.parse_args()
    main(cli_args)