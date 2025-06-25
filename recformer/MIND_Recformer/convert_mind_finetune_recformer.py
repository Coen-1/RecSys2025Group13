import os
import json
import argparse
import datetime
from collections import defaultdict
import random

def analyze_mind_categories(news_file):
    main_category_counts = defaultdict(int)
    subcategory_counts = defaultdict(int)
    main_to_subcategories = defaultdict(set)

    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) >= 3:
                main_category = arr[1]
                subcategory = arr[2]
                main_category_counts[main_category] += 1
                subcategory_counts[subcategory] += 1
                main_to_subcategories[main_category].add(subcategory)

    print("Main category distribution:")
    for category, count in sorted(main_category_counts.items(), key=lambda x: x[1], reverse=True):
        subcats = main_to_subcategories[category]
        print(f"  {category}: {count} items, {len(subcats)} subcategories")
        if len(subcats) <= 10:
            print(f"    Subcategories: {', '.join(sorted(subcats))}")

    print(f"\nTotal: {len(main_category_counts)} main categories, {len(subcategory_counts)} subcategories")
    return list(main_category_counts.keys()), list(subcategory_counts.keys())

def create_category_data(news_file, behaviors_file, main_category, min_seq_length=5):
    category_items = {}
    item_to_subcategory = {}

    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue

            item_id = arr[0]
            item_main_category = arr[1]
            subcategory = arr[2] if len(arr) > 2 else arr[1]
            title = arr[3].replace('\t', ' ').replace('\n', ' ').replace('"', '').strip()

            if item_main_category == main_category:
                category_items[item_id] = {
                    "title": title,
                    "category": subcategory
                }
                item_to_subcategory[item_id] = subcategory

    print(f"Found {len(category_items)} items for category '{main_category}'")

    user_sequences = defaultdict(list)

    with open(behaviors_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 5:
                continue

            user_id = arr[1]
            timestamp_str = arr[2].strip()

            try:
                timestamp_dt = datetime.datetime.strptime(timestamp_str, '%m/%d/%Y %I:%M:%S %p')
                timestamp = timestamp_dt.timestamp()
            except ValueError:
                timestamp = 0.0

            impressions = arr[4].strip().split()

            clicked_items = []
            for item in impressions:
                if '-' in item and item.endswith('-1'):
                    item_id = item.split('-')[0]
                    if item_id in category_items:
                        clicked_items.append((item_id, timestamp))

            for item_id, ts in clicked_items:
                user_sequences[user_id].append((item_id, ts))

    processed_sequences = {}
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
        seen = set()
        unique_sequence = []
        for item_id, ts in user_sequences[user_id]:
            if item_id not in seen:
                unique_sequence.append(item_id)
                seen.add(item_id)

        if len(unique_sequence) >= min_seq_length:
            processed_sequences[user_id] = unique_sequence

    print(f"Found {len(processed_sequences)} users with sequences >= {min_seq_length} items")

    return category_items, processed_sequences

def create_user_splits(user_sequences, smap, min_context_length=3, max_seq_length=50, 
                      train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split user sequences into train/val/test following RecFormer format:
    - Train: can have multiple targets per user
    - Val/Test: single target per user
    - All val/test users must exist in train
    """
    random.seed(seed)
    
    # Convert sequences to numeric and filter
    valid_users = {}
    for user_id, sequence in user_sequences.items():
        if len(sequence) > max_seq_length:
            sequence = sequence[-max_seq_length:]
        
        numeric_sequence = [smap[item_id] for item_id in sequence if item_id in smap]
        
        # Need at least min_context_length + 2 items for train/val/test split
        if len(numeric_sequence) >= min_context_length + 2:
            valid_users[user_id] = numeric_sequence
    
    print(f"Valid users for splitting: {len(valid_users)}")
    
    if len(valid_users) == 0:
        raise ValueError("No users with sufficient sequence length")
    
    # Create consistent user mapping
    all_users = sorted(valid_users.keys())  # Sort for reproducibility
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    
    # Determine which users get val/test data
    random.shuffle(all_users)
    
    num_val_users = max(1, int(len(all_users) * val_ratio))
    num_test_users = max(1, int(len(all_users) * val_ratio))  # Same as val_ratio
    
    val_users = set(all_users[:num_val_users])
    test_users = set(all_users[num_val_users:num_val_users + num_test_users])
    
    print(f"Users: {len(all_users)} total, {len(val_users)} val, {len(test_users)} test")
    
    train_data = {}
    val_data = {}
    test_data = {}
    
    for user_id, sequence in valid_users.items():
        user_idx = str(user_to_idx[user_id])
        
        if len(sequence) < min_context_length + 1:
            continue
            
        # Split sequence: use last items for val/test, rest for training
        if user_id in test_users and len(sequence) >= min_context_length + 2:
            # User gets train + val + test
            train_seq = sequence[:-2]
            val_item = sequence[-2]
            test_item = sequence[-1]
            
            train_data[user_idx] = train_seq
            val_data[user_idx] = [val_item]  # Single target
            test_data[user_idx] = [test_item]  # Single target
            
        elif user_id in val_users:
            # User gets train + val
            train_seq = sequence[:-1]
            val_item = sequence[-1]
            
            train_data[user_idx] = train_seq
            val_data[user_idx] = [val_item]  # Single target
            
        else:
            # User gets only training data
            train_data[user_idx] = sequence
    
    # Ensure all val/test users exist in train
    train_users = set(train_data.keys())
    val_users_idx = set(val_data.keys())
    test_users_idx = set(test_data.keys())
    
    missing_val = val_users_idx - train_users
    missing_test = test_users_idx - train_users
    
    if missing_val:
        print(f"ERROR: {len(missing_val)} val users missing from train: {list(missing_val)[:5]}")
        raise ValueError("All val users must exist in train set")
    
    if missing_test:
        print(f"ERROR: {len(missing_test)} test users missing from train: {list(missing_test)[:5]}")
        raise ValueError("All test users must exist in train set")
    
    print(f"Final data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"Validation: All val users in train: {val_users_idx.issubset(train_users)}")
    print(f"Validation: All test users in train: {test_users_idx.issubset(train_users)}")
    
    return user_to_idx, train_data, val_data, test_data

def create_mappings_and_splits(category_items, user_sequences, max_seq_length=50, min_context_length=3):
    """
    Create proper numeric mappings following RecFormer format:
    - Train: multiple targets per user allowed
    - Val/Test: single target per user
    - All val/test users must exist in train
    """
    # Create item mapping
    all_items = list(category_items.keys())
    smap = {item: idx for idx, item in enumerate(all_items)}
    
    # Split user sequences properly
    user_to_idx, train_data, val_data, test_data = create_user_splits(
        user_sequences, smap, min_context_length, max_seq_length
    )
    
    # Create user mapping for compatibility
    umap = {f"U{idx}": idx for idx in range(len(user_to_idx))}
    
    return smap, umap, train_data, val_data, test_data

def create_finetune_files(news_file, behaviors_file, output_dir, finetune_categories,
                         min_seq_length=5, max_seq_length=50, min_context_length=3, seed=42):

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    for main_category in finetune_categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {main_category}")
        print(f"{'='*50}")

        category_dir = os.path.join(output_dir, f"finetune_{main_category}")
        os.makedirs(category_dir, exist_ok=True)

        category_items, user_sequences = create_category_data(
            news_file, behaviors_file, main_category, min_seq_length
        )

        if not category_items or not user_sequences:
            print(f"Warning: No data found for category '{main_category}', skipping...")
            continue

        try:
            smap, umap, train_seqs, val_seqs, test_seqs = create_mappings_and_splits(
                category_items, user_sequences, max_seq_length, min_context_length
            )
        except ValueError as e:
            print(f"Error processing category '{main_category}': {e}")
            continue

        # Create metadata with original news IDs as keys
        metadata = {}
        for original_id, numeric_id in smap.items():
            if original_id in category_items:
                metadata[original_id] = category_items[original_id]

        files_to_write = [
            ('train.json', train_seqs),
            ('val.json', val_seqs), 
            ('test.json', test_seqs),
            ('umap.json', umap),
            ('smap.json', smap),
            ('meta_data.json', metadata)
        ]

        for filename, data in files_to_write:
            filepath = os.path.join(category_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2 if filename in ['umap.json', 'smap.json', 'meta_data.json'] else None)
            print(f"Created: {filepath}")

        print(f"\nSummary for '{main_category}':")
        print(f"  - Items: {len(metadata)}")
        print(f"  - Users: {len(umap)}")
        print(f"  - Train sequences: {len(train_seqs)} users")
        print(f"  - Val sequences: {len(val_seqs)} users") 
        print(f"  - Test sequences: {len(test_seqs)} users")
        
        # Check sample data format
        if train_seqs:
            sample_user = list(train_seqs.keys())[0]
            sample_seq = train_seqs[sample_user]
            print(f"  - Sample train format: user {sample_user} -> {len(sample_seq)} items: {sample_seq[:5]}...")
        
        if val_seqs:
            sample_user = list(val_seqs.keys())[0]
            sample_seq = val_seqs[sample_user]
            print(f"  - Sample val format: user {sample_user} -> {len(sample_seq)} items: {sample_seq}")
        
        if test_seqs:
            sample_user = list(test_seqs.keys())[0]
            sample_seq = test_seqs[sample_user]
            print(f"  - Sample test format: user {sample_user} -> {len(sample_seq)} items: {sample_seq}")

def main():
    parser = argparse.ArgumentParser(description="Convert MIND dataset to RecFormer finetune format")
    parser.add_argument("--input_dir", required=True, help="Input directory containing news.tsv and behaviors.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory for finetune folders")
    parser.add_argument("--finetune_categories", nargs='+', required=True, help="Main categories to create finetune datasets for")
    parser.add_argument("--min_seq_length", type=int, default=5, help="Minimum sequence length to include (default: 5)")
    parser.add_argument("--max_seq_length", type=int, default=50, help="Maximum sequence length (default: 50)")
    parser.add_argument("--min_context_length", type=int, default=3, help="Minimum context length for training examples (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--analyze_only", action='store_true', help="Only analyze categories without creating files")

    args = parser.parse_args()

    news_file = os.path.join(args.input_dir, "news.tsv")
    behaviors_file = os.path.join(args.input_dir, "behaviors.tsv")

    if not os.path.exists(news_file):
        print(f"Error: {news_file} not found")
        return
    if not os.path.exists(behaviors_file):
        print(f"Error: {behaviors_file} not found")
        return

    print("Analyzing MIND dataset categories...")
    main_categories, subcategories = analyze_mind_categories(news_file)

    if args.analyze_only:
        return

    invalid_categories = [cat for cat in args.finetune_categories if cat not in main_categories]
    if invalid_categories:
        print(f"Error: Invalid finetune categories: {invalid_categories}")
        print(f"Available main categories: {main_categories}")
        return

    print(f"\nCreating finetune datasets for categories: {args.finetune_categories}")

    create_finetune_files(
        news_file, behaviors_file, args.output_dir, args.finetune_categories,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        min_context_length=args.min_context_length,
        seed=args.seed
    )

    print(f"\n{'='*50}")
    print("Conversion completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()