import os
import json
import argparse
import datetime
from collections import defaultdict
import random

def create_recformer_metadata(news_file, output_file, main_categories_filter=None):
    """Create metadata.json file for RecFormer from MIND news.tsv file."""
    metadata = {}
    
    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:  # Need at least item_id, main_category, subcategory, title
                continue
            
            item_id = arr[0]
            main_category = arr[1]
            subcategory = arr[2] if len(arr) > 2 else arr[1]  # Use subcategory as category
            title = arr[3].replace('\t', ' ').replace('\n', ' ').strip()
            
            # Filter by main categories if specified
            if main_categories_filter and main_category not in main_categories_filter:
                continue
            
            # Clean title and subcategory from problematic characters
            title = title.replace('"', '').strip()
            subcategory = subcategory.replace('"', '').strip()
            
            # Create metadata entry (using subcategory as category, no brand info available)
            metadata[item_id] = {
                "title": title,
                "category": subcategory
            }
    
    # Write metadata to JSON file
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Created metadata file with {len(metadata)} items: {output_file}")
    return metadata

def create_user_sequences(behaviors_file, metadata, min_seq_length=5):
    """Create user interaction sequences from MIND behaviors.tsv file."""
    user_sequences = defaultdict(list)
    
    with open(behaviors_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 5:  # Need impression_id, user_id, time, history, impressions
                continue
            
            user_id = arr[1]
            timestamp_str = arr[2].strip()
            
            # Parse timestamp
            try:
                timestamp_dt = datetime.datetime.strptime(timestamp_str, '%m/%d/%Y %I:%M:%S %p')
                timestamp = timestamp_dt.timestamp()
            except ValueError:
                timestamp = 0.0
            
            # Process impressions (arr[4] contains the clicked/not-clicked items)
            impressions = arr[4].strip().split()
            
            clicked_items = []
            for item in impressions:
                if '-' in item and item.endswith('-1'):  # Only clicked items
                    item_id = item.split('-')[0]
                    # Only include items that are in our metadata (filtered by main category)
                    if item_id in metadata:
                        clicked_items.append((item_id, timestamp))
            
            # Add clicked items to user sequence
            for item_id, ts in clicked_items:
                user_sequences[user_id].append((item_id, ts))
    
    # Sort each user's sequence by timestamp and remove duplicates while preserving order
    for user_id in user_sequences:
        # Sort by timestamp
        user_sequences[user_id].sort(key=lambda x: x[1])
        # Remove consecutive duplicates while preserving order
        seen = set()
        unique_sequence = []
        for item_id, ts in user_sequences[user_id]:
            if item_id not in seen:
                unique_sequence.append(item_id)
                seen.add(item_id)
        user_sequences[user_id] = unique_sequence
    
    # Filter out sequences that are too short
    filtered_sequences = {user_id: seq for user_id, seq in user_sequences.items() 
                         if len(seq) >= min_seq_length}
    
    print(f"Created {len(filtered_sequences)} user sequences (min length: {min_seq_length})")
    return filtered_sequences

def split_sequences_train_dev(user_sequences, dev_ratio=0.1, max_seq_length=50):
    """Split user sequences into train and dev sets."""
    train_sequences = []
    dev_sequences = []
    
    users = list(user_sequences.keys())
    random.shuffle(users)
    dev_size = int(len(users) * dev_ratio)
    dev_users = set(users[:dev_size])
    
    for user_id, sequence in user_sequences.items():
        # Truncate sequence if too long
        if len(sequence) > max_seq_length:
            sequence = sequence[-max_seq_length:]
        
        if user_id in dev_users:
            dev_sequences.append(sequence)
        else:
            train_sequences.append(sequence)
    
    print(f"Split into {len(train_sequences)} train sequences and {len(dev_sequences)} dev sequences")
    return train_sequences, dev_sequences

def create_recformer_files(news_file, behaviors_file, output_dir, main_categories_filter=None, 
                          dev_ratio=0.1, min_seq_length=5, max_seq_length=50):
    """Create all RecFormer files from MIND dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata file
    metadata_file = os.path.join(output_dir, "meta_data.json")
    metadata = create_recformer_metadata(news_file, metadata_file, main_categories_filter)
    
    # Create user sequences
    user_sequences = create_user_sequences(behaviors_file, metadata, min_seq_length)
    
    # Split into train and dev
    train_sequences, dev_sequences = split_sequences_train_dev(user_sequences, dev_ratio, max_seq_length)
    
    # Write train.json
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, "w", encoding='utf-8') as f:
        json.dump(train_sequences, f, ensure_ascii=False)
    print(f"Created train file: {train_file}")
    
    # Write dev.json
    dev_file = os.path.join(output_dir, "dev.json")
    with open(dev_file, "w", encoding='utf-8') as f:
        json.dump(dev_sequences, f, ensure_ascii=False)
    print(f"Created dev file: {dev_file}")
    
    return metadata, train_sequences, dev_sequences

def analyze_mind_categories(news_file):
    """Analyze and print category distribution in the MIND dataset."""
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
        if len(subcats) <= 10:  # Show subcategories if not too many
            print(f"    Subcategories: {', '.join(sorted(subcats))}")
    
    print(f"\nTotal: {len(main_category_counts)} main categories, {len(subcategory_counts)} subcategories")
    return list(main_category_counts.keys()), list(subcategory_counts.keys())

def main():
    parser = argparse.ArgumentParser(description="Convert MIND dataset to RecFormer format")
    parser.add_argument("--input_dir", required=True, help="Input directory containing news.tsv and behaviors.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory for RecFormer JSON files")
    parser.add_argument("--main_categories", nargs='+', 
                       help="Main categories to include (will use subcategories as category values)")
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                       help="Ratio of users to use for dev set (default: 0.1)")
    parser.add_argument("--min_seq_length", type=int, default=5,
                       help="Minimum sequence length to include (default: 5)")
    parser.add_argument("--max_seq_length", type=int, default=50,
                       help="Maximum sequence length (will truncate longer sequences, default: 50)")
    parser.add_argument("--analyze_only", action='store_true',
                       help="Only analyze categories without creating files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    news_file = os.path.join(args.input_dir, "news.tsv")
    behaviors_file = os.path.join(args.input_dir, "behaviors.tsv")
    
    # Check if input files exist
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
    
    print(f"\nConverting MIND dataset to RecFormer format...")
    if args.main_categories:
        print(f"Filtering by main categories: {args.main_categories}")
        invalid_categories = [cat for cat in args.main_categories if cat not in main_categories]
        if invalid_categories:
            print(f"Warning: Invalid main categories: {invalid_categories}")
            print(f"Available main categories: {main_categories}")
    
    metadata, train_sequences, dev_sequences = create_recformer_files(
        news_file, behaviors_file, args.output_dir, 
        main_categories_filter=args.main_categories,
        dev_ratio=args.dev_ratio,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length
    )
    
    print(f"\nConversion completed!")
    print(f"Files created in: {args.output_dir}")
    print(f"- meta_data.json: {len(metadata)} items")
    print(f"- train.json: {len(train_sequences)} sequences")
    print(f"- dev.json: {len(dev_sequences)} sequences")

if __name__ == "__main__":
    main()