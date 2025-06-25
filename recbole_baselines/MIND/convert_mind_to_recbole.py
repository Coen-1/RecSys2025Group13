import os
import argparse
import datetime
from collections import defaultdict

def create_recbole_item_file(news_file, item_file, use_subcategory=True):
    """Create item file for RecBole from MIND news.tsv file."""
    with open(news_file, "r", encoding='utf-8') as f, \
            open(item_file, "w", encoding='utf-8') as f_out:

        f_out.write("item_id:token\ttitle:token_seq\tcategories:token_seq\n")
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:  # Need at least item_id, category, subcategory, title
                continue
            
            item_id = arr[0]
            # Use subcategory (column 2) instead of main category (column 1)
            category = arr[2] if use_subcategory and len(arr) > 2 else arr[1]
            title = arr[3].replace('\t', ' ').replace('\n', ' ').strip()
            
            # Clean title and category from any problematic characters
            title = title.replace('"', '').strip()
            category = category.replace('"', '').strip()
            
            f_out.write(f'{item_id}\t{title}\t{category}\n')

def create_recbole_inter_file(behaviors_file, interaction_file):
    """Create interaction file for RecBole from MIND behaviors.tsv file."""
    with open(behaviors_file, "r", encoding='utf-8') as f, \
            open(interaction_file, "w", encoding='utf-8') as f_out:

        f_out.write("user_id:token\titem_id:token\ttimestamp:float\n")

        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 5:  # Need impression_id, user_id, time, history, impressions
                continue
            
            # arr[0] = Impression_ID
            user_id = arr[1]  # Use actual User_ID from the data
            timestamp_str = arr[2].strip()
            
            # Parse timestamp correctly
            try:
                timestamp_dt = datetime.datetime.strptime(timestamp_str, '%m/%d/%Y %I:%M:%S %p')
                timestamp = timestamp_dt.timestamp()  # Keep as float for RecBole
            except ValueError:
                timestamp = 0.0
            
            # Process impressions (arr[4] contains the clicked/not-clicked items)
            impressions = arr[4].strip().split()
            
            for item in impressions:
                if '-' in item:
                    item_id = item.split('-')[0]
                    # Only include items that were actually clicked (ending with -1)
                    if item.endswith('-1'):
                        f_out.write(f'{user_id}\t{item_id}\t{timestamp}\n')

def split_by_main_categories_keep_subcategories(news_file, behaviors_file, output_dir, pretrain_categories, finetune_categories):
    """Split data by main categories but preserve subcategories in the output files."""
    
    # First, read all news items and map them to both main and subcategories
    item_to_main_category = {}
    item_to_subcategory = {}
    
    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            item_id = arr[0]
            main_category = arr[1]
            subcategory = arr[2] if len(arr) > 2 else arr[1]
            item_to_main_category[item_id] = main_category
            item_to_subcategory[item_id] = subcategory
    
    # Create category sets for easy lookup
    pretrain_set = set(pretrain_categories)
    finetune_set = set(finetune_categories)
    
    # Get the output directory name for naming files
    output_dir_name = os.path.basename(output_dir.rstrip('/'))
    
    # Create pretrain item file with proper naming (using subcategories in content)
    pretrain_item_file = os.path.join(output_dir, f"{output_dir_name}.item")
    
    with open(news_file, "r", encoding='utf-8') as f, \
         open(pretrain_item_file, "w", encoding='utf-8') as f_pretrain:
        
        # Write header for pretrain
        header = "item_id:token\ttitle:token_seq\tcategories:token_seq\n"
        f_pretrain.write(header)
        
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            
            item_id = arr[0]
            main_category = arr[1]
            subcategory = arr[2] if len(arr) > 2 else arr[1]  # Use subcategory for content
            title = arr[3].replace('\t', ' ').replace('\n', ' ').replace('"', '').strip()
            
            # Filter by main category but write subcategory to file
            if main_category in pretrain_set:
                output_line = f'{item_id}\t{title}\t{subcategory}\n'
                f_pretrain.write(output_line)
    
    # Create separate item files for each finetune main category (but group by main category)
    finetune_files = {}
    for main_category in finetune_categories:
        category_dir = os.path.join(output_dir, f"finetune_{main_category}")
        os.makedirs(category_dir, exist_ok=True)
        
        item_file = os.path.join(category_dir, f"finetune_{main_category}.item")
        finetune_files[main_category] = open(item_file, "w", encoding='utf-8')
        finetune_files[main_category].write("item_id:token\ttitle:token_seq\tcategories:token_seq\n")
    
    # Fill finetune item files (filter by main category, write subcategory)
    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            
            item_id = arr[0]
            main_category = arr[1]
            subcategory = arr[2] if len(arr) > 2 else arr[1]  # Use subcategory for content
            title = arr[3].replace('\t', ' ').replace('\n', ' ').replace('"', '').strip()
            
            # Filter by main category but write subcategory to file
            if main_category in finetune_set:
                output_line = f'{item_id}\t{title}\t{subcategory}\n'
                finetune_files[main_category].write(output_line)
    
    # Close finetune item files
    for f in finetune_files.values():
        f.close()
    
    # Create pretrain interaction file with proper naming
    pretrain_inter_file = os.path.join(output_dir, f"{output_dir_name}.inter")
    
    # Create separate interaction files for each finetune main category
    finetune_inter_files = {}
    for main_category in finetune_categories:
        category_dir = os.path.join(output_dir, f"finetune_{main_category}")
        inter_file = os.path.join(category_dir, f"finetune_{main_category}.inter")
        finetune_inter_files[main_category] = open(inter_file, "w", encoding='utf-8')
        finetune_inter_files[main_category].write("user_id:token\titem_id:token\ttimestamp:float\n")
    
    with open(behaviors_file, "r", encoding='utf-8') as f, \
         open(pretrain_inter_file, "w", encoding='utf-8') as f_pretrain:
        
        # Write header for pretrain
        header = "user_id:token\titem_id:token\ttimestamp:float\n"
        f_pretrain.write(header)
        
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
            
            for item in impressions:
                if '-' in item and item.endswith('-1'):  # Only clicked items
                    item_id = item.split('-')[0]
                    if item_id in item_to_main_category:
                        main_category = item_to_main_category[item_id]
                        interaction_line = f'{user_id}\t{item_id}\t{timestamp}\n'
                        
                        if main_category in pretrain_set:
                            f_pretrain.write(interaction_line)
                        elif main_category in finetune_set:
                            finetune_inter_files[main_category].write(interaction_line)
    
    # Close finetune interaction files
    for f in finetune_inter_files.values():
        f.close()

def split_by_categories(news_file, behaviors_file, output_dir, pretrain_categories, finetune_categories, use_subcategory=True):
    """Split data into pretrain and separate finetune sets based on categories."""
    
    # First, read all news items and categorize them
    item_to_category = {}
    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            item_id = arr[0]
            # Use subcategory (column 2) instead of main category (column 1)
            category = arr[2] if use_subcategory and len(arr) > 2 else arr[1]
            item_to_category[item_id] = category
    
    # Create category sets for easy lookup
    pretrain_set = set(pretrain_categories)
    finetune_set = set(finetune_categories)
    
    # Get the output directory name for naming files
    output_dir_name = os.path.basename(output_dir.rstrip('/'))
    
    # Create pretrain item file with proper naming
    pretrain_item_file = os.path.join(output_dir, f"{output_dir_name}.item")
    
    with open(news_file, "r", encoding='utf-8') as f, \
         open(pretrain_item_file, "w", encoding='utf-8') as f_pretrain:
        
        # Write header for pretrain
        header = "item_id:token\ttitle:token_seq\tcategories:token_seq\n"
        f_pretrain.write(header)
        
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            
            item_id = arr[0]
            # Use subcategory (column 2) instead of main category (column 1)
            category = arr[2] if use_subcategory and len(arr) > 2 else arr[1]
            title = arr[3].replace('\t', ' ').replace('\n', ' ').replace('"', '').strip()
            
            output_line = f'{item_id}\t{title}\t{category}\n'
            
            if category in pretrain_set:
                f_pretrain.write(output_line)
    
    # Create separate item files for each finetune category
    finetune_files = {}
    for category in finetune_categories:
        category_dir = os.path.join(output_dir, f"finetune_{category}")
        os.makedirs(category_dir, exist_ok=True)
        
        item_file = os.path.join(category_dir, f"finetune_{category}.item")
        finetune_files[category] = open(item_file, "w", encoding='utf-8')
        finetune_files[category].write("item_id:token\ttitle:token_seq\tcategories:token_seq\n")
    
    # Fill finetune item files
    with open(news_file, "r", encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            
            item_id = arr[0]
            # Use subcategory (column 2) instead of main category (column 1)
            category = arr[2] if use_subcategory and len(arr) > 2 else arr[1]
            title = arr[3].replace('\t', ' ').replace('\n', ' ').replace('"', '').strip()
            
            output_line = f'{item_id}\t{title}\t{category}\n'
            
            if category in finetune_set:
                finetune_files[category].write(output_line)
    
    # Close finetune item files
    for f in finetune_files.values():
        f.close()
    
    # Create pretrain interaction file with proper naming
    pretrain_inter_file = os.path.join(output_dir, f"{output_dir_name}.inter")
    
    # Create separate interaction files for each finetune category
    finetune_inter_files = {}
    for category in finetune_categories:
        category_dir = os.path.join(output_dir, f"finetune_{category}")
        inter_file = os.path.join(category_dir, f"finetune_{category}.inter")
        finetune_inter_files[category] = open(inter_file, "w", encoding='utf-8')
        finetune_inter_files[category].write("user_id:token\titem_id:token\ttimestamp:float\n")
    
    with open(behaviors_file, "r", encoding='utf-8') as f, \
         open(pretrain_inter_file, "w", encoding='utf-8') as f_pretrain:
        
        # Write header for pretrain
        header = "user_id:token\titem_id:token\ttimestamp:float\n"
        f_pretrain.write(header)
        
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
            
            for item in impressions:
                if '-' in item and item.endswith('-1'):  # Only clicked items
                    item_id = item.split('-')[0]
                    if item_id in item_to_category:
                        category = item_to_category[item_id]
                        interaction_line = f'{user_id}\t{item_id}\t{timestamp}\n'
                        
                        if category in pretrain_set:
                            f_pretrain.write(interaction_line)
                        elif category in finetune_set:
                            finetune_inter_files[category].write(interaction_line)
    
    # Close finetune interaction files
    for f in finetune_inter_files.values():
        f.close()

def analyze_categories(news_file, use_subcategory=True):
    """Analyze and print category distribution in the dataset."""
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
    
    print("\nSubcategory distribution:")
    for category, count in sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    
    if use_subcategory:
        print(f"\nUsing SUBCATEGORIES for processing ({len(subcategory_counts)} unique subcategories)")
        return list(subcategory_counts.keys())
    else:
        print(f"\nUsing MAIN CATEGORIES for processing ({len(main_category_counts)} unique main categories)")
        return list(main_category_counts.keys())

def main(input_dir, output_dir, pretrain_categories=None, finetune_categories=None, 
         use_subcategory=True, split_by_main_keep_sub=False):
    os.makedirs(output_dir, exist_ok=True)
    news_file = os.path.join(input_dir, "news.tsv")
    behaviors_file = os.path.join(input_dir, "behaviors.tsv")
    
    # Analyze categories first
    print("Analyzing dataset...")
    available_categories = analyze_categories(news_file, use_subcategory)
    
    if pretrain_categories is None or finetune_categories is None:
        print("\nNo category split specified. Creating unified dataset...")
        
        # Get the output directory name for naming files
        output_dir_name = os.path.basename(output_dir.rstrip('/'))
        
        # Create unified files with proper naming
        item_file = os.path.join(output_dir, f"{output_dir_name}.item")
        interaction_file = os.path.join(output_dir, f"{output_dir_name}.inter")
        
        create_recbole_item_file(news_file, item_file, use_subcategory)
        create_recbole_inter_file(behaviors_file, interaction_file)
        
        print("Conversion finished.")
        print(f"Items -> {item_file}")
        print(f"Interactions -> {interaction_file}")
        
    else:
        if split_by_main_keep_sub:
            print(f"\nSplitting dataset by MAIN categories but keeping SUBCATEGORIES in files:")
            print(f"Pretrain main categories: {pretrain_categories}")
            print(f"Finetune main categories: {finetune_categories}")
            
            split_by_main_categories_keep_subcategories(news_file, behaviors_file, output_dir, 
                                                      pretrain_categories, finetune_categories)
        else:
            print(f"\nSplitting dataset:")
            print(f"Pretrain categories: {pretrain_categories}")
            print(f"Finetune categories: {finetune_categories}")
            
            split_by_categories(news_file, behaviors_file, output_dir, 
                              pretrain_categories, finetune_categories, use_subcategory)
        
        output_dir_name = os.path.basename(output_dir.rstrip('/'))
        print("Split conversion finished.")
        print(f"Pretrain files -> {output_dir}/{output_dir_name}.*")
        for category in finetune_categories:
            print(f"Finetune '{category}' files -> {output_dir}/finetune_{category}/finetune_{category}.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MIND dataset to RecBole format")
    parser.add_argument("--input_dir", required=True, help="Input directory containing news.tsv and behaviors.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory for RecBole files")
    parser.add_argument("--pretrain_categories", nargs='+', 
                       help="Categories to use for pretraining")
    parser.add_argument("--finetune_categories", nargs='+',
                       help="Categories to use for finetuning")
    parser.add_argument("--use_main_category", action='store_true',
                       help="Use main categories instead of subcategories (default: use subcategories)")
    parser.add_argument("--split_by_main_keep_sub", action='store_true',
                       help="Split by main categories but preserve subcategories in the output files")
    
    args = parser.parse_args()
    
    # By default use subcategories, unless --use_main_category is specified
    use_subcategory = not args.use_main_category
    
    main(args.input_dir, args.output_dir, args.pretrain_categories, args.finetune_categories, 
         use_subcategory, args.split_by_main_keep_sub)