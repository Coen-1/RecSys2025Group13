import os
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils import read_json
from recformer import RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset
from finetune import eval, load_data

def main():
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    
    # Optional arguments
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    
    # Set device
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device >= 0 else torch.device('cpu')
    
    # Load data
    _, _, test, item_meta_dict, item2id, id2item = load_data(args)
    
    # Initialize model and tokenizer
    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    model = RecformerForSeqRec.from_pretrained(args.model_name_or_path, config)
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    # Prepare evaluation dataset and dataloader
    test_dataset = RecformerEvalDataset(test, item_meta_dict, tokenizer, item2id)
    test_collator = EvalDataCollatorWithPadding(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=args.dataloader_num_workers
    )
    
    # Run evaluation
    print("Evaluating model...")
    metrics = eval(model, test_loader, args)
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 