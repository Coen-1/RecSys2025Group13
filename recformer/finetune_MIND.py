import os
import json
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from collections import defaultdict, Counter

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset


def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for a list of values.
    This matches the baseline implementation exactly.
    """
    arr = np.array(values, dtype=float)
    if arr.sum() == 0: 
        return 0.0
    
    n = arr.size
    if n <= 1 or np.all(arr == arr[0]): 
        return 0.0
    
    arr = np.sort(arr)
    mu = arr.mean()
    if mu == 0: 
        return 0.0
    
    index = np.arange(1, n + 1)
    gini_val = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
    return gini_val


def calculate_intra_list_diversity(recommended_items, item_meta_dict, id2item, k=10):
    """
    Calculate intra-list diversity based on category diversity within top-k recommendations.
    """
    if len(recommended_items) == 0:
        return 0.0
    
    # Get categories for recommended items
    categories = []
    for item_id in recommended_items[:k]:
        item_key = id2item.get(item_id)
        if item_key and item_key in item_meta_dict:
            category = item_meta_dict[item_key].get('category', 'unknown')
            categories.append(category)
    
    if len(categories) == 0:
        return 0.0
    
    # Calculate diversity as 1 - Herfindahl index
    category_counts = Counter(categories)
    total = len(categories)
    herfindahl = sum((count / total) ** 2 for count in category_counts.values())
    diversity = 1 - herfindahl
    
    return diversity


def calculate_category_coverage(all_recommended_items, item_meta_dict, id2item, k=10):
    """
    Calculate category coverage across all users' top-k recommendations.
    """
    recommended_categories = set()
    all_categories = set()
    
    # Get all possible categories
    for item_key, meta in item_meta_dict.items():
        category = meta.get('category', 'unknown')
        all_categories.add(category)
    
    # Get categories from all recommendations
    for user_recs in all_recommended_items:
        for item_id in user_recs[:k]:
            item_key = id2item.get(item_id)
            if item_key and item_key in item_meta_dict:
                category = item_meta_dict[item_key].get('category', 'unknown')
                recommended_categories.add(category)
    
    if len(all_categories) == 0:
        return 0.0
    
    coverage = len(recommended_categories) / len(all_categories)
    return coverage


class DiversityRanker(Ranker):
    """
    Extended Ranker class that includes diversity metrics calculation.
    Modified to calculate Gini on category distributions like baselines.
    """
    def __init__(self, metrics_ks, item_meta_dict=None, id2item=None):
        super().__init__(metrics_ks)
        self.item_meta_dict = item_meta_dict
        self.id2item = id2item
        self.all_recommendations = []  # Store all recommendations for category coverage
        self.all_category_tokens_in_recommendations = []  # Store categories for Gini calculation
    
    def __call__(self, scores, labels):
        # Get original metrics
        original_metrics = super().__call__(scores, labels)
        
        # Get top-k recommendations for diversity metrics
        batch_size = scores.size(0)
        k = 10  # Fixed at 10 for diversity metrics
        
        _, top_indices = torch.topk(scores, k, dim=1)
        top_indices = top_indices.cpu().numpy()
        
        diversity_scores = []
        batch_recommendations = []
        batch_categories = []
        
        for i in range(batch_size):
            # Calculate intra-list diversity
            recommended_items = top_indices[i].tolist()
            diversity = calculate_intra_list_diversity(
                recommended_items, self.item_meta_dict, self.id2item, k
            )
            diversity_scores.append(diversity)
            
            # Store recommendations for category coverage
            batch_recommendations.append(recommended_items)
            
            # Extract categories for this user's recommendations (for Gini calculation)
            user_categories = []
            for item_id in recommended_items:
                item_key = self.id2item.get(item_id)
                if item_key and item_key in self.item_meta_dict:
                    category = self.item_meta_dict[item_key].get('category', 'UNKNOWN')
                    user_categories.append(category)
                else:
                    user_categories.append('UNKNOWN')
            
            batch_categories.extend(user_categories)
        
        # Store recommendations for global category coverage calculation
        self.all_recommendations.extend(batch_recommendations)
        
        # Store categories for Gini calculation (matches baseline approach)
        self.all_category_tokens_in_recommendations.extend(batch_categories)
        
        # Calculate batch average diversity
        avg_diversity = np.mean(diversity_scores)
        
        # Calculate Gini coefficient on category distribution (like baselines)
        if self.all_category_tokens_in_recommendations:
            category_token_counts = Counter(self.all_category_tokens_in_recommendations)
            gini_val = calculate_gini_coefficient(list(category_token_counts.values()))
        else:
            gini_val = 0.0
        
        # Return original metrics + diversity metrics
        return original_metrics + [gini_val, avg_diversity]
    
    def get_category_coverage(self, k=10):
        """
        Calculate category coverage across all stored recommendations.
        """
        if not self.all_recommendations:
            return 0.0
        
        return calculate_category_coverage(
            self.all_recommendations, self.item_meta_dict, self.id2item, k
        )
    
    def reset_recommendations(self):
        """Reset stored recommendations for new evaluation round."""
        self.all_recommendations = []
        self.all_category_tokens_in_recommendations = []


tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval(model, dataloader, args, item_meta_dict=None, id2item=None):

    model.eval()

    ranker = DiversityRanker(args.metric_ks, item_meta_dict, id2item)
    average_meter_set = AverageMeterSet()

    # Reset recommendations for this evaluation
    ranker.reset_recommendations()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)

        metrics = {}
        # Original metrics
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-5]  # Adjusted index due to new metrics
        metrics["AUC"] = res[-4]  # Adjusted index due to new metrics
        
        # New diversity metrics (Gini now calculated on categories like baselines)
        metrics["Gini@10"] = res[-2]
        metrics["ILD@10"] = res[-1]  # Intra-List Diversity

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    # Calculate category coverage across all users
    if item_meta_dict and id2item:
        category_coverage = ranker.get_category_coverage(k=10)
        average_meter_set.update("CategoryCoverage@10", category_coverage)

    average_metrics = average_meter_set.averages()

    return average_metrics

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()

def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    else:
        print(f'Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
    
    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)

    model.to(args.device) # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval(model, test_loader, args, item_meta_dict, id2item)
    print(f'Test set: {test_metrics}')
    
    best_target = float('-inf')
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args, item_meta_dict, id2item)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break
    
    print('Load best model in stage 1.')
    model.load_state_dict(torch.load(path_ckpt))

    patient = 3

    for epoch in range(args.num_train_epochs):

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args, item_meta_dict, id2item)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break

    print('Test with the best checkpoint.')  
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = eval(model, test_loader, args, item_meta_dict, id2item)
    print(f'Test set: {test_metrics}')
               
if __name__ == "__main__":
    main()