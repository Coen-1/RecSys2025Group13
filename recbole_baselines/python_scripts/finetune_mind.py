from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.utils import init_seed, get_model, get_trainer
from recbole.data import create_dataset, data_preparation
import datetime
import argparse
import torch
import numpy as np
from collections import Counter
import math

# GiniCoefficient class
class GiniCoefficient:
    def gini_coefficient(self, values):
        arr = np.array(values, dtype=float)
        if arr.sum() == 0: return 0.0
        n = arr.size
        if n <= 1 or np.all(arr == arr[0]): return 0.0
        arr = np.sort(arr)
        mu = arr.mean()
        if mu == 0: return 0.0
        index = np.arange(1, n + 1)
        gini_val = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini_val

    def calculate_list_gini(self, articles, key="category"):
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        if not freqs: return 0.0
        return self.gini_coefficient(list(freqs.values()))

# New Diversity Metrics for MIND Dataset
class NewsRecommendationMetrics:
    def __init__(self):
        pass
    
    def calculate_category_coverage(self, recommended_categories, total_categories_in_catalog):
        """
        Calculate the percentage of unique categories covered in recommendations
        Higher values indicate better diversity
        """
        if not recommended_categories or not total_categories_in_catalog:
            return 0.0
        
        unique_recommended = set(recommended_categories)
        total_unique = set(total_categories_in_catalog)
        
        coverage = len(unique_recommended) / len(total_unique)
        return coverage
    
    def calculate_intra_list_diversity(self, user_recommendations_with_categories):
        """
        Calculate average pairwise diversity within recommendation lists
        For each user's recommendation list, compute pairwise category differences
        """
        if not user_recommendations_with_categories:
            return 0.0
        
        total_diversity = 0.0
        valid_users = 0
        
        for user_categories in user_recommendations_with_categories:
            if len(user_categories) < 2:
                continue
                
            # Calculate pairwise diversity (Jaccard distance for categories)
            diversity_sum = 0.0
            pair_count = 0
            
            for i in range(len(user_categories)):
                for j in range(i + 1, len(user_categories)):
                    cat_i = set(user_categories[i]) if isinstance(user_categories[i], list) else {user_categories[i]}
                    cat_j = set(user_categories[j]) if isinstance(user_categories[j], list) else {user_categories[j]}
                    
                    # Jaccard distance (1 - Jaccard similarity)
                    intersection = len(cat_i.intersection(cat_j))
                    union = len(cat_i.union(cat_j))
                    
                    if union > 0:
                        jaccard_distance = 1 - (intersection / union)
                        diversity_sum += jaccard_distance
                        pair_count += 1
            
            if pair_count > 0:
                user_diversity = diversity_sum / pair_count
                total_diversity += user_diversity
                valid_users += 1
        
        return total_diversity / valid_users if valid_users > 0 else 0.0
    
    def calculate_personalization(self, all_user_recommendations):
        """
        Calculate how different users' recommendations are from each other
        Measures system-level personalization
        """
        if len(all_user_recommendations) < 2:
            return 0.0
        
        total_dissimilarity = 0.0
        pair_count = 0
        
        users = list(all_user_recommendations.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                rec_i = set(all_user_recommendations[users[i]])
                rec_j = set(all_user_recommendations[users[j]])
                
                # Calculate Jaccard distance between user recommendation lists
                intersection = len(rec_i.intersection(rec_j))
                union = len(rec_i.union(rec_j))
                
                if union > 0:
                    dissimilarity = 1 - (intersection / union)
                    total_dissimilarity += dissimilarity
                    pair_count += 1
        
        return total_dissimilarity / pair_count if pair_count > 0 else 0.0

# --- Enhanced Metrics Calculation Function ---
def calculate_enhanced_metrics_from_recs(
    recommended_item_ids_tensor, k_for_metrics, 
    base_dataset_object, 
    config_obj,
    existing_results_dict
):
    """
    Calculate multiple diversity and coverage metrics for news recommendations
    """
    category_field_name_in_dataset = 'categories'
    
    # Check if item features are available
    if not hasattr(base_dataset_object, 'item_feat') or base_dataset_object.item_feat is None:
        print("Warning (Enhanced Metrics): Item features (item_feat) not loaded. Cannot calculate enhanced metrics.")
        for metric_name in ['category_coverage', 'intra_list_diversity', 'personalization', 'gini_categories']:
            existing_results_dict[f'{metric_name}@{k_for_metrics}'] = "Error: item_feat missing"
        return existing_results_dict
    
    if category_field_name_in_dataset not in base_dataset_object.item_feat:
        print(f"Warning (Enhanced Metrics): Category field '{category_field_name_in_dataset}' not found.")
        for metric_name in ['category_coverage', 'intra_list_diversity', 'personalization', 'gini_categories']:
            existing_results_dict[f'{metric_name}@{k_for_metrics}'] = f"Error: {category_field_name_in_dataset} missing"
        return existing_results_dict

    all_item_categories_tensor = base_dataset_object.item_feat[category_field_name_in_dataset]
    
    # Get padding ID
    padding_id_for_categories = 0 
    pad_token_str = config_obj['PAD_TOKEN'] 
    if category_field_name_in_dataset in base_dataset_object.field2token_id and \
       pad_token_str in base_dataset_object.field2token_id[category_field_name_in_dataset]:
        padding_id_for_categories = base_dataset_object.field2token_id[category_field_name_in_dataset][pad_token_str]
    
    # Extract all categories and organize by user
    all_category_tokens_in_recommendations = []
    user_recommendations_with_categories = []
    all_user_recommendations = {}
    
    # Get all unique categories in the catalog for coverage calculation
    all_catalog_categories = set()
    for item_idx in range(all_item_categories_tensor.shape[0]):
        item_categories = all_item_categories_tensor[item_idx].tolist()
        for cat_id in item_categories:
            if cat_id != padding_id_for_categories:
                all_catalog_categories.add(cat_id)
    
    # Process recommendations per user
    for user_idx, user_recs_item_ids_1_indexed in enumerate(recommended_item_ids_tensor):
        user_categories = []
        user_items = []
        
        for internal_item_id_1_indexed in user_recs_item_ids_1_indexed.tolist():
            if internal_item_id_1_indexed == 0: 
                continue
            
            internal_item_id_0_indexed = internal_item_id_1_indexed - 1
            if not (0 <= internal_item_id_0_indexed < all_item_categories_tensor.shape[0]): 
                continue
            
            item_specific_category_token_ids = all_item_categories_tensor[internal_item_id_0_indexed].tolist()
            item_categories = []
            
            for cat_token_id in item_specific_category_token_ids:
                if cat_token_id != padding_id_for_categories:
                    all_category_tokens_in_recommendations.append(cat_token_id)
                    item_categories.append(cat_token_id)
            
            if item_categories:  # Only add if item has valid categories
                user_categories.append(item_categories[0])  # Use first category for simplicity
                user_items.append(internal_item_id_1_indexed)
        
        if user_categories:
            user_recommendations_with_categories.append(user_categories)
            all_user_recommendations[f'user_{user_idx}'] = user_items
    
    # Initialize metrics calculator
    metrics_calculator = NewsRecommendationMetrics()
    gini_calculator = GiniCoefficient()
    
    # Calculate Gini Coefficient
    if all_category_tokens_in_recommendations:
        category_token_counts = Counter(all_category_tokens_in_recommendations)
        gini_val = gini_calculator.gini_coefficient(list(category_token_counts.values()))
        existing_results_dict[f'gini_categories@{k_for_metrics}'] = gini_val
        print(f"Gini Coefficient (categories@{k_for_metrics}): {gini_val:.4f}")
    else:
        existing_results_dict[f'gini_categories@{k_for_metrics}'] = 0.0
    
    # Calculate Category Coverage
    coverage = metrics_calculator.calculate_category_coverage(
        all_category_tokens_in_recommendations, 
        list(all_catalog_categories)
    )
    existing_results_dict[f'category_coverage@{k_for_metrics}'] = coverage
    print(f"Category Coverage@{k_for_metrics}: {coverage:.4f}")
    
    # Calculate Intra-List Diversity
    ild = metrics_calculator.calculate_intra_list_diversity(user_recommendations_with_categories)
    existing_results_dict[f'intra_list_diversity@{k_for_metrics}'] = ild
    print(f"Intra-List Diversity@{k_for_metrics}: {ild:.4f}")
    
    # Calculate Personalization
    personalization = metrics_calculator.calculate_personalization(all_user_recommendations)
    existing_results_dict[f'personalization@{k_for_metrics}'] = personalization
    print(f"Personalization@{k_for_metrics}: {personalization:.4f}")
    
    return existing_results_dict

# --- Original Gini Calculation Function (kept for compatibility) ---
def calculate_gini_for_categories_from_recs(
    recommended_item_ids_tensor, k_for_gini, 
    base_dataset_object, 
    config_obj,
    existing_results_dict
):
    # Use the enhanced metrics function instead
    return calculate_enhanced_metrics_from_recs(
        recommended_item_ids_tensor, k_for_gini, 
        base_dataset_object, config_obj, existing_results_dict
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RecBole experiments with MIND dataset metrics.")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    config = Config(config_file_list=[args.config_file])
    if args.nproc > 1: config['nproc'] = args.nproc

    print(f"Start Time: {datetime.datetime.now()}")
    start_time = datetime.datetime.now()

    if args.load_model and 'pretrained_model_path' in config and config['pretrained_model_path']:
        init_seed(config['seed'], config['reproducibility'])
        dataset_obj = create_dataset(config) 
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)
        
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        
        checkpoint = torch.load(config['pretrained_model_path'], map_location=config['device'])
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        keys_to_delete_from_pretrained = []
        
        for key in list(pretrained_state_dict.keys()):
            if key in model_state_dict:
                if pretrained_state_dict[key].shape != model_state_dict[key].shape:
                    print(f"Size mismatch for {key}: pretrained {pretrained_state_dict[key].shape} vs current {model_state_dict[key].shape}")
                    if 'embedding' in key:
                        min_dim0_size = min(pretrained_state_dict[key].shape[0], model_state_dict[key].shape[0])
                        current_slice = [slice(None)] * model_state_dict[key].dim()
                        pretrained_slice = [slice(None)] * pretrained_state_dict[key].dim()
                        current_slice[0] = slice(0, min_dim0_size)
                        pretrained_slice[0] = slice(0, min_dim0_size)
                        for i in range(1, model_state_dict[key].dim()):
                            if pretrained_state_dict[key].shape[i] != model_state_dict[key].shape[i]:
                                min_dim_i_size = min(pretrained_state_dict[key].shape[i], model_state_dict[key].shape[i])
                                current_slice[i] = slice(0, min_dim_i_size)
                                pretrained_slice[i] = slice(0, min_dim_i_size)
                        model_state_dict[key][tuple(current_slice)] = pretrained_state_dict[key][tuple(pretrained_slice)]
                        print(f"Copied overlapping part for {key}")
                    keys_to_delete_from_pretrained.append(key)
            else: 
                keys_to_delete_from_pretrained.append(key)
        
        for key_to_del in keys_to_delete_from_pretrained:
            if key_to_del in pretrained_state_dict: 
                del pretrained_state_dict[key_to_del]
        
        model.load_state_dict(pretrained_state_dict, strict=False)
        print(f"Loaded pretrained model from {config['pretrained_model_path']} (strict=False)")

        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        
        print("Starting fine-tuning...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=config['show_progress']
        )
        print(f"Fine-tuning complete. Best validation result: {best_valid_result}")

        is_main_process = True
        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            is_main_process = trainer.accelerator.is_main_process
        
        test_result_std_metrics = {}
        if is_main_process:
            print("Evaluating on test set for standard metrics (main process)...")
            test_result_std_metrics = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
            
            final_model_for_metrics = trainer.model
            final_model_for_metrics.eval()

            print("Calculating enhanced diversity metrics for MIND dataset (main process)...")
            
            k_for_metrics = config['topk'][0] if isinstance(config['topk'], list) else config['topk']

            all_topk_item_indices_list = []
            with torch.no_grad():
                for batch_idx, batched_interaction_tuple_or_obj in enumerate(test_data):
                    actual_interaction = batched_interaction_tuple_or_obj[0] if isinstance(batched_interaction_tuple_or_obj, tuple) else batched_interaction_tuple_or_obj
                    interaction_on_device = actual_interaction.to(final_model_for_metrics.device)
                    
                    if hasattr(final_model_for_metrics, 'full_sort_predict'):
                        scores = final_model_for_metrics.full_sort_predict(interaction_on_device)
                    elif hasattr(final_model_for_metrics, 'predict'):
                        scores = final_model_for_metrics.predict(interaction_on_device)
                        print(f"Warning: Model lacks 'full_sort_predict'. Using 'predict()'. Output shape: {scores.shape}")
                    else:
                        print(f"Error: Model has neither full_sort_predict nor predict method. Skipping batch {batch_idx}.")
                        all_topk_item_indices_list.append(torch.empty(0, k_for_metrics, dtype=torch.long).cpu())
                        continue
                    
                    if scores.dim() == 1:
                        if interaction_on_device[config['USER_ID_FIELD']].shape[0] == 1:
                            scores = scores.unsqueeze(0)
                        else:
                            print(f"Error: Scores tensor is 1D but batch size > 1. Skipping batch {batch_idx}.")
                            all_topk_item_indices_list.append(torch.empty(0, k_for_metrics, dtype=torch.long).cpu())
                            continue
                    
                    if scores.dim() != 2:
                        print(f"Error: Unexpected scores dimension {scores.dim()}. Skipping batch {batch_idx}.")
                        all_topk_item_indices_list.append(torch.empty(0, k_for_metrics, dtype=torch.long).cpu())
                        continue
                    
                    if scores.shape[1] < k_for_metrics:
                        print(f"Warning: Not enough items ({scores.shape[1]}) for top-{k_for_metrics}. Using all available.")
                        current_k = scores.shape[1]
                    else:
                        current_k = k_for_metrics

                    if current_k == 0:
                         all_topk_item_indices_list.append(torch.empty(scores.shape[0], 0, dtype=torch.long).cpu())
                         continue

                    _, topk_indices_batch = torch.topk(scores, k=current_k, dim=1)
                    if current_k < k_for_metrics:
                         padding = torch.zeros((topk_indices_batch.shape[0], k_for_metrics - current_k), dtype=torch.long)
                         topk_indices_batch = torch.cat((topk_indices_batch.cpu(), padding), dim=1)

                    all_topk_item_indices_list.append(topk_indices_batch.cpu() + 1)

            if all_topk_item_indices_list:
                valid_tensors = [t for t in all_topk_item_indices_list if t.shape[0] > 0 and t.shape[1] > 0]
                if valid_tensors:
                    final_recommended_item_ids_tensor = torch.cat(valid_tensors, dim=0)
                    test_result_std_metrics = calculate_enhanced_metrics_from_recs(
                        final_recommended_item_ids_tensor,
                        k_for_metrics,
                        dataset_obj, 
                        config,
                        test_result_std_metrics
                    )
                else:
                    print("Warning: No valid recommendations generated for metrics calculation.")
                    for metric_name in ['category_coverage', 'intra_list_diversity', 'personalization', 'gini_categories']:
                        test_result_std_metrics[f'{metric_name}@{k_for_metrics}'] = "Error: No valid recs"
            else:
                print("Warning: No recommendations generated for metrics calculation.")
                for metric_name in ['category_coverage', 'intra_list_diversity', 'personalization', 'gini_categories']:
                    test_result_std_metrics[f'{metric_name}@{k_for_metrics}'] = "Error: No recs generated"
            
            print(f"Final Test Result (with enhanced MIND metrics): {test_result_std_metrics}")
        else:
            print("Skipping test evaluation (not on main DDP process).")

    else:
        print("Running standard RecBole experiment (enhanced metrics not calculated in this path).")
        run_recbole(config_file_list=[args.config_file], config_dict={'nproc': args.nproc} if args.nproc > 1 else {}, saved=True)

    end_time = datetime.datetime.now()
    print(f"--- Experiment Finished ---")
    print(f"End Time: {end_time}")
    print(f"Total Duration: {end_time - start_time}")