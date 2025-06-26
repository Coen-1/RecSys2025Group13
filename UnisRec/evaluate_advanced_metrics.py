# evaluate_saved_model.py
# Purpose: Load a saved model checkpoint and evaluate it using RecBole's standard metrics.

import argparse
import torch

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_logger, get_logger, set_color # For RecBole style logging

# Custom Model and Dataset Imports - CRITICAL
from unisrec import UniSRec 
from data.dataset import UniSRecDataset # Use UniSRecDataset for fine-tuned models
# from data.dataset import PretrainUniSRecDataset # Use this if evaluating a model saved from pretrain.py

def evaluate_checkpoint(model_checkpoint_path):
    print(f"Loading checkpoint from: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Override device to use GPU if available, or stick to CPU if not
    config['use_gpu'] = torch.cuda.is_available()
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize RecBole logger
    init_logger(config)
    logger = get_logger()
    
    logger.info("Loaded Configuration from checkpoint:")
    logger.info(config)
    
    # Prepare the dataset specified in the loaded config
    logger.info(f"Loading and preparing dataset: {config['dataset']}")
    
    # Determine which dataset class to use based on train_stage if available in config
    # This is important if you are evaluating a model from pre-training vs. fine-tuning
    if config['train_stage'] == 'pretrain':
        from data.dataset import PretrainUniSRecDataset # Import here to avoid circular if not needed
        dataset = PretrainUniSRecDataset(config)
        logger.info("Using PretrainUniSRecDataset for evaluation.")
    else: # Assume 'inductive_ft' or 'transductive_ft' or if train_stage not in config
        dataset = UniSRecDataset(config)
        logger.info("Using UniSRecDataset for evaluation.")
        
    logger.info(dataset) # This will print dataset stats
    # For evaluation, we typically only need the test data split
    # data_preparation will handle splitting based on config
    train_data, valid_data, test_data = data_preparation(config, dataset) # Need all for trainer init

    # Instantiate the model directly
    logger.info("Instantiating UniSRec model...")
    model = UniSRec(config, dataset.dataset).to(config['device']) # Pass dataset.dataset to model
                                                                # which is the underlying RecBole Dataset object
    
    logger.info(f"Loading model state_dict from {model_checkpoint_path}")
    model.load_state_dict(checkpoint['state_dict'])
    # logger.info(model) # Model summary can be very verbose

    # Get RecBole's trainer for evaluation
    # We need to import get_trainer from recbole.utils
    from recbole.utils import get_trainer as recbole_get_trainer 
    trainer = recbole_get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Evaluate the loaded model on the test data
    logger.info(f"Evaluating loaded model [{model_checkpoint_path}] on test data...")
    # load_best_model=False is crucial here because we are providing the *exact* model
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color(f'Test result for {model_checkpoint_path}', 'yellow') + f': {test_result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a saved RecBole model checkpoint for standard metrics.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .pth model file to evaluate.')
    args = parser.parse_args()
    
    evaluate_checkpoint(args.model_path)