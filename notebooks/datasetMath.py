# Process Math dataset train and test splits here

import os
import json
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import logging

# Set seed
import random
random.seed(42)

# Set numpy seed
import numpy as np
np.random.seed(42)
import hashlib

def get_problem_hash(problem: str) -> str:
    return hashlib.sha256(problem.encode()).hexdigest()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_math_data(split: str, data_dir: str = "datasets/math/raw_data") -> list:
    """
    Load math data from individual JSON files in the specified split directory.
    
    Args:
        split (str): The split to load ('train' or 'test')
        data_dir (str): Base directory containing the raw data
        
    Returns:
        list: List of dictionaries containing math problems and solutions
    """
    path_to_data = os.path.join(data_dir, split)
    all_data = []
    
    # Load all the pairs in the data
    for folder in tqdm(os.listdir(path_to_data), desc=f"Loading {split} data"):
        if not os.path.isdir(os.path.join(path_to_data, folder)):
            continue
            
        for file in tqdm(os.listdir(os.path.join(path_to_data, folder)), 
                        desc=f"Loading data for {folder}"):
            if not file.endswith(".json"):
                continue
                
            with open(os.path.join(path_to_data, folder, file), "r") as f:
                data = json.load(f)
                
            # Extract only the required fields
            example = {
                "question": data["problem"],
                "answer": data["solution"],
                "rationale": "",
                "type": data["type"],
                "level": data["level"],
                "category": folder,
                "split": split
            }
            all_data.append(example)
    
    return all_data

def take_equal_sample_size(dataset, sample_size):
    # Now sample feedback_sample_size examples such that each category is represented equally as much as possible
    feedback_dataset = []
    total_examples = 0
    for category in set(dataset["category"]):
        category_data = [x for x in dataset if x["category"] == category]
        feedback_dataset.extend(random.sample(category_data, sample_size // len(set(dataset["category"]))))
        total_examples = len(feedback_dataset)

    # Now take into account more data such that total becomes feedback_sample_size
    while total_examples < sample_size:
        category = random.choice(list(set(dataset["category"])))
        category_data = [x for x in dataset if x["category"] == category]
        feedback_dataset.extend(random.sample(category_data, 1))
        total_examples += 1

    return feedback_dataset

def sample_examples_from_datataset(dataset, sample_size=100):
    # Convert HuggingFace Dataset to list if needed
    if hasattr(dataset, 'to_list'):
        # It's a HuggingFace Dataset
        data_list = dataset.to_list()
    elif hasattr(dataset, '__iter__') and not isinstance(dataset, (str, dict)):
        # It's already a list or other iterable (but not string or dict)
        data_list = list(dataset)
    else:
        # Fallback: assume it's already a list
        data_list = dataset
    
    return random.sample(data_list, sample_size)

    

def create_math_dataset(validation_sample_size=100, strategy="random"):
    """
    Create a HuggingFace dataset from the math data with train and test splits.
    """
    # Load train and test data
    train_data = load_math_data("train")
    test_data = load_math_data("test")
    
    logger.info(f"Loaded {len(train_data)} training examples")
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)

    if strategy == "random":
        validation_dataset = sample_examples_from_datataset(train_dataset, validation_sample_size)
    elif strategy == "equal_sample_size":
        validation_dataset = take_equal_sample_size(train_dataset, validation_sample_size)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    logger.info(f"Validation dataset: {len(validation_dataset)} examples")
    train_dataset = Dataset.from_list([x for x in train_dataset if x not in validation_dataset])
    logger.info(f"Removed validation from train dataset: {len(train_dataset)} examples")


    validation_dataset = Dataset.from_list(validation_dataset)

    
    logger.info(f"Validation dataset: {len(validation_dataset)} examples")
    logger.info(f"Remaining Train dataset: {len(train_dataset)} examples")
    

    test_dataset = Dataset.from_list(test_data)
    logger.info(f"Test dataset: {len(test_dataset)} examples")
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": validation_dataset
    })

    
    # Save the dataset
    output_dir = f"datasets/math/raw_data/train_val_test_splits_dataset_{strategy}"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved dataset to {output_dir}")
    
    # Print some statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Train split: {len(train_dataset)} examples")
    logger.info(f"Test split: {len(test_dataset)} examples")
    
    # Print example types and levels
    train_types = set(train_dataset["type"])
    train_levels = set(train_dataset["level"])
    logger.info(f"\nUnique types in train: {train_types}")
    logger.info(f"Unique levels in train: {train_levels}")

def create_feedback_dataset(feedback_sample_size=100, strategy="random"):
    """
    Create a feedback dataset by sampling from training data and removing those samples from the main training set.
    
    Args:
        feedback_sample_size (int): Number of examples to sample for feedback dataset
        
    Returns:
        tuple: (feedback_dataset, remaining_train_dataset) where each is a list of examples
    """
    # Load all data
    main_data = load_from_disk(f"datasets/math/raw_data/train_val_test_splits_dataset_{strategy}")
    
    # Filter for training data
    train_data = main_data["train"]
    
    if strategy == "random":
        feedback_dataset = sample_examples_from_datataset(train_data, feedback_sample_size)
    elif strategy == "equal_sample_size":
        feedback_dataset = take_equal_sample_size(train_data, feedback_sample_size)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Create a set of problem hashes for efficient lookup
    feedback_problems = {get_problem_hash(x["question"]) for x in feedback_dataset}
    
    # Remove feedback samples from training data
    remaining_train_data = [
        x for x in train_data 
        if get_problem_hash(x["question"]) not in feedback_problems
    ]
    
    # Now create a dataset dict with train, test, validation and feedback splits
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(remaining_train_data),
        "test": Dataset.from_list(main_data["test"]),
        "validation": Dataset.from_list(main_data["validation"]),
        "feedback": Dataset.from_list(feedback_dataset)
    })

    # Print how many examples are in each split and how many in each category in each split
    for split in dataset_dict:
        logger.info(f"Split: {split}")
        logger.info(f"Number of examples: {len(dataset_dict[split])}")
        for category in set(dataset_dict[split]["category"]):
            logger.info(f"Category: {category}, Number of examples: {len([x for x in dataset_dict[split] if x['category'] == category])}")

    # Save the dataset
    output_dir = f"datasets/math/raw_data/train_val_test_splits_dataset_feedback_{feedback_sample_size}"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved dataset to {output_dir}")

if __name__ == "__main__":
    create_math_dataset()
    create_feedback_dataset(100)
    create_feedback_dataset(400)
    create_feedback_dataset(1600)
    create_feedback_dataset(6400)
    