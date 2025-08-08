from datasets import load_from_disk, load_dataset as hf_load_dataset


def load_dataset_smart(path: str):
    """Load dataset either from local disk or HuggingFace Hub.
    
    Args:
        path: Local path (starts with . or /) or HuggingFace dataset name
    """
    if path.startswith(".") or path.startswith("/"):
        return load_from_disk(path)
    
    # Load from HuggingFace Hub
    dataset = hf_load_dataset(path)
    
    # Handle DatasetDict vs Dataset - return the train split if it exists
    if hasattr(dataset, 'keys') and 'train' in dataset:
        return dataset['train']
    elif hasattr(dataset, 'keys') and len(dataset.keys()) == 1:
        # If there's only one split, return it
        split_name = list(dataset.keys())[0]
        return dataset[split_name]
    else:
        # Return as-is if it's already a Dataset
        return dataset


# Backward compatibility alias for existing tests
load_dataset = load_dataset_smart


def push_dataset_to_hub(dataset, username: str, dataset_name: str, token: str = None, private: bool = True):
    """Push dataset to HuggingFace Hub.
    
    Args:
        dataset: The dataset to push
        username: HuggingFace username
        dataset_name: Name for the dataset on HuggingFace Hub
        token: HuggingFace token for authentication
        private: Whether to make the dataset private (default: True)
    """
    repo_id = f"{username}/{dataset_name}"
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private
    )
    return repo_id
