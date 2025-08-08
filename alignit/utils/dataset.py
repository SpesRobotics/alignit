from datasets import load_from_disk, load_dataset as hf_load_dataset


def load_dataset(path: str):
    """Load a dataset from disk if given a local path, otherwise from the hub.

    This function wraps Hugging Face Datasets loaders while accepting either a
    filesystem path (./ or /) or a dataset name.
    """
    if path.startswith(".") or path.startswith("/"):
        return load_from_disk(path)
    return hf_load_dataset(path)
