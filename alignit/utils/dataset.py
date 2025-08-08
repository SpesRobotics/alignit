from datasets import load_from_disk, load_dataset


def load_dataset(path: str):
    if path.startswith(".") or path.startswith("/"):
        return load_from_disk(path)
    return load_dataset(path)
