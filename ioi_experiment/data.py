from datasets import load_dataset


def load_ioi_dataset(split: str = "train"):
    dataset = load_dataset("mib-bench/ioi", split=split)
    return dataset


def load_ioi_splits():
    dataset = load_dataset("mib-bench/ioi")
    return dataset


if __name__ == "__main__":
    dataset = load_ioi_splits()
    print(dataset)

    train = dataset["train"]
    print(f"\nFirst example:\n{train[0]}")
