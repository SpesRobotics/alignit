from datasets import load_from_disk

def main():
    dataset = load_from_disk("data/duck")

    # Print dataset info
    print(dataset)
    print(dataset.features)

    for i, example in enumerate(dataset):
        print(f"Example {i}:")
        print("Images:", example["images"])
        print("Action:", example["action"])

        # show images
        for img in example["images"]:
            img.show()

        if i >= 5:
            break


if __name__ == "__main__":
    main()
