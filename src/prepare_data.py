from datasets import load_dataset, concatenate_datasets

def prepare_dataset():
    # Load full IMDB dataset - A binary sentiment analysis dataset consisting of 50,000 reviews from the IMDb labeled as + or -
    dataset = load_dataset("imdb")

    # Shuffle both splits & set seed
    dataset = dataset.shuffle(seed=42)

    # Take a stratified balanced subset of each label to avoid bias toward one label
    def balance_subset(ds, size_per_class):
        label_0 = ds.filter(lambda example: example["label"] == 0).select(range(size_per_class))
        label_1 = ds.filter(lambda example: example["label"] == 1).select(range(size_per_class))
        # join the two dataset to create a balanced dataset
        return concatenate_datasets([label_0, label_1])

    # 2500 +ve & 2500 -ve reviews in train data = total 5000 reviews
    train_dataset = balance_subset(dataset["train"], 2500)

    # 500 +ve & 500 -ve reviews in test data = total 1000  
    test_dataset = balance_subset(dataset["test"], 500)     

    # Save datasets to local directories
    train_dataset.save_to_disk("data/train")
    test_dataset.save_to_disk("data/test")

    # Print the sizes of train & test dataset
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset

# Makes sure the function runs only if this file is executed as a script
if __name__ == "__main__":
    prepare_dataset()
