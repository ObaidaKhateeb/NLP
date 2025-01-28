import os 
from datasets import load_dataset, load_from_disk


def main():
    #choosing subset of the IMDB dataset, it the subset already exists to load it (section 1)
    if os.path.exists('imdb_subset'):
        subset = load_from_disk("imdb_subset")
    else: 
        dataset = load_dataset("imdb") #loading the IMDB dataset
        subset = dataset["train"].shuffle(seed=42).select(range(500))
        subset.save_to_disk("imdb_subset")
        subset = load_from_disk("imdb_subset")

if __name__ == "__main__":
    main()

