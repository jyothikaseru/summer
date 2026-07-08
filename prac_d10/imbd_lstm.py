from datasets import load_dataset

def main():
    print("Loading IMDb dataset...")

    dataset = load_dataset("imdb")

    print("Dataset loaded successfully!\n")

    print(dataset)

if __name__ == "__main__":
    main()