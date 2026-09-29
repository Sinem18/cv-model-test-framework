
from ultralytics.data.utils import check_det_dataset


def main():
    dataset = check_det_dataset("coco8.yaml")

    print("Dataset downloaded successfully.")
    print(f"Dataset path: {dataset['path']}")
    print(f"Train: {dataset['train']}")
    print(f"Validation: {dataset['val']}")


if __name__ == "__main__":
    main()