import cv2

from src.dataset_manager import DatasetManager
from src.model_runner import ModelRunner
from src.evaluator import Evaluator


def main():

    dataset = DatasetManager(
        r"C:\Users\hp\datasets\coco8"
    )

    model = ModelRunner("yolo26n.pt")

    evaluator = Evaluator(
        iou_threshold=0.5
    )

    images = dataset.get_images("val")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print(
        f"Number of validation images: {len(images)}"
    )

    for image in images:

        print("\n" + "=" * 60)
        print(f"Image: {image}")

        labels = dataset.get_labels(image)

        image_data = cv2.imread(str(image))

        height, width = image_data.shape[:2]

        predictions = model.predict(
            str(image)
        )

        result = evaluator.evaluate_image(
            labels,
            predictions,
            width,
            height
        )

        tp = result["tp"]
        fp = result["fp"]
        fn = result["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"Ground Truth: {len(labels)}")
        print(f"Predictions: {len(predictions)}")

        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")

    precision = evaluator.calculate_precision(
        total_tp,
        total_fp
    )

    recall = evaluator.calculate_recall(
        total_tp,
        total_fn
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


if __name__ == "__main__":
    main()