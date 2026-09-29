import cv2
import numpy as np


def draw_boxes(
    image,
    boxes,
    color,
    label_prefix="Object"
):
    """
    Görüntü üzerine bounding box çizer.

    Desteklenen formatlar:

    1. YOLO ground-truth formatı:

    {
        "class_id": 0,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.3
    }

    2. Model prediction formatı:

    {
        "class_id": 0,
        "box": (x1, y1, x2, y2),
        "confidence": 0.85
    }
    """

    output = image.copy()

    height, width = output.shape[:2]

    for box in boxes:

        class_id = box.get("class_id", 0)

        # ==================================================
        # MODEL PREDICTION
        # ==================================================

        if "box" in box:

            x1, y1, x2, y2 = box["box"]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

        # ==================================================
        # GROUND TRUTH
        # ==================================================

        elif all(
            key in box
            for key in [
                "x_center",
                "y_center",
                "width",
                "height"
            ]
        ):

            x_center = box["x_center"]
            y_center = box["y_center"]

            box_width = box["width"]
            box_height = box["height"]

            x1 = int(
                (x_center - box_width / 2) * width
            )

            y1 = int(
                (y_center - box_height / 2) * height
            )

            x2 = int(
                (x_center + box_width / 2) * width
            )

            y2 = int(
                (y_center + box_height / 2) * height
            )

        else:
            continue

        # ==================================================
        # IMAGE BOUNDS
        # ==================================================

        x1 = max(
            0,
            min(x1, width - 1)
        )

        y1 = max(
            0,
            min(y1, height - 1)
        )

        x2 = max(
            0,
            min(x2, width - 1)
        )

        y2 = max(
            0,
            min(y2, height - 1)
        )

        # ==================================================
        # BOUNDING BOX
        # ==================================================

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ==================================================
        # LABEL
        # ==================================================

        if "confidence" in box:

            confidence = box["confidence"]

            label = (
                f"{label_prefix}: "
                f"{class_id} "
                f"{confidence:.2f}"
            )

        else:

            label = (
                f"{label_prefix}: "
                f"{class_id}"
            )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    return output


def visualize_predictions(
    image,
    ground_truth,
    predictions
):
    """
    Ground Truth ve model tahminlerini
    aynı görüntü üzerinde gösterir.

    Ground Truth:
        Yeşil

    Predictions:
        Kırmızı
    """

    # Görüntüyü kopyala
    result = image.copy()

    # ==================================================
    # GROUND TRUTH
    # ==================================================

    result = draw_boxes(
        result,
        ground_truth,
        color=(0, 255, 0),
        label_prefix="GT"
    )

    # ==================================================
    # MODEL PREDICTIONS
    # ==================================================

    result = draw_boxes(
        result,
        predictions,
        color=(0, 0, 255),
        label_prefix="Pred"
    )

    return result