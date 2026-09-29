
class Evaluator:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Bounding box format:
        (x1, y1, x2, y2)
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection_area = (
            intersection_width * intersection_height
        )

        box1_width = max(0, box1[2] - box1[0])
        box1_height = max(0, box1[3] - box1[1])

        box2_width = max(0, box2[2] - box2[0])
        box2_height = max(0, box2[3] - box2[1])

        box1_area = box1_width * box1_height
        box2_area = box2_width * box2_height

        union_area = (
            box1_area
            + box2_area
            - intersection_area
        )

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def yolo_to_xyxy(label, image_width, image_height):
        """
        YOLO format:
        {
            class_id,
            x_center,
            y_center,
            width,
            height
        }

        Output:
        (x1, y1, x2, y2)
        """

        x_center = label["x_center"] * image_width
        y_center = label["y_center"] * image_height

        width = label["width"] * image_width
        height = label["height"] * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        return (
            x1,
            y1,
            x2,
            y2
        )

    def evaluate_image(
        self,
        ground_truths,
        predictions,
        image_width,
        image_height
    ):
        """
        Bir görüntüdeki Ground Truth ve
        Prediction sonuçlarını karşılaştırır.
        """

        gt_objects = []

        for gt in ground_truths:

            box = self.yolo_to_xyxy(
                gt,
                image_width,
                image_height
            )

            gt_objects.append({
                "class_id": gt["class_id"],
                "box": box,
                "matched": False
            })

        pred_objects = []

        for prediction in predictions:

            pred_objects.append({
                "class_id": prediction["class_id"],
                "box": prediction["box"],
                "confidence": prediction["confidence"],
                "matched": False
            })

        matches = []

        # Her prediction için en iyi Ground Truth'u bul
        for pred in pred_objects:

            best_gt = None
            best_iou = 0.0

            for gt in gt_objects:

                if gt["matched"]:
                    continue

                # Sınıf farklıysa eşleşme yok
                if pred["class_id"] != gt["class_id"]:
                    continue

                iou = self.calculate_iou(
                    pred["box"],
                    gt["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if (
                best_gt is not None
                and best_iou >= self.iou_threshold
            ):

                pred["matched"] = True
                best_gt["matched"] = True

                matches.append({
                    "prediction": pred,
                    "ground_truth": best_gt,
                    "iou": best_iou
                })

        tp = len(matches)

        fp = sum(
            1
            for pred in pred_objects
            if not pred["matched"]
        )

        fn = sum(
            1
            for gt in gt_objects
            if not gt["matched"]
        )

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matches": matches,
            "ground_truths": gt_objects,
            "predictions": pred_objects
        }

    @staticmethod
    def calculate_precision(tp, fp):

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    @staticmethod
    def calculate_recall(tp, fn):

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)