
from src.evaluator import Evaluator


def test_iou_identical_boxes():

    box1 = (100, 100, 300, 300)
    box2 = (100, 100, 300, 300)

    iou = Evaluator.calculate_iou(box1, box2)

    print(f"IoU: {iou}")

    assert iou == 1.0

def test_iou_partial_overlap():

    box1 = (100, 100, 300, 300)
    box2 = (150, 150, 350, 350)

    iou = Evaluator.calculate_iou(box1, box2)

    print(f"IoU: {iou}")

    assert 0 < iou < 1
def test_perfect_match():

    evaluator = Evaluator(iou_threshold=0.5)

    ground_truths = [
        {
            "class_id": 0,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.4,
            "height": 0.4
        }
    ]

    predictions = [
        {
            "class_id": 0,
            "box": (300, 300, 700, 700),
            "confidence": 0.95
        }
    ]

    result = evaluator.evaluate_image(
        ground_truths,
        predictions,
        1000,
        1000
    )

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0        