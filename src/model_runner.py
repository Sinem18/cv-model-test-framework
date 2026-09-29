from ultralytics import YOLO


class ModelRunner:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_path: str):

        results = self.model.predict(
            source=image_path,
            verbose=False
        )

        result = results[0]

        predictions = []

        for box in result.boxes:

            xyxy = box.xyxy[0].tolist()

            class_id = int(box.cls[0].item())

            confidence = float(
                box.conf[0].item()
            )

            predictions.append({
                "class_id": class_id,
                "box": tuple(xyxy),
                "confidence": confidence
            })

        return predictions