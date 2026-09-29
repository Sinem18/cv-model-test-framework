
from pathlib import Path


class DatasetManager:

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def get_images(self, split: str = "val"):
        """
        Dataset içerisindeki görüntüleri döndürür.
        """

        image_dir = self.dataset_path / "images" / split

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        extensions = {".jpg", ".jpeg", ".png"}

        images = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in extensions
        ]

        return sorted(images)

    def get_label_path(self, image_path: Path):
        """
        Bir görüntünün karşılık gelen YOLO label dosyasını bulur.
        """

        label_path = (
            self.dataset_path
            / "labels"
            / image_path.parent.name
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        return label_path
    def get_labels(self, image_path: Path):
        """
        Bir görüntünün YOLO formatındaki annotationlarını okur.
        """

        label_path = self.get_label_path(image_path)

        labels = []

        with open(label_path, "r") as file:
            for line in file:
                values = line.strip().split()

                if len(values) != 5:
                    raise ValueError(
                        f"Invalid label format: {label_path}"
                    )

                class_id = int(values[0])
                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

                labels.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                })

        return labels