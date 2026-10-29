
from pathlib import Path

import cv2
import numpy as np

from knn import knn
from face_utils import detect_faces, extract_face


DATA_DIR = Path("data")
K = 5
DISTANCE_THRESHOLD = 25.0


def load_dataset():
    samples = []
    labels = []
    names = {}

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "The data folder does not exist. "
            "Run collect_faces.py first."
        )

    for file_path in sorted(DATA_DIR.glob("*.npy")):
        data = np.load(file_path, allow_pickle=False)

        if data.ndim != 2 or data.shape[1] != 10000:
            print(f"Skipping incompatible dataset: {file_path.name}")
            continue

        if len(data) == 0:
            continue

        label_id = len(names)
        names[label_id] = file_path.stem

        samples.extend(data)
        labels.extend([label_id] * len(data))

        print(f"Loaded {len(data)} samples from {file_path.name}")

    if not samples:
        raise ValueError(
            "No valid training samples found. "
            "Run collect_faces.py first."
        )

    train_x = np.asarray(samples, dtype=np.float32)
    train_y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

    trainset = np.concatenate((train_x, train_y), axis=1)

    return trainset, names


def main():
    try:
        trainset, names = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            faces = detect_faces(frame)

            for box in faces:
                x, y, w, h = map(int, box)

                features = extract_face(frame, box)

                if features is None:
                    continue

                label_id = knn(
                    trainset,
                    features,
                    k=K,
                    threshold=DISTANCE_THRESHOLD
                )

                name = names.get(label_id, "Unknown")

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("KNN Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
