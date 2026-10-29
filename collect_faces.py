
import re
from pathlib import Path

import cv2
import numpy as np

from face_utils import detect_faces, extract_face


DATA_DIR = Path("data")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    name = input("Enter person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    # Create a safe filename without losing the displayed name.
    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        name
    ).strip("_")

    if not safe_name:
        print("Please enter a name containing letters or numbers.")
        return

    file_path = DATA_DIR / f"{safe_name}.npy"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    existing = []

    if file_path.exists():
        existing = list(
            np.load(file_path, allow_pickle=False)
        )

    print("Look at the camera and vary your head angle.")
    print("Press S to save a face sample, Q to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            faces = detect_faces(frame)

            for x, y, w, h in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Samples: {len(existing)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect Face Samples", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                if len(faces) == 0:
                    print("No face detected. Try again.")
                    continue

                # Capture the largest face.
                face_box = max(
                    faces,
                    key=lambda box: int(box[2]) * int(box[3])
                )

                features = extract_face(frame, face_box)

                if features is not None:
                    existing.append(features)

                    np.save(
                        file_path,
                        np.asarray(existing, dtype=np.float32)
                    )

                    print(
                        f"Saved sample {len(existing)} "
                        f"for {name}"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved dataset: {file_path}")


if __name__ == "__main__":
    main()
