
import cv2
import numpy as np


FACE_SIZE = (100, 100)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifier.")


def detect_faces(image):
    """Return detected face coordinates from a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def extract_face(image, box):
    """
    Extract a face, convert it to grayscale,
    resize to 100x100 and normalize pixels.
    """
    x, y, w, h = map(int, box)

    height, width = image.shape[:2]
    offset = 10

    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(width, x + w + offset)
    y2 = min(height, y + h + offset)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    gray_face = cv2.cvtColor(
        face, cv2.COLOR_BGR2GRAY
    )

    gray_face = cv2.resize(
        gray_face, FACE_SIZE
    )

    # Histogram equalization can improve contrast.
    gray_face = cv2.equalizeHist(gray_face)

    # Normalize pixels to [0, 1].
    normalized = gray_face.astype(
        np.float32
    ) / 255.0

    return normalized.flatten()


def get_largest_face(image):
    """Return the feature vector and bounding box of the largest face."""
    faces = detect_faces(image)

    if len(faces) == 0:
        return None, None

    largest = max(
        faces,
        key=lambda box: int(box[2]) * int(box[3])
    )

    features = extract_face(image, largest)

    return features, largest
