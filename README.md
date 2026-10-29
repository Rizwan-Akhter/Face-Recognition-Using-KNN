
# Face Recognition Using OpenCV and KNN

A face detection and recognition project built using Python, OpenCV,
Haar Cascade classifiers, NumPy, and the K-Nearest Neighbors (KNN)
algorithm.

## Features

- Face detection using Haar Cascade.
- Face preprocessing and feature extraction.
- Face registration using webcam images.
- KNN-based face recognition.
- Unknown-face rejection using a distance threshold.
- Desktop webcam application.
- Streamlit web interface.
- Image upload and camera capture for recognition.

## Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit
- Haar Cascade
- K-Nearest Neighbors (KNN)

## Project Structure

    Face-Recognition-Using-KNN/
    ├── app.py
    ├── knn.py
    ├── face_utils.py
    ├── face_detection.py
    ├── collect_faces.py
    ├── webcam_recognition.py
    ├── requirements.txt
    ├── .gitignore
    ├── README.md
    └── data/

## Installation

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/Face-Recognition-Using-KNN.git

Move into the project directory:

    cd Face-Recognition-Using-KNN

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install dependencies for the web application:

    pip install -r requirements.txt

## Run the Web Application

    streamlit run app.py

Open the local URL displayed in your terminal.

Register a person, capture face samples, and test recognition.

## Run Desktop Face Detection

For desktop scripts, install the GUI-enabled OpenCV package:

    pip uninstall opencv-python-headless -y
    pip install opencv-python

Run:

    python face_detection.py

## Collect Face Samples

    python collect_faces.py

Enter a person's name, press S to save face samples,
and press Q to exit.

Repeat for other people.

## Run Desktop Face Recognition

    python webcam_recognition.py

## How It Works

1. Haar Cascade detects faces in an image.
2. The largest detected face is extracted.
3. The face is converted to grayscale and resized to 100 × 100.
4. Pixel values are normalized and flattened into a feature vector.
5. KNN calculates Euclidean distances between training and test samples.
6. The nearest neighbors vote for a predicted identity.
7. A distance threshold can reject a poor match as Unknown.

## Deployment

The Streamlit application can be deployed through
Streamlit Community Cloud using this repository and app.py
as the entrypoint.

## Privacy and Limitations

- Collect face samples only with the person's consent.
- Face samples are biometric data and should be protected.
- The web application keeps registrations in session memory.
- Desktop training files are excluded from Git by default.
- This project is intended for learning and demonstration,
  not high-security identity verification.
- Accuracy depends on training samples, lighting, camera quality,
  and threshold selection.
