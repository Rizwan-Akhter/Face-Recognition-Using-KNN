
import streamlit as st
import cv2
import numpy as np

from knn import knn
from face_utils import detect_faces, get_largest_face


st.set_page_config(
    page_title="Face Recognition Using KNN",
    page_icon="👤",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "samples" not in st.session_state:
    st.session_state.samples = []

if "labels" not in st.session_state:
    st.session_state.labels = []

if "names" not in st.session_state:
    st.session_state.names = {}

if "next_id" not in st.session_state:
    st.session_state.next_id = 0


# --------------------------------------------------
# IMAGE CONVERSION
# --------------------------------------------------

def uploaded_to_bgr(uploaded_file):
    """Convert an uploaded image to an OpenCV BGR image."""
    if uploaded_file is None:
        return None

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    return image


def draw_faces(image):
    """Draw rectangles around all detected faces."""
    result = image.copy()
    faces = detect_faces(result)

    for x, y, w, h in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return result, faces


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Face Recognition Using OpenCV & KNN")

st.write(
    "Register people with their permission, "
    "then test face recognition using KNN."
)

st.info(
    "Demo privacy: registrations are held in your current "
    "Streamlit session and are not permanently saved by this app."
)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Recognition Settings")

k_value = st.sidebar.slider(
    "Number of neighbors (K)",
    min_value=1,
    max_value=9,
    value=5,
    step=2
)

threshold = st.sidebar.slider(
    "Distance threshold",
    min_value=5.0,
    max_value=60.0,
    value=25.0,
    step=1.0
)

st.sidebar.caption(
    "A lower threshold is stricter. Tune it using separate "
    "test images; this value is not a universal accuracy guarantee."
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

register_tab, recognize_tab, info_tab = st.tabs(
    ["Register Person", "Recognize Face", "Project Info"]
)


# --------------------------------------------------
# REGISTER A PERSON
# --------------------------------------------------

with register_tab:
    st.subheader("Register a person")

    person_name = st.text_input(
        "Person's name",
        placeholder="Enter a name"
    )

    st.write(
        "Capture several images with different lighting, "
        "angles and expressions for better results."
    )

    camera_image = st.camera_input(
        "Capture a registration image",
        key="register_camera"
    )

    if st.button(
        "Add face sample",
        key="add_sample"
    ):
        if not person_name.strip():
            st.warning("Please enter a person's name.")

        elif camera_image is None:
            st.warning("Please capture an image first.")

        else:
            frame = uploaded_to_bgr(camera_image)

            if frame is None:
                st.error("Could not read the image.")

            else:
                features, box = get_largest_face(frame)

                if features is None:
                    st.warning(
                        "No face detected. Try again with "
                        "better lighting and a clear frontal face."
                    )

                else:
                    clean_name = person_name.strip()

                    # Reuse the same ID for the same name.
                    existing_id = next(
                        (
                            label
                            for label, name
                            in st.session_state.names.items()
                            if name.casefold()
                            == clean_name.casefold()
                        ),
                        None
                    )

                    if existing_id is None:
                        label_id = st.session_state.next_id
                        st.session_state.next_id += 1
                        st.session_state.names[label_id] = clean_name
                    else:
                        label_id = existing_id

                    st.session_state.samples.append(features)
                    st.session_state.labels.append(label_id)

                    st.success(
                        f"Face sample added for {clean_name}. "
                        f"Total samples: {len(st.session_state.samples)}"
                    )

    st.divider()

    st.subheader("Registered people")

    if st.session_state.names:
        for label_id, name in st.session_state.names.items():
            count = sum(
                label == label_id
                for label in st.session_state.labels
            )

            st.write(f"**{name}** — {count} sample(s)")
    else:
        st.write("No people registered yet.")

    if st.button(
        "Clear all registrations",
        key="clear_registrations"
    ):
        st.session_state.samples = []
        st.session_state.labels = []
        st.session_state.names = {}
        st.session_state.next_id = 0
        st.rerun()


# --------------------------------------------------
# RECOGNIZE A FACE
# --------------------------------------------------

with recognize_tab:
    st.subheader("Recognize a face")

    recognition_source = st.radio(
        "Choose image source",
        ["Camera", "Upload image"],
        horizontal=True
    )

    if recognition_source == "Camera":
        recognition_image = st.camera_input(
            "Capture a face for recognition",
            key="recognition_camera"
        )
    else:
        recognition_image = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
            key="recognition_upload"
        )

    if st.button(
        "Recognize face",
        key="recognize_button"
    ):
        if not st.session_state.samples:
            st.warning(
                "Register at least one person and add face samples first."
            )

        elif recognition_image is None:
            st.warning("Capture or upload an image first.")

        else:
            frame = uploaded_to_bgr(recognition_image)

            if frame is None:
                st.error("Could not read the image.")

            else:
                features, box = get_largest_face(frame)

                if features is None:
                    st.warning("No face detected in the image.")

                else:
                    samples = np.asarray(
                        st.session_state.samples,
                        dtype=np.float32
                    )

                    labels = np.asarray(
                        st.session_state.labels,
                        dtype=np.float32
                    ).reshape(-1, 1)

                    trainset = np.concatenate(
                        (samples, labels),
                        axis=1
                    )

                    prediction = knn(
                        trainset,
                        features,
                        k=k_value,
                        threshold=threshold
                    )

                    x, y, w, h = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    if prediction == -1:
                        display_name = "Unknown"
                        st.warning(
                            "No sufficiently close match was found. "
                            "This does not prove the person is unregistered."
                        )
                    else:
                        display_name = st.session_state.names.get(
                            prediction, "Unknown"
                        )

                        st.success(
                            f"Predicted person: {display_name}"
                        )

                    cv2.putText(
                        frame,
                        display_name,
                        (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    st.image(
                        cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        ),
                        caption="Recognition result",
                        use_container_width=True
                    )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with info_tab:
    st.subheader("About this project")

    st.markdown(
        """
        **Technologies**
        - Python
        - OpenCV
        - Haar Cascade face detection
        - NumPy
        - K-Nearest Neighbors (KNN)
        - Streamlit

        **Workflow**
        1. Capture and register face samples.
        2. Convert detected faces into 100 × 100 grayscale features.
        3. Calculate Euclidean distances using KNN.
        4. Display the predicted label or Unknown.

        **Limitations**
        - This is a learning demonstration, not a secure
          authentication system.
        - Recognition accuracy depends on lighting, image quality,
          sample diversity and threshold selection.
        - The app does not implement liveness detection.
        """
    )
