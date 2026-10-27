
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pneumonia_cnn.keras"

IMAGE_SIZE = (150, 150)
THRESHOLD = 0.5

# ---------------- DARK GREEN THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #071A13;
    color: #E5F5EC;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: #071A13;
}

[data-testid="stAppViewContainer"] {
    background: #071A13;
}

h1, h2, h3 {
    color: #D1FAE5 !important;
}

p, label {
    color: #D5E7DC;
}

.subtitle {
    color: #A7C9B5;
    font-size: 15px;
    margin-bottom: 22px;
    line-height: 1.7;
}

.notice {
    background: #123D2B;
    color: #BBF7D0;
    border: 1px solid #246747;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 15px 0 24px;
    line-height: 1.7;
}

.card {
    background: #10291F;
    border: 1px solid #24513B;
    border-radius: 15px;
    padding: 17px;
    margin: 8px 0 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}

.card small {
    color: #A7C9B5;
    font-size: 11px;
    letter-spacing: 1px;
    font-weight: 600;
}

.card h3 {
    color: #4ADE80 !important;
    margin: 8px 0 0;
    font-size: 23px;
}

[data-testid="stFileUploader"] {
    background: #10291F;
    border: 1px dashed #3F8F62;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stFileUploader"] * {
    color: #E5F5EC;
}

[data-testid="stFileUploaderDropzone"] {
    background: #10291F;
    border-color: #3F8F62;
}

div.stButton > button {
    background: #15803D;
    color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #22A052;
    font-weight: 700;
    min-height: 45px;
    padding: 10px 18px;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: #166534;
    color: #FFFFFF;
    border-color: #4ADE80;
}

.result-normal {
    background: #123D2B;
    border: 1px solid #2F855A;
    border-radius: 14px;
    padding: 20px;
    margin: 18px 0;
}

.result-normal h3 {
    color: #86EFAC !important;
}

.result-pneumonia {
    background: #38291B;
    border: 1px solid #A16207;
    border-radius: 14px;
    padding: 20px;
    margin: 18px 0;
}

.result-pneumonia h3 {
    color: #FCD34D !important;
}

.result-description {
    color: #D5E7DC;
    font-size: 14px;
    line-height: 1.7;
}

[data-testid="stProgressBar"] > div {
    background: #244534;
}

[data-testid="stProgressBar"] [data-testid="stProgressBarFill"] {
    background: #22C55E;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.stCaption, small {
    color: #A7C9B5;
}

hr {
    border-color: #24513B;
    margin-top: 28px;
}

.footer {
    text-align: center;
    color: #8EAD9B;
    font-size: 12px;
    padding: 12px 0 20px;
    line-height: 1.8;
}

@media (max-width: 640px) {
    .card {
        padding: 12px;
    }

    .card h3 {
        font-size: 19px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


# ---------------- HEADER ----------------
st.title("🩻 Pneumonia Detection from Chest X-rays")

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Mini Project using a '
    'Convolutional Neural Network (CNN)'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="notice">'
    '<b>Educational demonstration only.</b> '
    'This tool is not clinically validated and must not be used '
    'to diagnose pneumonia or make medical decisions.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- MODEL INFORMATION CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <small>MODEL TYPE</small>
        <h3>CNN</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <small>IMAGE SIZE</small>
        <h3>150 × 150</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <small>TEST ACCURACY</small>
        <h3>73.24%</h3>
    </div>
    """, unsafe_allow_html=True)


# ---------------- IMAGE UPLOAD ----------------
st.subheader("Upload a Chest X-ray Image")

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Select an X-ray image to run the model prediction."
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Identify the current upload so old results are not reused.
        signature = (uploaded_file.name, uploaded_file.size)

        if st.session_state.get("image_signature") != signature:
            st.session_state["image_signature"] = signature
            st.session_state.pop("prediction_result", None)

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

        if st.button("Analyze Image", use_container_width=True):

            if not MODEL_PATH.exists():
                st.error(
                    "Model file pneumonia_cnn.keras was not found. "
                    "Place it in the same folder as app.py."
                )
            else:
                try:
                    with st.spinner("Analyzing image..."):

                        model = load_model()

                        resized_image = image.resize(IMAGE_SIZE)

                        image_array = np.asarray(
                            resized_image,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # The model already contains Rescaling(1/255).
                        prediction_score = float(
                            model.predict(
                                image_array,
                                verbose=0
                            )[0][0]
                        )

                        predicted_class = (
                            "PNEUMONIA"
                            if prediction_score >= THRESHOLD
                            else "NORMAL"
                        )

                        st.session_state["prediction_result"] = {
                            "class": predicted_class,
                            "score": prediction_score
                        }

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        # ---------------- DISPLAY RESULT ----------------
        result = st.session_state.get("prediction_result")

        if (
            result is not None
            and st.session_state.get("image_signature") == signature
        ):

            predicted_class = result["class"]
            prediction_score = result["score"]

            if predicted_class == "PNEUMONIA":

                st.markdown(
                    """
                    <div class="result-pneumonia">
                        <h3>🟠 Model Prediction: Pneumonia</h3>
                        <div class="result-description">
                            The model output met or exceeded the
                            configured classification threshold.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                class_score = prediction_score

            else:

                st.markdown(
                    """
                    <div class="result-normal">
                        <h3>🟢 Model Prediction: Normal</h3>
                        <div class="result-description">
                            The model output was below the
                            configured classification threshold.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                class_score = 1.0 - prediction_score

            st.write("**Model score for predicted class**")

            st.progress(
                max(0.0, min(1.0, class_score))
            )

            st.write(f"**{class_score:.2%}**")

            st.caption(
                f"Raw Pneumonia sigmoid output: "
                f"{prediction_score:.4f} | "
                f"Classification threshold: {THRESHOLD:.2f}"
            )

            st.info(
                "This score is not a calibrated probability of disease. "
                "The model cannot confirm or rule out pneumonia."
            )

    except Exception as error:
        st.error(
            f"Could not process the uploaded image: {error}"
        )


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <b style="color:#4ADE80;">PneumoAI</b><br>
        Built with Python, TensorFlow and Streamlit<br>
        Deep Learning Mini Project · Not for medical diagnosis
    </div>
    """,
    unsafe_allow_html=True
)
