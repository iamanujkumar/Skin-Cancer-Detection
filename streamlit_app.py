import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Dropout, Flatten, Dense

IMG_HEIGHT = 180
IMG_WIDTH = 180
NUM_CLASSES = 9

CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion',
]


def build_model() -> tf.keras.Model:
    model = Sequential([Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))])

    model.add(Conv2D(32, 3, padding="same", activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(128, 3, padding="same", activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.15))

    model.add(Conv2D(256, 3, padding="same", activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.20))

    model.add(Conv2D(512, 3, padding="same", activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> tf.keras.Model:
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_path)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(image, dtype=np.float32)
    x = x[None, ...] 
    return x


def main() -> None:
    st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")
    st.title("Skin Lesion Classifier")
    st.caption("Upload a dermatoscopic image to classify into 9 categories.")

    weights_path = "./cnn_fc_model.weights.h5"
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error("Could not load model weights. Make sure 'cnn_fc_model.weights.h5' is in the app folder.")
        st.exception(e)
        return

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]) 
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Input image", use_column_width=True)

        with st.spinner("Predicting..."):
            x = preprocess_image(image)
            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        st.subheader("Prediction")
        st.write(f"Label: **{pred_label}**")
        st.write(f"Confidence: **{confidence:.4f}**")

        top_k = 3
        top_idx = np.argsort(probs)[-top_k:][::-1]
        st.write("Top classes:")
        for i in top_idx:
            st.write(f"- {CLASS_NAMES[i]}: {probs[i]:.4f}")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})


if __name__ == "__main__":
    main()


