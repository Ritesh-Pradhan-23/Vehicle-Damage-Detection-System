import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("car_detaction_RS5.h5")  # ✅ matches notebook
    return model

model = load_model()

st.sidebar.title("🚀 Project Information")
st.sidebar.write("**Model:** CNN Vehicle Damage Detection")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.write("Upload a vehicle image to detect damage.")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.title("🚗 Vehicle Damage Detection System")
st.markdown("### Upload a vehicle image and let the AI detect damage")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((200, 200))          # ✅ matches image_size = 200 in notebook
    img_array = np.array(img)
    img_array = img_array[:, :, ::-1]       # ✅ RGB → BGR to match cv2.imread training
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image... Please wait"):
            time.sleep(1)
            prediction = model.predict(img_array)
            confidence = float(np.max(prediction))
            predicted_class = np.argmax(prediction)

        with col2:
            st.subheader("📊 Prediction Result")

            if confidence >= confidence_threshold:
                if predicted_class == 1:   # 1 = Damaged (cate index 1)
                    st.error("🚨 Damaged Vehicle Detected")
                else:                      # 0 = Undamaged (cate index 0)
                    st.success("✅ No Damage Detected")
            else:
                st.warning("⚠️ Low confidence prediction")

            st.write(f"**Confidence Score:** {confidence:.2f}")
            st.progress(confidence)

st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit | CNN Deep Learning Project</center>",
    unsafe_allow_html=True
)