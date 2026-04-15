import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionAI Object Detector", page_icon="🔍", layout="wide")

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_base_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # This will automatically download yolov8n.pt (6MB) on the first run
    model = YOLO('yolov8n.pt') 
    return model

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("This app uses a **YOLOv8 nano** model running locally on the server. No API keys required.")
    
    confidence = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25)
    st.divider()
    st.markdown("""
    ### How to use:
    1. Upload a JPG/PNG image.
    2. The model will automatically detect objects.
    3. View the annotated image and detection details below.
    """)

# --- MAIN INTERFACE ---
st.title("🔍 VisionAI: Instant Object Detection")
st.write("Upload an image to identify objects in real-time using Computer Vision.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner('🤖 AI is thinking...'):
        # Run Inference
        results = model.predict(source=img_array, conf=confidence)
        
        # Plot results on the image
        # results[0].plot() returns a BGR numpy array
        res_plotted = results[0].plot()
        # Convert BGR to RGB for Streamlit
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Detection Results")
        st.image(res_plotted_rgb, use_container_width=True)

    # --- DATA TABLE ---
    st.divider()
    st.subheader("📋 Detection Details")
    
    detections = results[0].boxes
    if len(detections) > 0:
        # Prepare data for a clean table
        obj_data = []
        for box in detections:
            obj_data.append({
                "Object": model.names[int(box.cls)],
                "Confidence": f"{float(box.conf):.2%}",
                "Coordinates": f"{[round(float(x), 2) for x in box.xyxy[0].tolist()]}"
            })
        st.table(obj_data)
    else:
        st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")

else:
    st.info("👆 Please upload an image to start detection.")
    # Example placeholder visual
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&q=80&w=1000", 
             caption="Example Landscape", use_container_width=True, alpha=0.3)
