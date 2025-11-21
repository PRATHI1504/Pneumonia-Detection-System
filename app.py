import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from src.model_utils import get_model, GradCAM, overlay_heatmap

# --- Configuration ---
st.set_page_config(page_title="Pneumonia Detector", page_icon="", layout="wide")
MODEL_PATH = ".streamlit/pneumonia_model.pth"
CLASSES = ['NORMAL', 'PNEUMONIA'] # Matches folder names
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model (Cached for Speed) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    model = get_model(num_classes=2)
    # Load weights
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# --- Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- User Interface ---
st.title("AI-Powered Pneumonia Detection System")
st.markdown("Upload a Chest X-Ray (DICOM/JPEG/PNG) to detect signs of Pneumonia.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an X-Ray...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Patient X-Ray", use_container_width=True)

with col2:
    st.subheader("2. AI Analysis")
    
    if uploaded_file:
        model = load_model()
        
        if not model:
            st.error(f"❌ Model file '{MODEL_PATH}' not found!")
            st.info("👉 Please run 'python train_model.py' in your terminal first.")
        else:
            # Make Prediction
            with st.spinner("Analyzing lung opacity patterns..."):
                input_tensor = preprocess_image(image)
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, preds = torch.max(probs, 1)
                
                label = CLASSES[preds.item()]
                score = confidence.item()

            # Display Result
            if label == "PNEUMONIA":
                st.error(f"### Result: {label}")
            else:
                st.success(f"### Result: {label}")
                
            st.progress(score)
            st.caption(f"Confidence: {score:.2%}")
            
            # Explainability (Grad-CAM)
            st.divider()
            st.subheader("3. Visual Evidence (Grad-CAM)")
            st.write("The heatmap shows *where* the model is looking:")
            
            try:
                # We target 'layer4' which is the last conv block in ResNet
                grad_cam = GradCAM(model, model.layer4)
                heatmap = grad_cam(input_tensor)
                
                # Overlay
                heatmap_img = overlay_heatmap(image.resize((224, 224)), heatmap)
                st.image(heatmap_img, caption="Attention Heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate heatmap: {e}")

# Sidebar Info
st.sidebar.info("System Status: Online")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("Model Loaded")
else:
    st.sidebar.warning("Model Not Trained")