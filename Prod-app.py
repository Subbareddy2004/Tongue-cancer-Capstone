import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import SwinForImageClassification
import os
import gdown

# Setup
st.set_page_config(page_title="Oral Cancer Detection", page_icon="🧠")
st.title("🧠 Oral Cancer Detection from Tongue Images")

# Download model from Google Drive if not present
MODEL_PATH = "oral_cancer_swin_new.pth"
file_id = "1CIDpp_rZYc1us5H6iXE9r4WDOQBLXm9f"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model from Google Drive..."):
        gdown.download(gdrive_url, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_model():
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
class_names = ["CANCER", "NON CANCER"]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Upload image
uploaded_file = st.file_uploader("📤 Upload a tongue image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output.logits, dim=1).item()
                prob = torch.softmax(output.logits, dim=1)[0][pred].item()

            result = class_names[pred]
            color = "red" if result == "CANCER" else "green"
            emoji = "⚠️" if result == "CANCER" else "✅"

            st.markdown(f"<h3 style='text-align:center; color:{color}'>{emoji} Prediction: {result}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center'>Confidence: <b>{prob*100:.2f}%</b></p>", unsafe_allow_html=True)
