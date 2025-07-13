import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import SwinForImageClassification
import os
import gdown
import time

# Set Hugging Face token - ADD THIS AT THE TOP
# You can get your token from https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token
os.environ['HUGGINGFACE_HUB_TOKEN'] = HUGGINGFACE_TOKEN

# Page configuration
st.set_page_config(
    page_title="Oral Cancer Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #f44336;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .prediction-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Oral Cancer Detection System</h1>
    <p style="font-size: 1.2rem; margin: 0;">AI-Powered Early Detection Using Tongue Image Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## 📋 About This System")
    st.markdown("""
    <div class="sidebar-info">
        <p><strong>🎯 Purpose:</strong> Early detection of oral cancer from tongue images using advanced AI technology.</p>
        <p><strong>🧠 Model:</strong> Swin Transformer - State-of-the-art vision model</p>
        <p><strong>📊 Accuracy:</strong> Trained on medical imaging datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔍 How It Works")
    st.markdown("""
    1. **Upload** a clear tongue image
    2. **AI Analysis** using Swin Transformer
    3. **Get Results** with confidence score
    4. **Consult** healthcare professional
    """)
    
    st.markdown("## 📞 Emergency Contacts")
    st.markdown("""
    - **Cancer Helpline:** 1-800-CANCER
    - **Local Emergency:** 911
    - **Consultation:** Contact your doctor
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Important Notice
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Important Medical Disclaimer</h3>
        <p style="line-height: 1.6; margin-bottom: 0;"><strong>This tool is for educational and screening purposes only.</strong> 
        It should not replace professional medical diagnosis or consultation with healthcare providers. 
        Always consult with qualified medical professionals for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-bottom: 1rem;">📝 Instructions for Best Results</h3>
        <div style="line-height: 1.8;">
            <p><strong>🔸 Image Quality:</strong> Use clear, well-lit photographs</p>
            <p><strong>🔸 Tongue Position:</strong> Extend tongue fully, keep it straight</p>
            <p><strong>🔸 Lighting:</strong> Good natural or white lighting preferred</p>
            <p><strong>🔸 Background:</strong> Plain background works best</p>
            <p><strong>🔸 Format:</strong> JPG, JPEG, or PNG files supported</p>
            <p style="margin-bottom: 0;"><strong>🔸 Size:</strong> High resolution images (but not too large)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Model status
    st.markdown("""
    <div class="metric-card">
        <h4>🤖 AI Model Status</h4>
        <p style="color: green; font-weight: bold;">✅ Ready for Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics (you can update these with real data)
    st.markdown("""
    <div class="metric-card">
        <h4>📊 System Statistics</h4>
        <p><strong>Model:</strong> Swin Transformer</p>
        <p><strong>Classes:</strong> 2 (Cancer/Non-Cancer)</p>
        <p><strong>Image Size:</strong> 224x224</p>
    </div>
    """, unsafe_allow_html=True)

# Model Setup
MODEL_PATH = "oral_cancer_swin_new.pth"
file_id = "1CIDpp_rZYc1us5H6iXE9r4WDOQBLXm9f"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading AI model... Please wait."):
        try:
            gdown.download(gdrive_url, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")

# Load model with better error handling
@st.cache_resource
def load_model():
    try:
        # Try to load with token authentication (updated parameter name)
        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            num_labels=2,
            ignore_mismatched_sizes=True,
            token=HUGGINGFACE_TOKEN  # Updated parameter name from use_auth_token to token
        )
        model.load_state_dict(torch.load(
            MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("🔧 Troubleshooting tips:")
        st.error("1. Make sure your Hugging Face token is valid")
        st.error("2. Check your internet connection")
        st.error("3. The model file might be corrupted - try deleting it and re-downloading")
        return None

# Load the model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Failed to load the model. Please check your Hugging Face token and try again.")
    st.markdown("""
    <div class="error-box">
        <h4>🔧 How to fix this:</h4>
        <ol>
            <li>Go to <a href="https://huggingface.co/settings/tokens" target="_blank">Hugging Face Tokens</a></li>
            <li>Create a new token (Read access is sufficient)</li>
            <li>Replace <code>your_token_here</code> in the code with your actual token</li>
            <li>Restart the application</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

class_names = ["CANCER", "NON CANCER"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Upload section
st.markdown("""
<div class="upload-section">
    <h2 style="text-align: center; margin-bottom: 1rem;">📤 Upload Tongue Image for Analysis</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the tongue for analysis"
)

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        
        # Image info
        st.markdown(f"""
        <div class="info-box">
            <p><strong>📝 Image Details:</strong></p>
            <ul>
                <li><strong>Filename:</strong> {uploaded_file.name}</li>
                <li><strong>Size:</strong> {image.size[0]} × {image.size[1]} pixels</li>
                <li><strong>Format:</strong> {image.format}</li>
                <li><strong>File Size:</strong> {uploaded_file.size / 1024:.1f} KB</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the image... Please wait."):
                    # Add a progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Preprocess image
                        img_tensor = transform(image).unsqueeze(0)
                        
                        # Make prediction
                        with torch.no_grad():
                            output = model(img_tensor)
                            pred = torch.argmax(output.logits, dim=1).item()
                            prob = torch.softmax(output.logits, dim=1)[0][pred].item()
                        
                        # Display results
                        result = class_names[pred]
                        confidence = prob * 100
                        
                        # Results container
                        st.markdown("""
                        <div class="prediction-container">
                        """, unsafe_allow_html=True)
                        
                        if result == "CANCER":
                            st.markdown(f"""
                            <div class="error-box">
                                <h2 style="color: #d32f2f; text-align: center;">⚠️ CANCER DETECTED</h2>
                                <p style="text-align: center; font-size: 1.3rem;">
                                    <strong>Confidence: {confidence:.1f}%</strong>
                                </p>
                                <p style="text-align: center; margin-top: 1rem;">
                                    <strong>🚨 URGENT: Please consult an oncologist immediately for proper medical evaluation.</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box">
                                <h2 style="color: #2e7d32; text-align: center;">✅ NO CANCER DETECTED</h2>
                                <p style="text-align: center; font-size: 1.3rem;">
                                    <strong>Confidence: {confidence:.1f}%</strong>
                                </p>
                                <p style="text-align: center; margin-top: 1rem;">
                                    <strong>Continue regular oral health check-ups for prevention.</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Additional recommendations
                        st.markdown("""
                        <div class="info-box">
                            <h3>📋 Next Steps & Recommendations</h3>
                            <ul>
                                <li><strong>Medical Consultation:</strong> Share these results with your healthcare provider</li>
                                <li><strong>Regular Check-ups:</strong> Schedule routine oral examinations</li>
                                <li><strong>Lifestyle:</strong> Maintain good oral hygiene and avoid tobacco/alcohol</li>
                                <li><strong>Follow-up:</strong> Monitor any changes in oral health</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if confidence < 70:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Low Confidence Warning</h4>
                                <p>The AI model has low confidence in this prediction. Please ensure the image quality is good and consider retaking the photo with better lighting and clarity.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.markdown("""
                        <div class="error-box">
                            <p>An error occurred during analysis. Please try again with a different image or contact support.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>🎓 Final Year Project - Oral Cancer Detection System</strong></p>
    <p>Built with ❤️ using Streamlit and PyTorch | AI-Powered Medical Imaging</p>
    <p><em>Remember: This tool is for educational purposes and should not replace professional medical advice.</em></p>
</div>
""", unsafe_allow_html=True)