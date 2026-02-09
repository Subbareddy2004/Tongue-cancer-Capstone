import streamlit as st
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from transformers import SwinForImageClassification
import os
import gdown
import time
import numpy as np
from io import BytesIO
import base64
from datetime import datetime

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

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
    
    .quality-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .quality-good {
        background: #c8e6c9;
        color: #2e7d32;
    }
    
    .quality-warning {
        background: #ffe0b2;
        color: #f57c00;
    }
    
    .quality-poor {
        background: #ffcdd2;
        color: #c62828;
    }
    
    .gauge-container {
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, #4caf50 0%, #ffeb3b 50%, #f44336 100%);
        border-radius: 15px;
        position: relative;
        margin: 1rem 0;
    }
    
    .gauge-needle {
        position: absolute;
        width: 4px;
        height: 40px;
        background: #333;
        top: -5px;
        border-radius: 2px;
    }
    
    .confidence-bar {
        height: 25px;
        border-radius: 12px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .lesion-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# FEATURE 3: IMAGE QUALITY CHECKER
# =====================================================
def check_image_quality(image):
    """
    Check image quality including blur, brightness, and resolution.
    Returns a dictionary with quality metrics and recommendations.
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # 1. Check Resolution
    width, height = image.size
    min_resolution = 224  # Minimum for model input
    recommended_resolution = 400
    
    if width < min_resolution or height < min_resolution:
        resolution_status = "poor"
        resolution_message = f"Resolution too low ({width}×{height}). Minimum: {min_resolution}×{min_resolution}"
    elif width < recommended_resolution or height < recommended_resolution:
        resolution_status = "warning"
        resolution_message = f"Resolution acceptable ({width}×{height}). Recommended: {recommended_resolution}×{recommended_resolution}+"
    else:
        resolution_status = "good"
        resolution_message = f"Resolution is good ({width}×{height})"
    
    # 2. Check Brightness
    # Convert to grayscale for brightness analysis
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 50:
        brightness_status = "poor"
        brightness_message = "Image is too dark. Use better lighting."
    elif mean_brightness < 80:
        brightness_status = "warning"
        brightness_message = "Image is slightly dark. Consider better lighting."
    elif mean_brightness > 220:
        brightness_status = "poor"
        brightness_message = "Image is overexposed. Reduce lighting."
    elif mean_brightness > 200:
        brightness_status = "warning"
        brightness_message = "Image is slightly bright. Adjust lighting."
    else:
        brightness_status = "good"
        brightness_message = f"Brightness is good ({mean_brightness:.0f}/255)"
    
    # 3. Check Blur using Laplacian variance
    gray_pil = image.convert('L')
    # Apply Laplacian filter to detect edges
    laplacian = gray_pil.filter(ImageFilter.FIND_EDGES)
    laplacian_array = np.array(laplacian)
    variance = np.var(laplacian_array)
    
    if variance < 100:
        blur_status = "poor"
        blur_message = "Image is too blurry. Hold camera steady and ensure focus."
    elif variance < 300:
        blur_status = "warning"
        blur_message = "Image is slightly blurry. Try to improve focus."
    else:
        blur_status = "good"
        blur_message = f"Image sharpness is good (variance: {variance:.0f})"
    
    # 4. Overall quality assessment
    status_values = {"poor": 0, "warning": 1, "good": 2}
    overall_score = (status_values[resolution_status] + 
                     status_values[brightness_status] + 
                     status_values[blur_status]) / 3
    
    if overall_score < 0.5:
        overall_status = "poor"
        overall_message = "Image quality is poor. Please retake the photo."
    elif overall_score < 1.5:
        overall_status = "warning"
        overall_message = "Image quality is acceptable but could be improved."
    else:
        overall_status = "good"
        overall_message = "Image quality is good for analysis."
    
    return {
        "resolution": {"status": resolution_status, "message": resolution_message, "value": f"{width}×{height}"},
        "brightness": {"status": brightness_status, "message": brightness_message, "value": f"{mean_brightness:.0f}/255"},
        "blur": {"status": blur_status, "message": blur_message, "value": f"{variance:.0f}"},
        "overall": {"status": overall_status, "message": overall_message, "score": overall_score}
    }


def display_quality_results(quality_results):
    """Display image quality check results with visual indicators."""
    st.markdown("### 📊 Image Quality Analysis")
    
    # Create columns for each metric
    col1, col2, col3 = st.columns(3)
    
    status_colors = {
        "good": "#4caf50",
        "warning": "#ff9800",
        "poor": "#f44336"
    }
    
    status_icons = {
        "good": "✅",
        "warning": "⚠️",
        "poor": "❌"
    }
    
    with col1:
        status = quality_results["resolution"]["status"]
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; border: 2px solid {status_colors[status]};">
            <h4>📐 Resolution</h4>
            <p style="font-size: 1.5rem;">{status_icons[status]}</p>
            <p><strong>{quality_results["resolution"]["value"]}</strong></p>
            <p style="font-size: 0.85rem; color: #666;">{quality_results["resolution"]["message"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = quality_results["brightness"]["status"]
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; border: 2px solid {status_colors[status]};">
            <h4>💡 Brightness</h4>
            <p style="font-size: 1.5rem;">{status_icons[status]}</p>
            <p><strong>{quality_results["brightness"]["value"]}</strong></p>
            <p style="font-size: 0.85rem; color: #666;">{quality_results["brightness"]["message"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = quality_results["blur"]["status"]
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; border: 2px solid {status_colors[status]};">
            <h4>🔍 Sharpness</h4>
            <p style="font-size: 1.5rem;">{status_icons[status]}</p>
            <p><strong>Variance: {quality_results["blur"]["value"]}</strong></p>
            <p style="font-size: 0.85rem; color: #666;">{quality_results["blur"]["message"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall status
    overall = quality_results["overall"]
    status = overall["status"]
    
    if status == "poor":
        st.markdown(f"""
        <div class="error-box">
            <h4>{status_icons[status]} {overall["message"]}</h4>
            <p>For accurate analysis, please upload a clearer image with proper lighting and focus.</p>
        </div>
        """, unsafe_allow_html=True)
    elif status == "warning":
        st.markdown(f"""
        <div class="warning-box">
            <h4>{status_icons[status]} {overall["message"]}</h4>
            <p>You may proceed with analysis, but results might be more accurate with a better quality image.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
            <h4>{status_icons[status]} {overall["message"]}</h4>
            <p>The image meets quality requirements for accurate analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    return overall["status"] != "poor"


# =====================================================
# FEATURE 6: LESION SIZE ESTIMATION
# =====================================================
def estimate_lesion_size(image, prediction, confidence):
    """
    Estimate the approximate size of suspicious regions.
    Uses simple image analysis to identify potential lesion areas.
    """
    if prediction != "CANCER":
        return None
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Convert to HSV for better color segmentation
    # Lesions often appear as darker or redder areas
    
    # Simple approach: identify areas that deviate from normal tongue color
    # Normal tongue is typically pinkish
    
    # Calculate color statistics
    mean_r = np.mean(img_array[:, :, 0])
    mean_g = np.mean(img_array[:, :, 1])
    mean_b = np.mean(img_array[:, :, 2])
    
    # Identify potentially suspicious pixels (darker or abnormal color)
    # This is a simplified heuristic
    suspicious_mask = (
        (img_array[:, :, 0] < mean_r * 0.7) |  # Darker red
        (img_array[:, :, 1] < mean_g * 0.7) |  # Darker green
        ((img_array[:, :, 0] > mean_r * 1.3) & (img_array[:, :, 1] < mean_g * 0.8))  # Reddish areas
    )
    
    # Calculate percentage of suspicious area
    total_pixels = img_array.shape[0] * img_array.shape[1]
    suspicious_pixels = np.sum(suspicious_mask)
    suspicious_percentage = (suspicious_pixels / total_pixels) * 100
    
    # Estimate physical size (assuming average tongue dimensions)
    # Average tongue visible area is approximately 15-20 cm²
    avg_tongue_area_cm2 = 17.5
    estimated_area_cm2 = (suspicious_percentage / 100) * avg_tongue_area_cm2
    
    # Adjust based on confidence
    estimated_area_cm2 = estimated_area_cm2 * (confidence / 100)
    
    # Classification based on TNM staging for oral cancer
    if estimated_area_cm2 < 0.5:
        size_category = "Very Small"
        size_description = "Potentially early-stage lesion"
        stage_estimate = "T1 (≤2cm)"
    elif estimated_area_cm2 < 1.5:
        size_category = "Small"
        size_description = "Small suspicious area detected"
        stage_estimate = "T1 (≤2cm)"
    elif estimated_area_cm2 < 4:
        size_category = "Medium"
        size_description = "Moderate suspicious area"
        stage_estimate = "T2 (2-4cm)"
    else:
        size_category = "Large"
        size_description = "Significant suspicious area"
        stage_estimate = "T3+ (>4cm)"
    
    return {
        "percentage": suspicious_percentage,
        "estimated_area_cm2": estimated_area_cm2,
        "size_category": size_category,
        "size_description": size_description,
        "stage_estimate": stage_estimate,
        "suspicious_pixels": suspicious_pixels,
        "total_pixels": total_pixels
    }


def display_lesion_info(lesion_data):
    """Display lesion size estimation results."""
    if lesion_data is None:
        return
    
    st.markdown("### 📏 Lesion Size Estimation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Estimated Area",
            value=f"{lesion_data['estimated_area_cm2']:.2f} cm²",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Size Category",
            value=lesion_data['size_category'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Possible Stage",
            value=lesion_data['stage_estimate'],
            delta=None
        )
    
    st.markdown(f"""
    <div class="lesion-info">
        <h4>📋 Lesion Analysis Details</h4>
        <ul>
            <li><strong>Coverage:</strong> Approximately {lesion_data['percentage']:.1f}% of visible tongue area</li>
            <li><strong>Description:</strong> {lesion_data['size_description']}</li>
            <li><strong>Estimated Stage:</strong> {lesion_data['stage_estimate']} (based on size only)</li>
        </ul>
        <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
            <em>⚠️ Note: This is an AI-based estimation. Actual lesion size and staging can only be 
            determined through professional medical examination including biopsy and imaging.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# FEATURE 9: CONFIDENCE VISUALIZATION
# =====================================================
def display_confidence_visualization(confidence, prediction):
    """Display confidence using visual elements like progress bars and gauges."""
    st.markdown("### 📊 Confidence Visualization")
    
    # Determine colors based on prediction and confidence
    if prediction == "CANCER":
        if confidence >= 80:
            bar_color = "#d32f2f"  # Strong red
            risk_level = "High Confidence Detection"
        elif confidence >= 60:
            bar_color = "#f57c00"  # Orange
            risk_level = "Moderate Confidence Detection"
        else:
            bar_color = "#fbc02d"  # Yellow
            risk_level = "Low Confidence Detection"
    else:
        if confidence >= 80:
            bar_color = "#388e3c"  # Strong green
            risk_level = "High Confidence - Likely Normal"
        elif confidence >= 60:
            bar_color = "#689f38"  # Light green
            risk_level = "Moderate Confidence - Likely Normal"
        else:
            bar_color = "#fbc02d"  # Yellow
            risk_level = "Low Confidence - Uncertain"
    
    # Main confidence bar
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <p style="font-weight: bold; margin-bottom: 0.5rem;">Model Confidence: {confidence:.1f}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence}%; background: {bar_color};">
                {confidence:.1f}%
            </div>
        </div>
        <p style="text-align: center; color: #666; font-style: italic;">{risk_level}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dual probability bars (Cancer vs Non-Cancer)
    cancer_prob = confidence if prediction == "CANCER" else 100 - confidence
    non_cancer_prob = 100 - cancer_prob
    
    st.markdown("#### Probability Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #ffebee; border-radius: 10px;">
            <p style="margin: 0; font-weight: bold; color: #c62828;">Cancer Probability</p>
            <div style="background: #e0e0e0; border-radius: 10px; height: 20px; margin: 0.5rem 0;">
                <div style="background: linear-gradient(90deg, #ef5350, #c62828); width: {cancer_prob}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #c62828;">{cancer_prob:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #e8f5e9; border-radius: 10px;">
            <p style="margin: 0; font-weight: bold; color: #2e7d32;">Non-Cancer Probability</p>
            <div style="background: #e0e0e0; border-radius: 10px; height: 20px; margin: 0.5rem 0;">
                <div style="background: linear-gradient(90deg, #66bb6a, #2e7d32); width: {non_cancer_prob}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #2e7d32;">{non_cancer_prob:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence interpretation
    st.markdown("#### 📖 Confidence Interpretation Guide")
    
    confidence_guide = """
    | Confidence Level | Interpretation | Recommended Action |
    |-----------------|----------------|-------------------|
    | **90-100%** | Very High Confidence | Results are highly reliable |
    | **70-89%** | High Confidence | Results are reliable, follow recommendations |
    | **50-69%** | Moderate Confidence | Consider retaking image for verification |
    | **Below 50%** | Low Confidence | Image may be unclear, please retake |
    """
    st.markdown(confidence_guide)


# =====================================================
# FEATURE 5: PDF MEDICAL REPORT GENERATOR
# =====================================================
def generate_pdf_report(image, prediction, confidence, quality_results, lesion_data, filename):
    """Generate a detailed PDF medical report."""
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#667eea')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#764ba2')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("🏥 Oral Cancer Detection Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
    elements.append(Spacer(1, 20))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
    elements.append(Paragraph(f"<b>Image File:</b> {filename}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Save image to buffer for PDF
    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Add image to PDF (centered)
    img_width = 3 * inch
    img_height = 3 * inch
    rl_image = RLImage(img_buffer, width=img_width, height=img_height)
    elements.append(rl_image)
    elements.append(Spacer(1, 20))
    
    # Prediction Results
    elements.append(Paragraph("📋 Analysis Results", heading_style))
    
    result_color = colors.red if prediction == "CANCER" else colors.green
    result_text = "⚠️ CANCER DETECTED" if prediction == "CANCER" else "✅ NO CANCER DETECTED"
    
    result_data = [
        ["Prediction", result_text],
        ["Confidence", f"{confidence:.1f}%"],
        ["Analysis Date", report_date]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f7fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0'))
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 20))
    
    # Image Quality Results
    elements.append(Paragraph("📊 Image Quality Assessment", heading_style))
    
    quality_data = [
        ["Metric", "Status", "Details"],
        ["Resolution", quality_results["resolution"]["status"].upper(), quality_results["resolution"]["value"]],
        ["Brightness", quality_results["brightness"]["status"].upper(), quality_results["brightness"]["value"]],
        ["Sharpness", quality_results["blur"]["status"].upper(), f"Variance: {quality_results['blur']['value']}"],
        ["Overall", quality_results["overall"]["status"].upper(), quality_results["overall"]["message"]]
    ]
    
    quality_table = Table(quality_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0'))
    ]))
    elements.append(quality_table)
    elements.append(Spacer(1, 20))
    
    # Lesion Size Estimation (if applicable)
    if lesion_data:
        elements.append(Paragraph("📏 Lesion Size Estimation", heading_style))
        
        lesion_info = [
            ["Metric", "Value"],
            ["Estimated Area", f"{lesion_data['estimated_area_cm2']:.2f} cm²"],
            ["Size Category", lesion_data['size_category']],
            ["Coverage", f"{lesion_data['percentage']:.1f}% of visible area"],
            ["Possible Stage", lesion_data['stage_estimate']]
        ]
        
        lesion_table = Table(lesion_info, colWidths=[2.5*inch, 3.5*inch])
        lesion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196f3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0'))
        ]))
        elements.append(lesion_table)
        elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("📋 Recommendations", heading_style))
    
    if prediction == "CANCER":
        recommendations = [
            "🚨 <b>URGENT:</b> Please consult an oncologist immediately for proper medical evaluation.",
            "📋 Share this report with your healthcare provider.",
            "🔬 A biopsy may be required for definitive diagnosis.",
            "📅 Schedule follow-up appointments as recommended by your doctor.",
            "❌ Do not delay seeking professional medical advice."
        ]
    else:
        recommendations = [
            "✅ Continue regular oral health check-ups.",
            "🦷 Maintain good oral hygiene practices.",
            "🚭 Avoid tobacco and excessive alcohol consumption.",
            "📅 Schedule routine dental examinations.",
            "👀 Monitor for any changes in oral health and report to your doctor."
        ]
    
    for rec in recommendations:
        elements.append(Paragraph(rec, normal_style))
    
    elements.append(Spacer(1, 30))
    
    # Disclaimer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
    elements.append(Spacer(1, 10))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.gray,
        alignment=TA_CENTER
    )
    
    elements.append(Paragraph(
        "<b>⚠️ MEDICAL DISCLAIMER</b><br/>"
        "This report is generated by an AI-based screening tool for educational purposes only. "
        "It should NOT replace professional medical diagnosis or consultation with healthcare providers. "
        "Always consult with qualified medical professionals for proper diagnosis and treatment.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


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
    2. **Quality Check** analyzes image quality
    3. **AI Analysis** using Swin Transformer
    4. **Get Results** with confidence visualization
    5. **Download Report** as PDF
    6. **Consult** healthcare professional
    """)
    
    st.markdown("## 🆕 New Features")
    st.markdown("""
    - ✅ **Image Quality Checker**
    - ✅ **PDF Report Generator**
    - ✅ **Lesion Size Estimation**
    - ✅ **Confidence Visualization**
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
    
    # PDF availability status
    if PDF_AVAILABLE:
        st.markdown("""
        <div class="metric-card">
            <h4>📄 PDF Reports</h4>
            <p style="color: green; font-weight: bold;">✅ Available</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4>📄 PDF Reports</h4>
            <p style="color: orange; font-weight: bold;">⚠️ Install reportlab</p>
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
        # Method 1: Try with token first
        try:
            model = SwinForImageClassification.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                num_labels=2,
                ignore_mismatched_sizes=True,
                token=HUGGINGFACE_TOKEN
            )
        except Exception as token_error:
            st.warning(f"Token authentication failed: {str(token_error)}")
            st.info("🔄 Trying without token (public access)...")
            
            # Method 2: Try without token (public access)
            model = SwinForImageClassification.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        
        # Load your fine-tuned weights
        model.load_state_dict(torch.load(
            MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("🔧 Troubleshooting tips:")
        st.error("1. Generate a new Hugging Face token (your current one may be expired)")
        st.error("2. Check your internet connection")
        st.error("3. Try clearing Streamlit cache with Ctrl+Shift+R")
        st.error("4. The model file might be corrupted - try deleting it and re-downloading")
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
            st.image(image, caption="📸 Uploaded Image")
        
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
        
        # FEATURE 3: Image Quality Check
        st.markdown("---")
        quality_results = check_image_quality(image)
        quality_ok = display_quality_results(quality_results)
        
        # Store results in session state for PDF generation
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_disabled = not quality_ok
            if st.button("🔍 Analyze Image", use_container_width=True, disabled=analyze_disabled):
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
                        
                        # Store results for PDF
                        lesion_data = estimate_lesion_size(image, result, confidence)
                        st.session_state.analysis_results = {
                            'image': image,
                            'prediction': result,
                            'confidence': confidence,
                            'quality_results': quality_results,
                            'lesion_data': lesion_data,
                            'filename': uploaded_file.name
                        }
                        
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
                        
                        # FEATURE 9: Confidence Visualization
                        st.markdown("---")
                        display_confidence_visualization(confidence, result)
                        
                        # FEATURE 6: Lesion Size Estimation
                        if result == "CANCER":
                            st.markdown("---")
                            display_lesion_info(lesion_data)
                        
                        # Additional recommendations
                        st.markdown("---")
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
                        
                        # FEATURE 5: PDF Report Download
                        if PDF_AVAILABLE:
                            st.markdown("---")
                            st.markdown("### 📄 Download Medical Report")
                            
                            pdf_buffer = generate_pdf_report(
                                image, result, confidence, quality_results, 
                                lesion_data, uploaded_file.name
                            )
                            
                            if pdf_buffer:
                                report_filename = f"oral_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=report_filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                                st.success("✅ Report generated successfully! Click above to download.")
                        else:
                            st.markdown("---")
                            st.warning("📄 PDF Report generation requires the `reportlab` package. Install it with: `pip install reportlab`")
                        
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
        
        if analyze_disabled:
            st.warning("⚠️ Please upload a better quality image before analysis. The current image quality is too poor for reliable results.")
    
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