## Overview

This is a comprehensive capstone project for **Oral (Tongue) Cancer Detection** using deep learning. The project utilizes a Swin Transformer model trained on oral cancer imagery to detect and classify cancer lesions from medical images.

## 🎯 Project Objectives

- Develop an automated system to detect oral cancer from images
- Provide early detection capabilities to support medical professionals
- Deploy the model across multiple platforms for accessibility
- Offer both local and cloud-based solutions

## 📁 Project Structure

### Core Applications

- **`Local-app.py`** - Local Streamlit application for testing and development
- **`Prod-app.py`** - Production Streamlit application with enhanced features
- **`requirements.txt`** - Root project dependencies (Streamlit, PyTorch, Transformers, etc.)

### Deployment Options

#### 1. **FastAPI Backend** (`oral-cancer-api/`)

- RESTful API built with FastAPI
- Efficient model serving with Uvicorn
- **Dependencies**: FastAPI, Uvicorn, PyTorch, Transformers, Pillow, Torchvision
- **Usage**: Run as a backend service for integration with other applications

#### 2. **Vercel API** (`vercel-api/`)

- Serverless deployment optimized for Vercel
- **Components**:
  - `api.py` - Main API endpoint
  - `model_loader.py` - ML model initialization
  - `predict.py` - Inference logic
  - `vercel.json` - Vercel configuration

#### 3. **Hugging Face Space** (`oral-cancer-space/`)

- Public-facing demo interface
- Streamlit-based interactive UI
- Deployed on Hugging Face Spaces for easy access

#### 4. **Node.js Integration** (`Node/`)

- `package.json` - JavaScript dependencies
- `testapi.mjs` - API testing module (ES modules)

## 🤖 Model Details

- **Model**: Swin Transformer for Image Classification
- **Model File**: `oral_cancer_swin_new.pth` (root directory & model folder)
- **Framework**: PyTorch + Hugging Face Transformers
- **Input**: Medical images of oral cavity
- **Output**: Classification result (Cancer/No Cancer) with confidence scores

## 🔧 Technologies Used

- **Deep Learning**: PyTorch, Transformers (Swin Vision Transformer)
- **Frontend**: Streamlit (interactive web UI)
- **Backend API**: FastAPI
- **Cloud Deployment**: Vercel, Hugging Face Spaces
- **Image Processing**: Pillow, NumPy
- **Utilities**: gdown (for model downloads), ReportLab (PDF generation)

## 📋 Dependencies

### Main Requirements

```
torch
torchvision
transformers
streamlit
fastapi
uvicorn
Pillow
numpy
gdown
reportlab
```

## 🚀 Getting Started

### Local Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the local application:
   ```bash
   streamlit run Local-app.py
   ```

### Running the API

```bash
cd oral-cancer-api
pip install -r requirements.txt
uvicorn app:app --reload
```

### Environment Variables

- Set Hugging Face token for model access (if required):
  ```
  HUGGINGFACE_HUB_TOKEN=your_token_here
  ```

## 📊 Features

- **Image Upload**: Support for medical image inputs
- **Real-time Prediction**: Fast inference using optimized model
- **Confidence Scores**: Probability-based classification results
- **PDF Reports**: Generate clinical reports (if PDF module available)
- **Multi-platform Support**: Web, API, and serverless deployments

## 🌐 Deployment Endpoints

- **Local**: `http://localhost:8501` (Streamlit)
- **API**: `http://localhost:8000` (FastAPI)
- **Vercel**: Check `vercel-api/README.md` for deployment details
- **Hugging Face**: Check `oral-cancer-space/README.md` for Spaces details

## ⚠️ Important Notes

- Ensure GPU availability for optimal performance
- Model downloading may require gdown authentication
- Hugging Face API token required for certain model access
- This tool is for research and demonstration purposes - not for clinical diagnosis

## 💡 Model Performance

- **Architecture**: Swin Transformer (Vision Transformer variant)
- **Training Data**: Oral cancer medical imagery dataset
- **Input Size**: Standard medical image dimensions
- **Output Classes**: Binary classification (Cancer/No Cancer)

## 🔗 Resources & Links

### Dataset

- **Kaggle Dataset**: [Oral Cancer Dataset 2.0](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset?select=Oral+cancer+Dataset+2.0)

### Deployment & Demo

- **Live Demo (Streamlit)**: [Capstone Project App](https://capstoneprojectapp.streamlit.app/)
- **Hugging Face Space**: [Oral Cancer Detection](https://huggingface.co/spaces/Subbareddy1/oral-cancer-detection)

### Training & Model

- **Training Code (Google Colab)**: [Notebook Link](https://colab.research.google.com/drive/1onAzH5oT1NUX6G_dnrOrQ2KUNYueGHLd?usp=sharing)
- **Trained Model**: [Download Model Weights](https://drive.usercontent.google.com/download?id=1CIDpp_rZYc1us5H6iXE9r4WDOQBLXm9f&authuser=0)
