"""
Underwater Image Classifier
Combined DeepFish and Seagrass Classification App
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import timm
import time
import gdown

# ============================================
# MODEL DOWNLOAD FROM GOOGLE DRIVE
# ============================================

# Google Drive file IDs
DEEPFISH_MODEL_ID = "1tcMMAxtSTimryZ8MoefgOrv7vakJVLTK"
SEAGRASS_MODEL_ID = "1yDuMBlGS5DYxiQACle7EgXG2orOfkDIo"

# Model file paths
DEEPFISH_MODEL_PATH = "deepfish.pth"
SEAGRASS_MODEL_PATH = "efficientnet_seagrass.pth"

def download_model_from_gdrive(file_id, output_path):
    """Download model from Google Drive if it doesn't exist"""
    if not os.path.exists(output_path):
        try:
            st.info(f"Downloading {output_path} from Google Drive... This may take a few minutes.")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            st.success(f"✅ {output_path} downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to download {output_path}: {str(e)}")
            return False
    return True

# Page config
st.set_page_config(
    page_title="Underwater Image Classifier",
    page_icon="🐠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
    
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DEEPFISH MODEL
# ============================================

class DeepFishClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepFishClassifier, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

@st.cache_resource
def load_deepfish_model():
    try:
        # Download model if it doesn't exist
        if not download_model_from_gdrive(DEEPFISH_MODEL_ID, DEEPFISH_MODEL_PATH):
            return None, None, "Failed to download model from Google Drive"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepFishClassifier(num_classes=2)
        checkpoint = torch.load(DEEPFISH_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, device, None
    except Exception as e:
        return None, None, str(e)

def predict_deepfish(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        
        if pred_class.item() == 1:
            prediction = "Fish"
            conf_score = probs[0][1].item()
        else:
            prediction = "No Fish"
            conf_score = probs[0][0].item()
    
    return prediction, conf_score, probs[0].tolist()

# ============================================
# SEAGRASS MODEL
# ============================================

SEAGRASS_CLASSES = ['Amphibolis', 'Background', 'Halophila', 'Posidonia']
IMG_SIZE = 456

@st.cache_resource
def load_seagrass_model():
    try:
        # Download model if it doesn't exist
        if not download_model_from_gdrive(SEAGRASS_MODEL_ID, SEAGRASS_MODEL_PATH):
            return None, "Failed to download model from Google Drive"
        
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=False, num_classes=4)
        checkpoint = torch.load(SEAGRASS_MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_seagrass(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor

def predict_seagrass(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities[0].tolist()

# ============================================
# MAIN APP
# ============================================

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# HOME PAGE
if st.session_state.page == 'home':
    st.markdown('<h1 class="main-title">🐠 Underwater Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning for Marine Life & Seagrass Detection</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                    height: 220px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🐟</div>
            <h2 style="color: white; font-size: 1.8rem; margin-bottom: 0.8rem; font-weight: 700;">DeepFish</h2>
            <p style="color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.5;">Detect presence of fish in underwater images.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.3);
                    height: 220px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🌿</div>
            <h2 style="color: white; font-size: 1.8rem; margin-bottom: 0.8rem; font-weight: 700;">Seagrass</h2>
            <p style="color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.5;">Classify seagrass species for ecological studies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True, type="primary"):
            st.session_state.page = 'classify'
            st.rerun()
    
    # Features section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem;">
        <h3 style="color: #a78bfa; margin-bottom: 2rem; font-size: 1.8rem;">✨ Key Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #1e293b; border-radius: 12px; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
            <h4 style="color: #cbd5e1; margin-bottom: 0.5rem;">Fast Processing</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Real-time inference with optimized models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #1e293b; border-radius: 12px; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="color: #cbd5e1; margin-bottom: 0.5rem;">High Accuracy</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">State-of-the-art deep learning models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #1e293b; border-radius: 12px; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #cbd5e1; margin-bottom: 0.5rem;">Detailed Analytics</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Confidence scores and probabilities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: #1e293b; border-radius: 12px; height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔬</div>
            <h4 style="color: #cbd5e1; margin-bottom: 0.5rem;">Research-Grade</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Suitable for ecological studies</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #334155;">
        <p style="font-size: 0.9rem;">🌊 Made with ❤️ for Marine Conservation | Powered by PyTorch & Streamlit</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem; color: #475569;">EfficientNet-B0 & EfficientNet-B5 Models</p>
    </div>
    """, unsafe_allow_html=True)

# CLASSIFICATION PAGE
else:
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #06b6d4 0%, #10b981 100%); padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🐠 Underwater Image Classification</h1>
        <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Upload an image and classify</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Diagnostics
    with st.expander("🔬 Model Diagnostics", expanded=False):
        df_model, df_device, df_error = load_deepfish_model()
        sg_model, sg_error = load_seagrass_model()
        
        st.markdown("### System & Model Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🐟 DeepFish Model")
            if df_error:
                st.error(f"❌ Failed to load")
                st.caption(f"Error: {df_error}")
                st.info("💡 Place 'deepfish.pth' in the same directory")
            else:
                st.success("✅ Successfully loaded")
                st.markdown(f"""
                **Architecture:** EfficientNet-B0  
                **Device:** {"🚀 GPU (CUDA)" if df_device.type == 'cuda' else "💻 CPU"}  
                **Parameters:** ~5.3M  
                **Test Accuracy:** 99.98%  
                **Input Size:** 224×224  
                **Classes:** 2 (Fish / No Fish)
                """)
        
        with col2:
            st.markdown("#### 🌿 Seagrass Model")
            if sg_error:
                st.error(f"❌ Failed to load")
                st.caption(f"Error: {sg_error}")
                st.info("💡 Place 'efficientnet_seagrass.pth' in the same directory")
            else:
                st.success("✅ Successfully loaded")
                st.markdown(f"""
                **Architecture:** EfficientNet-B5 (NoisyStudent)  
                **Device:** 💻 CPU  
                **Parameters:** ~30M  
                **Input Size:** 456×456  
                **Classes:** 4 (3 species + background)  
                **Framework:** PyTorch + timm
                """)
        
        st.markdown("---")
        
        # System information
        st.markdown("### 💻 System Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**PyTorch**")
            st.caption(f"Version: {torch.__version__}")
            st.caption(f"CUDA Available: {'Yes ✓' if torch.cuda.is_available() else 'No'}")
        
        with col2:
            st.markdown("**Streamlit**")
            st.caption(f"Version: {st.__version__}")
        
        with col3:
            st.markdown("**Device**")
            if torch.cuda.is_available():
                st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.caption("Running on CPU")
        
        st.markdown("---")
        
        # Performance tips
        st.markdown("### ⚡ Performance Tips")
        st.markdown("""
        - **GPU Acceleration**: For faster inference, ensure CUDA-compatible GPU is available
        - **Batch Processing**: Process multiple images for better throughput
        - **Image Size**: Resize very large images before upload for faster processing
        - **Model Loading**: Models are cached after first load for better performance
        """)
    
    # Tabs
    tab1, tab2 = st.tabs(["🐟 DeepFish", "🌿 Seagrass"])
    
    # DEEPFISH TAB
    with tab1:
        st.markdown("### 🐟 DeepFish Classifier")
        st.markdown("**Binary Classification**: Detects presence or absence of fish in underwater images using EfficientNet-B0")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['jpg', 'jpeg', 'png'],
            help="Limit 200MB per file • JPG, JPEG, PNG",
            key="deepfish_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("#### 📸 Uploaded Image")
                st.image(image, use_container_width=True)
                
                # Image info in an organized way
                with st.expander("📊 Image Details", expanded=False):
                    st.write(f"**Dimensions**: {image.size[0]} × {image.size[1]} px")
                    st.write(f"**Format**: {uploaded_file.type.split('/')[-1].upper()}")
                    st.write(f"**File Size**: {uploaded_file.size / 1024:.2f} KB")
                    st.write(f"**Color Mode**: RGB")
            
            with col2:
                st.markdown("#### 🎯 Prediction Results")
                
                if df_error:
                    st.error("❌ Model not loaded")
                    st.info("💡 Make sure 'deepfish.pth' is in the same directory")
                else:
                    with st.spinner('🔍 Analyzing image...'):
                        start = time.time()
                        prediction, confidence, probs = predict_deepfish(image, df_model, df_device)
                        inference_time = time.time() - start
                    
                    # Result display with better styling
                    if prediction == "Fish":
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                    padding: 1.5rem; 
                                    border-radius: 12px; 
                                    text-align: center;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 5px 20px rgba(16, 185, 129, 0.3);">
                            <h2 style="color: white; margin: 0; font-size: 1.8rem;">✓ Fish Detected!</h2>
                            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">The model identified fish in this image</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                                    padding: 1.5rem; 
                                    border-radius: 12px; 
                                    text-align: center;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 5px 20px rgba(59, 130, 246, 0.3);">
                            <h2 style="color: white; margin: 0; font-size: 1.8rem;">✓ No Fish</h2>
                            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">No fish detected in this image</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics in columns
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    with col_b:
                        st.metric("Inference", f"{inference_time*1000:.0f} ms")
                    with col_c:
                        st.metric("Device", "GPU" if df_device.type == 'cuda' else "CPU")
                    
                    # Confidence bar
                    st.markdown("**Confidence Level:**")
                    st.progress(confidence, text=f"{confidence*100:.2f}%")
                    
                    # Class probabilities with better visualization
                    st.markdown("---")
                    st.markdown("**Detailed Probabilities:**")
                    
                    classes = ["No Fish", "Fish"]
                    for i, cls in enumerate(classes):
                        col_name, col_prob, col_bar = st.columns([2, 1, 3])
                        with col_name:
                            emoji = "🐟" if cls == "Fish" else "🌊"
                            is_predicted = (i == 1 and prediction == "Fish") or (i == 0 and prediction == "No Fish")
                            if is_predicted:
                                st.markdown(f"**{emoji} {cls}** ✓")
                            else:
                                st.markdown(f"{emoji} {cls}")
                        with col_prob:
                            st.markdown(f"**{probs[i]*100:.2f}%**")
                        with col_bar:
                            st.progress(probs[i])
        else:
            st.info("👆 Upload an underwater image to detect fish")
            
            # Usage tips
            with st.expander("💡 Tips for Best Results", expanded=False):
                st.markdown("""
                **For optimal fish detection:**
                - Use clear, well-lit underwater images
                - Ensure the image is in focus
                - Avoid heavily distorted or blurry images
                - Supported formats: JPG, JPEG, PNG
                - Maximum file size: 200MB
                
                **Model Information:**
                - Trained on DeepFish dataset
                - Can detect multiple fish species
                - Works with various underwater conditions
                - Optimized for real-time inference
                """)
            
            # Example info
            with st.expander("📖 About DeepFish Model", expanded=False):
                st.markdown("""
                **Architecture**: EfficientNet-B0
                - Compound scaling method
                - Balances depth, width, and resolution
                - ~5.3M parameters
                
                **Training Details**:
                - Dataset: DeepFish (underwater fish images)
                - Augmentation: Rotation, flipping, color jitter
                - Optimizer: Adam with weight decay
                - Loss function: Cross-entropy
                
                **Performance**:
                - Test Accuracy: 99.98%
                - Precision: 99.97%
                - Recall: 99.99%
                - F1 Score: 99.98%
                """)
    
    # SEAGRASS TAB
    with tab2:
        st.markdown("### 🌿 Seagrass Classifier")
        st.markdown("**Multi-class Classification**: Identifies seagrass species for marine ecological research using EfficientNet-B5")
        st.markdown("---")
        
        uploaded_file2 = st.file_uploader(
            "Drag and drop file here",
            type=['jpg', 'jpeg', 'png'],
            help="Limit 200MB per file • JPG, JPEG, PNG",
            key="seagrass_upload"
        )
        
        if uploaded_file2:
            image = Image.open(uploaded_file2).convert('RGB')
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("#### 📸 Uploaded Image")
                st.image(image, use_container_width=True)
                
                # Image info
                with st.expander("📊 Image Details", expanded=False):
                    st.write(f"**Dimensions**: {image.size[0]} × {image.size[1]} px")
                    st.write(f"**Format**: {uploaded_file2.type.split('/')[-1].upper()}")
                    st.write(f"**File Size**: {uploaded_file2.size / 1024:.2f} KB")
                    st.write(f"**Color Mode**: RGB")
            
            with col2:
                st.markdown("#### 🎯 Classification Results")
                
                if sg_error:
                    st.error("❌ Model not loaded")
                    st.info("💡 Make sure 'efficientnet_seagrass.pth' is in the same directory")
                else:
                    with st.spinner('🔍 Classifying seagrass species...'):
                        start = time.time()
                        processed = preprocess_seagrass(image)
                        pred_class, confidence, probs = predict_seagrass(sg_model, processed)
                        inference_time = time.time() - start
                    
                    predicted_species = SEAGRASS_CLASSES[pred_class]
                    
                    # Species descriptions
                    species_info = {
                        'Amphibolis': {'emoji': '🌿', 'color': '#10b981', 'desc': 'Temperate Australian seagrass with wire-like leaves'},
                        'Background': {'emoji': '🌊', 'color': '#3b82f6', 'desc': 'Ocean floor without seagrass presence'},
                        'Halophila': {'emoji': '🍃', 'color': '#06b6d4', 'desc': 'Small delicate seagrass with oval leaves'},
                        'Posidonia': {'emoji': '🌾', 'color': '#059669', 'desc': 'Large robust seagrass forming dense meadows'}
                    }
                    
                    # Result display with better styling
                    info = species_info[predicted_species]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {info['color']} 0%, {info['color']}dd 100%); 
                                padding: 1.5rem; 
                                border-radius: 12px; 
                                text-align: center;
                                margin-bottom: 1rem;
                                box-shadow: 0 5px 20px {info['color']}50;">
                        <h2 style="color: white; margin: 0; font-size: 1.8rem;">{info['emoji']} {predicted_species}</h2>
                        <p style="color: rgba(255,255,255,0.95); margin-top: 0.5rem; font-size: 0.95rem;">{info['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    with col_b:
                        st.metric("Inference", f"{inference_time*1000:.0f} ms")
                    with col_c:
                        # Confidence level
                        if confidence > 0.9:
                            st.metric("Level", "Very High")
                        elif confidence > 0.7:
                            st.metric("Level", "High")
                        elif confidence > 0.5:
                            st.metric("Level", "Moderate")
                        else:
                            st.metric("Level", "Low")
                    
                    # Confidence bar
                    st.markdown("**Confidence Level:**")
                    st.progress(confidence, text=f"{confidence*100:.2f}%")
                    
                    # Species probabilities with better visualization
                    st.markdown("---")
                    st.markdown("**Species Probabilities:**")
                    
                    # Sort by probability for better visualization
                    prob_data = [(SEAGRASS_CLASSES[i], probs[i], species_info[SEAGRASS_CLASSES[i]]['emoji']) 
                                 for i in range(len(SEAGRASS_CLASSES))]
                    prob_data.sort(key=lambda x: x[1], reverse=True)
                    
                    for species, prob, emoji in prob_data:
                        col_name, col_prob, col_bar = st.columns([2, 1, 3])
                        with col_name:
                            if species == predicted_species:
                                st.markdown(f"**{emoji} {species}** ✓")
                            else:
                                st.markdown(f"{emoji} {species}")
                        with col_prob:
                            st.markdown(f"**{prob*100:.2f}%**")
                        with col_bar:
                            st.progress(prob)
                    
                    # Additional insights
                    st.markdown("---")
                    st.markdown("**Classification Insights:**")
                    
                    # Determine confidence assessment
                    if confidence > 0.9:
                        conf_assessment = "Very high confidence - Model is highly certain about this classification"
                        conf_color = "#10b981"
                    elif confidence > 0.7:
                        conf_assessment = "High confidence - Model has strong certainty about this classification"
                        conf_color = "#3b82f6"
                    elif confidence > 0.5:
                        conf_assessment = "Moderate confidence - Consider reviewing the image quality"
                        conf_color = "#f59e0b"
                    else:
                        conf_assessment = "Low confidence - Image may be unclear or ambiguous"
                        conf_color = "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background: #1e293b; 
                                padding: 1.2rem; 
                                border-radius: 10px; 
                                border-left: 5px solid {conf_color};
                                margin-top: 0.5rem;">
                        <p style="margin: 0; color: #cbd5e1;"><strong>Assessment:</strong> {conf_assessment}</p>
                        <p style="margin-top: 0.8rem; margin-bottom: 0; color: #94a3b8;">
                            <strong>Top Prediction:</strong> {predicted_species} ({confidence*100:.2f}%)<br>
                            <strong>Second Best:</strong> {prob_data[1][0]} ({prob_data[1][1]*100:.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload a seagrass image for species classification")
            
            # Species information cards
            st.markdown("#### 🌿 Seagrass Species Information")
            
            species_details = {
                'Amphibolis': {
                    'emoji': '🌿',
                    'name': 'Amphibolis',
                    'scientific': 'Amphibolis spp.',
                    'habitat': 'Temperate Australian coastal waters',
                    'characteristics': 'Wire-like stems, small clustered leaves, forms dense meadows',
                    'importance': 'Critical habitat for fish and invertebrates'
                },
                'Halophila': {
                    'emoji': '🍃',
                    'name': 'Halophila',
                    'scientific': 'Halophila spp.',
                    'habitat': 'Tropical and subtropical waters worldwide',
                    'characteristics': 'Small oval leaves, delicate appearance, rapid growth',
                    'importance': 'Pioneer species in seagrass colonization'
                },
                'Posidonia': {
                    'emoji': '🌾',
                    'name': 'Posidonia',
                    'scientific': 'Posidonia spp.',
                    'habitat': 'Mediterranean and Australian waters',
                    'characteristics': 'Large robust leaves, forms extensive meadows, slow-growing',
                    'importance': 'Biodiversity hotspot, carbon sequestration'
                },
                'Background': {
                    'emoji': '🌊',
                    'name': 'Background',
                    'scientific': 'N/A',
                    'habitat': 'Ocean floor without vegetation',
                    'characteristics': 'Sand, rocks, or bare substrate',
                    'importance': 'Reference class for seagrass detection'
                }
            }
            
            for species_name, details in species_details.items():
                with st.expander(f"{details['emoji']} {details['name']}", expanded=False):
                    st.markdown(f"**Scientific Name:** {details['scientific']}")
                    st.markdown(f"**Habitat:** {details['habitat']}")
                    st.markdown(f"**Characteristics:** {details['characteristics']}")
                    st.markdown(f"**Importance:** {details['importance']}")
            
            # Usage tips
            with st.expander("💡 Tips for Best Results", expanded=False):
                st.markdown("""
                **For optimal seagrass classification:**
                - Use clear overhead or angled shots of seagrass beds
                - Ensure good lighting conditions
                - Include sufficient leaf detail in the frame
                - Avoid excessive water turbidity
                - Best captured at depths of 1-3 meters
                - Supported formats: JPG, JPEG, PNG
                - Maximum file size: 200MB
                
                **Model Capabilities:**
                - Distinguishes between 4 different classes
                - Handles various lighting conditions
                - Works with different camera angles
                - Optimized for ecological research
                """)
            
            # About the model
            with st.expander("📖 About Seagrass Model", expanded=False):
                st.markdown("""
                **Architecture**: EfficientNet-B5 (NoisyStudent)
                - Advanced compound scaling
                - Semi-supervised learning approach
                - ~30M parameters
                - Higher capacity than B0 variant
                
                **Training Details**:
                - Dataset: Annotated seagrass imagery
                - Classes: 4 (3 species + background)
                - Augmentation: Rotation, flipping, color adjustment, cropping
                - Optimizer: AdamW with cosine annealing
                - Loss function: Cross-entropy with label smoothing
                
                **Applications**:
                - Marine ecological surveys
                - Seagrass meadow monitoring
                - Biodiversity assessment
                - Climate change research
                - Conservation planning
                """)
    
    # Back button
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("← Back to Home", use_container_width=True, type="primary"):
            st.session_state.page = 'home'
            st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #334155;">
        <p style="font-size: 0.9rem;">🌊 Made with ❤️ for Marine Conservation | Powered by PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)