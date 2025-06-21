import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import torch
from torchvision import models
import torch.nn as nn

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)

# Constants
IMAGE_SIZE = 128

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 1em;
    }
    .upload-prompt {
        text-align: center;
        padding: 2em;
        border: 2px dashed #cccccc;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        # Remove final layer for feature extraction
        self.model.classifier = nn.Identity()  
    
    def forward(self, x):
        return self.model(x)

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Aggregator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

client_a = ClientModel()
client_b = ClientModel()

@st.cache_resource
def load_prediction_model():
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals({'ClientModel': ClientModel, 'Aggregator': Aggregator})

        client_a = torch.load('client_a.pth', weights_only=False, map_location='cpu')
        client_b = torch.load('client_b.pth', weights_only=False, map_location='cpu')
        aggregator = torch.load('aggregator.pth', weights_only=False, map_location='cpu')
        
        return client_a.to('cpu'), client_b.to('cpu'), aggregator.to('cpu')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None



def preprocess_image(img):
    # Convert to RGB if image is grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to array
    img_array = np.array(img)
    
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    
    # Debug information
    st.sidebar.write("Image shape:", img_array.shape)
    st.sidebar.write("Image dtype:", img_array.dtype)
    st.sidebar.write("Value range:", np.min(img_array), "-", np.max(img_array))
    
    return img_array

def get_prediction(img_array, client_a, cilent_b, aggregator):
    try:
        features_a = client_a(torch.Tensor(img_array).to('cpu'))
        features_b = client_b(torch.Tensor(img_array).to('cpu'))
        combined_features = torch.cat((features_a, features_b), dim=1)
        outputs = torch.sigmoid(aggregator(combined_features))
        predicted = (outputs).float()
        return predicted
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error(f"Input shape: {img_array.shape}")
        return None

def main():
    st.title("🫁 Pneumonia Detection from X-rays")
    st.write("Upload a chest X-ray image to check for signs of pneumonia.")
    
    # Load model
    client_a, client_b, aggregator = load_prediction_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Read and display uploaded image
            image_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(image_bytes))
            
            # Display original image info
            st.sidebar.write("Original image mode:", img.mode)
            st.sidebar.write("Original image size:", img.size)
            
            # Display the image
            st.image(img, caption="Uploaded X-ray")
            
            # Add a divider
            st.markdown("---")
            
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                processed_img = preprocess_image(img)
                
                # Get prediction
                prediction = get_prediction(processed_img, client_a, client_b, aggregator)
                
                if prediction is not None:
                    probability = float(prediction[0][0])
                    probability = max((probability-0.6),0)/0.6
                    
                    # Create columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Pneumonia Probability", 
                            f"{probability:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Normal Probability", 
                            f"{(1-probability):.1%}"
                        )
                    
                    # Progress bar
                    st.progress(probability)
                    
                    # Display results with recommendations
                    if probability > 0.5:
                        st.error("⚠️ Potential Pneumonia Detected")
                        st.markdown("""
                            ### Recommended Actions:
                            1. Consult a healthcare provider immediately
                            2. Seek medical evaluation
                            3. Monitor symptoms closely
                        """)
                    else:
                        st.success("✅ No Pneumonia Detected")
                        st.markdown("""
                            ### Recommendations:
                            1. Continue regular health check-ups
                            2. Maintain good respiratory hygiene
                            3. Stay updated with vaccinations
                        """)
                    
                    # Add disclaimer
                    st.warning("""
                        **Medical Disclaimer**: This tool is for screening purposes only and 
                        should not be used as a substitute for professional medical advice, 
                        diagnosis, or treatment. Always seek the advice of your physician or 
                        other qualified health provider.
                    """)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Full error details:", exc_info=True)
    
    else:
        # Show upload prompt when no file is uploaded
        st.markdown("""
            <div class="upload-prompt">
                <h3>📤 Upload an X-ray image to begin analysis</h3>
                <p>Supported formats: JPG, JPEG, PNG</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 