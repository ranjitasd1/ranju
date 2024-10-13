import streamlit as st
import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import base64
from io import BytesIO

device = "cpu"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 55 * 55, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

@st.cache_resource()
def load_model(model_path):
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image, class_names):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def page_predict():
    st.title("Image Classification")
    model_path = "checkpoint/simple_cnn_model.pth"
    if model_path:
        model = load_model(model_path)
        st.write("Model loaded successfully!")
        class_names = ['1 Mixed local stock 2', 'Carniolan honey bee', 'Italian honey bee', 'Russian honey bee']
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_str = image_to_base64(image)
            st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{img_str}' alt='Uploaded Image' width='200'></div>", unsafe_allow_html=True)
            st.write("")
            with st.spinner("Classifying..."):
                label = predict_image(model, image, class_names)
                st.markdown(f"<h3 style='color: green; text-align: center;'>Predicted class: {label}</h3>", unsafe_allow_html=True)

def page_results():
    st.title("Results")
    results_folder = "results"
    if results_folder:
        files = os.listdir(results_folder)
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                st.image(os.path.join(results_folder, file), caption=file, use_column_width=True)
            elif file.endswith(".csv"):
                df = pd.read_csv(os.path.join(results_folder, file))
                # st.write("Results")
                st.dataframe(df)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Image", "View Results"])

if page == "Predict Image":
    page_predict()
elif page == "View Results":
    page_results()

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h3 {
        font-size: 1.75em;  /* Increase the font size of the prediction text */
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)