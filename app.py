import streamlit as st
import dropbox
import os
import pickle
import numpy as np
from PIL import Image

image_icon = Image.open('temp/icon/mango.png')

st.set_page_config(
    page_title='Mango Leaf Classifier', 
    layout='wide', 
    initial_sidebar_state='auto', 
    page_icon=image_icon)

# Dropbox access token
DROPBOX_TOKEN = 'sl.Bfj-jdNTIkGy2LaWidV1y4oFgtYigkDQnm_r3tUrN_SY92B3y8nz8PaGZHqQy-nhdYwiHT4mmRo47cKLjfjCjoV5vFGFPGdCusONhgT0uSJQpuIK284mPtdnFq-TCaTrPP4IljSIef0'

# Dropbox folder path and model file name
DROPBOX_FOLDER_PATH = '/image-classify-yuniar'
MODEL_FILE_NAME = 'paramC5_acc91%.pkl'

# Local directory to save the model file
LOCAL_DIR = 'temp/model'

# Check if the model file already exists locally
def check_model_exists():
    model_path = os.path.join(LOCAL_DIR, MODEL_FILE_NAME)
    return os.path.exists(model_path)

# Download the model file from Dropbox
def download_model_from_dropbox():
    if check_model_exists():
        return

    # Create a Dropbox client
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)

    # Create the local directory if it doesn't exist
    os.makedirs(LOCAL_DIR, exist_ok=True)

    with st.spinner('Downloading model file...'):
        # Specify the local path to save the model file
        local_model_path = os.path.join(LOCAL_DIR, MODEL_FILE_NAME)

        # Download the model file from Dropbox
        dbx.files_download_to_file(local_model_path, f"{DROPBOX_FOLDER_PATH}/{MODEL_FILE_NAME}")

    st.success('Model file downloaded successfully')

# Run the download function
download_model_from_dropbox()

@st.cache_data()
def load_model():
    # Check if the model file exists locally
    local_model_path = 'temp/model/paramC5_acc91%.pkl'
    if not os.path.exists(local_model_path):
        # Download the model file from Dropbox
        download_model_from_dropbox()
    
    # Load the model
    model = pickle.load(open(local_model_path, 'rb'))
    return model

# Load the model
model = load_model()

# Create a function to make predictions
def predict_image_class(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image
    image_array = np.array(image)  # Convert image to array
    image_array = image_array / 255  # Normalize the image
    
    # Make prediction
    prediction = model.predict(image_array.reshape(1, -1))
    
    # Get the predicted class label
    return prediction

# Create the web app
def main():
    st.title("Image Prediction Web App")
    with st.sidebar:
        st.image(image_icon)
        st.title('Mango Leafes Classifier')
        menu = st.selectbox("Menu", ["Home", "About"])
        st.markdown("---")
        st.caption('This project trained with [Mango🥭 Leaf🍃🍂 Disease Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset), using Support Vector Machine [(SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and normalization to preprocess data.')
        st.markdown("---")
        st.caption('Result of this project is model can classify image into 5 class Antrachnose, Bacterial Canker, Die Back, Healthy and Powdery Mildew with XX% validation_accuracy.')
        
    if menu == "Home":
        st.header("Upload an Image")
        
        # Upload the image file
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            image = image.resize((300, 300)) 
            st.image(image, caption="Uploaded Image")
            
            # Predict the image class
            if st.button("Predict"):
                with st.spinner('Predicting...'):
                    predicted_class = predict_image_class(image)
                    st.success(f"Predicted Class: {predicted_class}")
    
    elif menu == "About":
        st.header("About")
        st.write("This is a web app for image prediction using a pre-trained model.")
        # Add more information about the app here
        
# Run the web app
if __name__ == '__main__':
    main()