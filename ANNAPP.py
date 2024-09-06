import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load the trained model (ANN)
#model = tf.keras.models.load_model("ann_model.h5")
import joblib

# Load the trained model (ANN)
model = joblib.load("ann_model.joblib")


# Page title
st.title("Digit Recognition using ANN")

# Display accuracy 
st.write("This model predicts handwritten digits (0-9) using the ANN trained on the load_digits dataset.")

# Image upload functionality
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Resize to 8x8 (the size of images in the load_digits dataset)
    image = image.resize((8, 8))
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Invert the colors (white background, black digit)
    image_array = 255 - image_array
    
    # Normalize and reshape the image to fit the model input
    image_array = image_array / 16.0  # Scaling similar to load_digits
    image_array = image_array.reshape(1, 64)  # Flatten the image to 64 features
    return image_array

# If an image is uploaded
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make Prediction
    if st.button("Predict"):
        prediction = model.predict(preprocessed_image)
        predicted_label = np.argmax(prediction, axis=-1)
        st.write(f'Prediction: The uploaded digit is likely a {predicted_label[0]}')

# Let the user know the app is ready
st.write("Please upload an image of a handwritten digit to predict its label.")