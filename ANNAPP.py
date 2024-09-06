import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define ANN model
def create_ann_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the ANN model
model = create_ann_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Page title
st.title("Digit Recognition using ANN")

# Display accuracy 
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

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

