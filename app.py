import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/cancer_detection_model3.h5')

# Class labels
class_labels = ['class1', 'class2', 'class3', 'class4']

# Streamlit App
st.title("Cancer Detection App")

# Navigation bar
menu = ["Prediction", "Performance Analysis"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Prediction":
    st.header("Image Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        percentage_predictions = [f"{label}: {round(prob * 100, 2)}%" for label, prob in zip(class_labels, predictions[0])]

        # Display predictions
        st.subheader("Predictions:")
        for percentage in percentage_predictions:
            st.write(percentage)

elif choice == "Performance Analysis":
    st.header("Model Performance Analysis")

    # Display confusion matrix
    st.subheader("Confusion Matrix:")
    confusion_matrix_image = Image.open('/path/to/your/confusion_matrix_image.png')
    st.image(confusion_matrix_image, caption="Confusion Matrix", use_column_width=True)

    # Display loss chart
    st.subheader("Loss Chart:")
    loss_chart_image = Image.open('/path/to/your/loss_chart_image.png')
    st.image(loss_chart_image, caption="Loss Chart", use_column_width=True)

    # Display accuracy chart
    st.subheader("Accuracy Chart:")
    accuracy_chart_image = Image.open('/path/to/your/accuracy_chart_image.png')
    st.image(accuracy_chart_image, caption="Accuracy Chart", use_column_width=True)
