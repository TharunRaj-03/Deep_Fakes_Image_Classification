import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


model = tf.keras.models.load_model(r'model_1_h5.h5')


st.title('Deep fake Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    

    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    image_array = cv2.resize(image_array, (150, 150))
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    st.image(image_array, caption='Uploaded Image', use_column_width=True)

    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)

    color_dict = {0: 'red', 1: 'green'}
    
    # Mapping of class labels to their corresponding class names
    pred_dict = {0: 'Fake', 1: 'Original'}
    
    # Getting the predicted class
    predicted_class = pred_dict[np.argmax(prediction)]
    
    # Displaying the predicted class with the corresponding color
    st.subheader("Prediction:")
    st.markdown(f'<font color="{color_dict[np.argmax(prediction)]}">{predicted_class}</font>', unsafe_allow_html=True)
    
    # Displaying prediction probabilities for each class
    st.subheader("Prediction Probabilities:")
    
    for i, class_label in pred_dict.items():
        st.write(f"{class_label}: {prediction[0][i]:.2f}")

    st.balloons()


