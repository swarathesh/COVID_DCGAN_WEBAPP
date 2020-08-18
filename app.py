import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


def predict(image_data, model):
    image = np.asarray(ImageOps.fit(image_data, (75, 75), Image.ANTIALIAS).convert('RGB'))
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


model = tf.keras.models.load_model('model.hdf5')

st.write("""
         # Covid-19 Detection
         """
         )

st.write("Predict Covid-19 from the X-ray")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("Normal")
    else:
        st.write("Covid-19 detected")

    st.text("Probability (0: Normal, 1: Covid-19)")
    st.write(prediction)
