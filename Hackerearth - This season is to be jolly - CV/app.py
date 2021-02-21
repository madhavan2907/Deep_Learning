# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 2021 - 13:43:00
@author: Sethumadhavan Aravindakshan
"""

import numpy as np
import os
import pickle
import pandas as pd
import streamlit as st
import keras
from PIL import Image
# from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


pred_model1=keras.models.load_model("./Notebooks/Models/TL_Inception_resnet_v2.h5")
# pred_model2=keras.models.load_model("TL_VGG16.h5")
# pred_model3=keras.models.load_model("TL_inception.h5")


def main():
    # st.title("Image classification through Transfer Learning")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Image classification through Transfer Learning </h2>
    # <h2 style="color:white;text-align:center;">Image classification through Transfer Learning </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/originals/85/6f/31/856f31d9f475501c7552c97dbe727319.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Getting Inputs and processing it for prediction
    image_upload = st.file_uploader("Upload the image", type='jpg')

    if image_upload is not None:
        image = Image.open(image_upload)
        new_width  = 224
        new_height = 224
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        st.image(image_upload, caption='Uploaded Image.', use_column_width=True)

#     from PIL import Image
# img = Image.open('/your iamge path/image.jpg') # image extension *.png,*.jpg
# new_width  = 128
# new_height = 128
# img = img.resize((new_width, new_height), Image.ANTIALIAS)
# img.save('/new directory path/output image name.png') 

    if st.button("Predict"):

        print('Predicting')

        # inputShape = (224, 224)
        preprocess = imagenet_utils.preprocess_input
        # preprocess = preprocess_input
        # preprocess = imagenet_utils.preprocess_input

        
        # image = load_img(image_upload, target_size=inputShape)
        image = img_to_array(image)

        image = image/255
        print("Done")


        # image = np.expand_dims(image, axis=0)
        # image = preprocess(image)

        print(f"[#] classifying image with Inception_Resnet_V2'... ")
        pred1 = pred_model1.predict(image.reshape(1,224,224,3))
        # pred2 = pred_model2.predict(image.reshape(1,224,224,3))
        # pred3 = pred_model3.predict(image.reshape(1,224,224,3))
        # print(pred)

        classes = ['Airplane', 'Candle', 'Christmas_Tree', 'Jacket', 'Miscellaneous', 'Snowman']
        prediction1=[]
        prediction1.append(classes[np.argmax(pred1[0])])
        print(prediction1)
        st.warning(f"Pred1 {prediction1}")

        # prediction2=[]
        # prediction2.append(classes[np.argmax(pred2[0])])
        # print(prediction2)
        # st.warning(f"Pred2 {prediction2}")

        # prediction3=[]
        # prediction3.append(classes[np.argmax(pred3[0])])
        # print(prediction3)
        # st.warning(f"Pred3 {prediction3}")
        
        
    


if __name__ == '__main__':
    main()