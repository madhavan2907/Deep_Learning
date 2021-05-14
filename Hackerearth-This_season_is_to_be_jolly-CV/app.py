# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 2021 - 13:43:00
@author: Sethumadhavan Aravindakshan
"""

import numpy as np
import os
import pandas as pd
import streamlit as st
import keras
from PIL import Image
# from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from pathlib import Path
# p = Path('.')

export_path = os.path.join(os.getcwd(), 'TL_Inception_resnet_v2.h5')
print(export_path)

@st.cache(suppress_st_warning=True)
def prediction_fun(img):
    # st.write("Caching for the first time")
    # pred_model=keras.models.load_model('./TL_Inception_resnet_v2.h5')
    pred_model=keras.models.load_model(export_path)
    return pred_model.predict(img.reshape(1,224,224,3))
    




def main():
    # st.title("Image classification through Transfer Learning")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Image classification through Transfer Learning </h2>
    </div>
    
    <div style="background-color:white;padding:10px">
    <p style="color:black;text-align:left;">This is an Image classification application built for the below Hackerearth CV competition: </p>
    <a href="https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-holidays/">Competition Link</a>
    
    <h2 style="color:black;text-align:left;">Problem statement: </h2>
    <p style="color:black;text-align:left;">You work for a social media platform. Your task is to create a solution using deep learning to discern whether a post is holiday-related in an effort to better monetize the platform.</p>
    
    <h2 style="color:black;text-align:left;">Task: </h2>
    <p style="color:black;text-align:left;">You are given the following six categories. You are required to classify the images in the dataset based on these categories.</p>
    <ul>
    <li>Miscellaneous</li>
    <li>Christmas_Tree</li>
    <li>Jacket</li>
    <li>Candle</li>
    <li>Airplane</li>
    <li>Snowman</li>
    </ul>

    <h2 style="color:black;text-align:left;">Dataset: </h2>
    <a href="https://www.kaggle.com/nikhil741/hackerearth-holiday-season">Dataset Link</a>
    <p style="color:black;text-align:left;">The data folder consists of two folders and one .csv file. The details are as follows:</p>
    <ul>
    <li>train: Contains 6469 images for 6 classes</li>
    <li>test: Contains 3489 images</li>
    <li>train.csv: 3489 x 2</li>
    </ul>

  
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

    fig_path = os.path.join(os.getcwd(), 'plot.jpg')
    print(fig_path)  
    freq_plot = Image.open(fig_path)
    st.write("Output class distribution")
    st.image(freq_plot,  use_column_width=True)
    # Getting Inputs and processing it for prediction
    image_upload = st.file_uploader("Upload the image", type='jpg')

    if image_upload is not None:
            st.image(image_upload, caption='Uploaded Image.', use_column_width=True)

    

    
    if st.button("Predict"):

        print('Predicting')

        
        # preprocess = imagenet_utils.preprocess_input
        image = Image.open(image_upload)
        new_width  = 224
        new_height = 224
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        image = img_to_array(image)
        image = image/255
        # print("Done")


        print(f"[#] classifying image with Inception_Resnet_V2'... ")
        pred = prediction_fun(image)


        classes = ['Airplane', 'Candle', 'Christmas_Tree', 'Jacket', 'Miscellaneous', 'Snowman']
        prediction=[]
        prediction.append(classes[np.argmax(pred[0])])
        print(prediction)
        st.warning(f"The uploaded image is a : {prediction[0]}")
        st.balloons()


if __name__ == '__main__':
    main()
