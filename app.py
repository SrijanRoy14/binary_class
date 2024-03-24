import streamlit as st
import tensorflow as tf
from keras.models import load_model

import os
from PIL import Image
import numpy as np

st.title("Mood Detection")

st.header("Upload an image of a person either happy or sad")


file=st.file_uploader('',type=['jpeg','jpg','bmp','png'])

model=load_model(os.path.join('models','binaryimageclassifiernewversion.h5'))

if file is not None:
    image=Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)

    resize=tf.image.resize(np.array(image),(256,256))
    
    yhat=model.predict(np.expand_dims(resize/255,0))
    if yhat>0.5:
        class_name="Sad Person"
        conf_score=yhat[0][0]*100

    else:
        print('Prediction: this is a happy person')
        class_name="Happy Person"
        conf_score=100-(yhat[0][0]*100)

    

    st.write("## {}".format(class_name))
    st.write("### Confidence: {:.2f}%".format(conf_score))



