import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
from scipy.stats import kurtosis, skew
import skimage.measure    

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

if __name__ == "__main__":
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    str = "Predict whether the note is authentic or fake"
    
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
        # our_image = cv2.imread(our_image, cv2.IMREAD_GRAYSCALE)
    
        curtosis = kurtosis(our_image, axis = None)
        skewness = skew(our_image, axis = None)
        variance = np.var(our_image)
        entropy = skimage.measure.shannon_entropy(our_image)
   
        if st.button("Predict"):
            result=predict_note_authentication(variance,skewness,curtosis,entropy)
            
        if(result == 0):
            str = "Beware!! The note is fake!!"
        else:
            str = "The note is authentic"
        
    st.success(f'{str}')