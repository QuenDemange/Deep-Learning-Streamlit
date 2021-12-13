import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path


st.set_page_config(page_title="Ship prediction", page_icon="🚢", layout='centered', initial_sidebar_state="collapsed")

labels_inv = {'Cargo': 0,
              'Military': 1,
              'Carrier': 2,
              'Cruise': 3,
              'Tankers': 4}
labels = {v:k for k,v in labels_inv.items()}

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:DarkRed;text-align:left;"> Ship prediction  🚢 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
            The detection of ships is an important military subject, every country in the world would like to know from a satellite image to recognize the type of ship in order to prepare a defense if an attack arrives.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


    with col2:
        df = pd.DataFrame()
        upload_file = st.file_uploader("Choose a file of ship.")

        if upload_file is not None:
            bytes_data = upload_file.getvalue()
            fd = open("./img_to_predict.jpg", "wb")
            fd.write(bytes_data)
            fd.close()
            st.image(bytes_data)
        
        if st.button('Predict'):
            loaded_model = tf.keras.models.load_model("./model.h5", compile=True)
            loaded_model.summary()
            if Path("./img_to_predict.jpg").exists():
                img = tf.keras.preprocessing.image.load_img("./img_to_predict.jpg", target_size=(128, 128), interpolation='lanczos')
                img = tf.keras.preprocessing.image.img_to_array(img)
                pred = loaded_model.predict(np.array([img]))
                pred_label = np.argsort(pred)
                for i in pred_label[0]:
                    st.write(f"{labels[i]} Ship : {pred[0][i]*100:0.2f} %")

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()