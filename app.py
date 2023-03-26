import pandas as pd
import streamlit as st
import base64
from view import show_data_base
from explore import show_explore_page
from interextrpolate import process

from streamlit_lottie import st_lottie
import requests
from home import home
from pyngrok import ngrok
import nest_asyncio


st.set_page_config(layout="wide")

def load_lottie(url:str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

# plane = load_lottie("https://assets9.lottiefiles.com/packages/lf20_CK0nF6.json
# ")
plane1 = load_lottie("https://assets7.lottiefiles.com/packages/lf20_xhlbndhm.json")


def main():
    
    logo_path = "incoming_data\logo with title.png"
    # input_file_path = "incoming_data\inputfile.csv"

    # st.image(logo_path,width=250)
    # st.title('NW Propulsion Database')
    # st_lottie(plane,height=215,width=222)
    st_lottie(plane1,height=125,width=122)

    sure =st.sidebar.selectbox("Explore or View or Predict", ("Home","View","Visualize","Predict"))

    if sure == "Visualize":
        show_explore_page()
        
    elif sure == "View":
        show_data_base()

    elif sure == "Predict":
        process()
    
    else:
        home()

if __name__ == "__main__":
    main()