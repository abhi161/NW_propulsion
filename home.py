import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import itertools


def home():
    

    logo_path = "incoming_data\logo with title.png"

    st.image(logo_path,width=250)
    st.title("NW R&D Division")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.image("incoming_data\prop.png", width = 600)

    with col2:
        st.header("Pioneering Electric Aviation")

        st.write("We’re India’s 1st Electric Propulsion Technology Company. Our mission is to contribute to the global effort of .\
                  sustainable aviation by developing advanced electric propulsion systems for aircrafts & unmanned systems. We believe .\
                 that the future of flight lies in the integration of electric power & are committed to working towards a cleaner & more efficient industry.")
    

