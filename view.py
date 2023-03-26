import pandas as pd
import streamlit as st
import base64


def show_data_base():
    logo_path = "incoming_data\logo with title.png"
    input_file_path = "incoming_data\inputfile.csv"

    container = st.container()
    with container:
        col1, col2 = st.columns(2)

    with col1:
        st.image(logo_path,width=250)
        st.write("")
        st.title('NW Propulsion Database')
    
    # with col2:
    #     col1, col2, col3 = st.columns(3)
    #     col1.metric("Temperature", "70 °F", "1.2 °F")
    #     col2.metric("Wind", "9 mph", "-8%")
    #     col3.metric("Humidity", "86%", "4%")


    data = pd.read_csv(input_file_path)
    df= data.copy()

    st.write("")
    container = st.container()
    with container:
        col1, col2, col3, col4, col5, col6, col7, col8, col9= st.columns(9)

    # col1, col2, col3, col4, col5,col6,col7 = st.columns(7)

    with col1:
        months =df['month'].unique()
        selected_month = st.multiselect('Month', months)
        df = df[df['month'].isin(selected_month)]

    with col2:
        dates =df['date'].unique()
        selected_date = st.multiselect('Date', dates ,key ='selected_month')
        df = df[df['date'].isin(selected_date)]

    with col3:
        hours =df['hour'].unique()
        selected_hour = st.multiselect('Hour', hours ,key ='selected_date')
        df = df[df['hour'].isin(selected_hour)]

    with col4:
        company_names =df['company_name'].unique()
        selected_company_name = st.multiselect('Company_Name', company_names, key = 'selected_hour')
        df = df[df['company_name'].isin(selected_company_name)]

    with col5:
        product_names =df['product_name'].unique()
        selected_product_name = st.multiselect('Product_Name', product_names, key ='selected_company_name' )
        df = df[df['product_name'].isin(selected_product_name)]

    with col6:
        sizes =df['size'].unique()
        selected_size = st.multiselect('Sizes', sizes, key = 'selected_product_name')
        df = df[df['size'].isin(selected_size)]

    with col7:
        config =df['config'].unique()
        selected_config = st.multiselect('Config', config, key = 'selected_size')
        df = df[df['config'].isin(selected_config)]
    

    with col8:
        pitch =df['pitch'].unique()
        selected_pitch = st.multiselect('Pitch', pitch, key = 'selected_config')
        df = df[df['pitch'].isin(selected_pitch)]

    with col9:
        voltage =sorted(df['voltage'].unique())
        selected_voltage = st.multiselect('Voltage', voltage, key = 'selected_pitch')
        df = df[df['voltage'].isin(selected_voltage)]


    ok =st.button("Enter")

    if ok:
        # d = df[(df['date'].isin(selected_date)) & (df['month'].isin(selected_month)) & (df['company_name'].isin(selected_company_name)) &
        #            (df['product_name'].isin(selected_product_name))&(df['size'].isin(selected_size)) & (df['pitch'].isin(selected_pitch)) & (df['config']).isin(selected_config)]
        st.dataframe(df)

    def download_csv():
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode() 
        href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
        return href

    # add a download button
    st.markdown(download_csv(), unsafe_allow_html=True)

        