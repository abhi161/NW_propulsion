from model import models
from data_preprocessing import preprocess
from explore import plot_normal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import streamlit as st
import base64
import plotly.express as px
import itertools
import plotly.graph_objs as go
import numpy as np

def process():
    
    all_models = models.all_models()
    data_preprocess = preprocess.preprocess()


    logo_path = "incoming_data\logo with title.png"
    input_file_path = "incoming_data\inputfile.csv"

    st.image(logo_path,width=250)
    st.title('Propeller Performance prediction')

    data = pd.read_csv(input_file_path)
    df= data.copy()

    st.header("")
    st.text("")
    st.subheader("Make a selection to EXPLORE")

    container = st.container()
    with container:
        col1, col2, col3, col4, col5, col6, col7, col8, col9= st.columns(9)
        

    # col1, col2, col3, col4, col5,col6,col7 = st.columns(7)

    with col1:
        months =sorted(df['month'].unique())
        selected_month = st.multiselect('Month', months)
        df = df[df['month'].isin(selected_month)]

    with col2:
        dates =sorted(df['date'].unique())
        selected_date = st.multiselect('Date', dates ,key ='selected_month')
        df = df[df['date'].isin(selected_date)]
   
    with col3:
        company_names =sorted(df['company_name'].unique())
        selected_company_name = st.multiselect('Company_Name', company_names, key = 'selected_date')
        df = df[df['company_name'].isin(selected_company_name)]

    with col4:
        product_names =sorted(df['product_name'].unique())
        selected_product_name = st.multiselect('Product_Name', product_names, key ='selected_company_name' )
        df = df[df['product_name'].isin(selected_product_name)]

    with col5:
        sizes =sorted(df['size'].unique())
        selected_size = st.multiselect('Sizes', sizes, key = 'selected_product_name')
        df = df[df['size'].isin(selected_size)]

    with col6:
        config =sorted(df['config'].unique())
        selected_config = st.multiselect('Config', config, key = 'selected_size')
        df = df[df['config'].isin(selected_config)]
    
    with col7:
        pitch =sorted(df['pitch'].unique())
        selected_pitch = st.multiselect('Pitch', pitch, key = 'selected_config')
        df = df[df['pitch'].isin(selected_pitch)]

    with col8:
        voltage =sorted(df['voltage'].unique())
        selected_voltage = st.multiselect('Voltage', voltage, key = 'selected_pitch')
        df = df[df['voltage'].isin(selected_voltage)]

    with col9:
        hours =sorted(df['hour'].unique())
        selected_hour = st.multiselect('Hour', hours ,key ='selected_voltage')
        df = df[df['hour'].isin(selected_hour)]


    ok =st.button("Enter")

    df.loc[:,'Propeller_Mech_Efficiency_gfW'] = df['Propeller_Mech_Efficiency_gfW'].apply(lambda x: 0 if x >25 else x)
    df.loc[:,'Overall_Efficiency_gfW'] = df['Overall_Efficiency_gfW'].apply(lambda x: 0 if x >25 else x)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.reindex()

    if ok:
        # d = df[(df['date'].isin(selected_date)) & (df['month'].isin(selected_month)) & (df['company_name'].isin(selected_company_name)) &
        #            (df['product_name'].isin(selected_product_name))&(df['size'].isin(selected_size)) & (df['pitch'].isin(selected_pitch)) & (df['config']).isin(selected_config)]
        st.dataframe(df)
    
    
    columns = df.columns.values

    if selected_month and selected_date and selected_hour and selected_company_name and selected_product_name and selected_size and selected_config and selected_pitch and selected_voltage:
        st.write("")
        st.write("")
        container = st.container()
        with container:
            col1, col2, col3, col4, col5= st.columns(5)
        with col1:
            input_column_x = st.selectbox("Select input column ", columns)
        with col2:
            input_column_y = st.selectbox("Select column to be PREDICTED ", columns)
        with col3:
            min_value = int(st.number_input("Give min value for predicted column"))
        with col4:
            max_value = int(st.number_input("Give max value for predicted column"))
        with col5:
            steps = int(st.number_input("Give step size for predicted column"))

        combinations = list(itertools.product(df['size'].unique(), df['product_name'].unique(),df['config'].unique(),df['pitch'].unique()))
        df_new = pd.DataFrame()
        for i,j,k,l in combinations:
            match =df.loc[(df['size'] ==i) & (df['product_name'] == j) & (df['config'] ==k) & (df['pitch'] ==l)]
            if match.empty:
                print(f'not present {i}{j}{k}{l}')
            else:
    
                d = df[(df['product_name'].isin([j])) & (df['size'].isin([i])) & (df['config'].isin([k])) & (df['ESC_signal_s'] > 1000) & (df['pitch'].isin([l]))]
                
                d = d[[input_column_x,input_column_y]]
            
                x =d[input_column_x].values.reshape(-1,1) 
                y =d[input_column_y]
    
                x3,x2,x1,inter,polymodel = all_models.polynomial_model_train(3,x,y)

                #new data creation 
                new_df = data_preprocess.new_data_creation(df,input_column_x,i,j,k,l,min_value,max_value,steps)

                x_new =new_df[input_column_x].values.reshape(-1,1)
                predictions = all_models.predict(polymodel,x_new,3)
                new_df[input_column_y] =predictions
                df_new = pd.concat([df_new,new_df])

        st.write("")
        st.write("")
        st.write("")
        st.write("")

        container = st.container()
        with container:
            col1,col2 = st.columns(2,gap="medium")

        with col1: 
            st.dataframe(df_new)
        
        with col2:

            colors = px.colors.qualitative.T10
            # create a line plot with different colors for each line
            fig = go.Figure()
            for group in df_new.groupby(['size', 'product_name', 'config','pitch']):
                key, data = group
                color = colors[hash(str(key)) % len(colors)]
                y_max_avg = data.groupby(input_column_x)[input_column_y].max().mean() # calculate mean of max y values
                fig.add_trace(go.Scatter(x=data[input_column_x], y=data[input_column_y], mode='lines', name=f"{str(key)} (Avg. Max {y_max_avg:.2f})", line=dict(color=color)))

            # Set layout
            fig.update_layout(
                title=f"{input_column_x} vs {input_column_y}",
                xaxis_title=input_column_x,
                yaxis_title=input_column_y,
                legend_title="Grouping",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig)
    
        def download_csv():
            csv = df_new.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode() 
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
            return href

        # add a download button
        st.markdown(download_csv(), unsafe_allow_html=True)
    
    else:
        st.write("please select the options")
