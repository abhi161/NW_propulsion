import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import itertools
import plotly.graph_objs as go
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)

def show_explore_page():

    logo_path = "incoming_data\logo with title.png"
    input_file_path = "incoming_data\inputfile.csv"

    st.image(logo_path,width=250)
    st.title('NW Propulsion Visualization')

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
        hours =sorted(df['hour'].unique())
        selected_hour = st.multiselect('Hour', hours ,key ='selected_date')
        df = df[df['hour'].isin(selected_hour)]

    with col4:
        company_names =sorted(df['company_name'].unique())
        selected_company_name = st.multiselect('Company_Name', company_names, key = 'selected_hour')
        df = df[df['company_name'].isin(selected_company_name)]

    with col5:
        product_names =sorted(df['product_name'].unique())
        selected_product_name = st.multiselect('Product_Name', product_names, key ='selected_company_name' )
        df = df[df['product_name'].isin(selected_product_name)]

    with col6:
        sizes =sorted(df['size'].unique())
        selected_size = st.multiselect('Sizes', sizes, key = 'selected_product_name')
        df = df[df['size'].isin(selected_size)]

    with col7:
        config =sorted(df['config'].unique())
        selected_config = st.multiselect('Config', config, key = 'selected_size')
        df = df[df['config'].isin(selected_config)]
    
    with col8:
        pitch =sorted(df['pitch'].unique())
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
    

    df.loc[:,'Propeller_Mech_Efficiency_gfW'] = df['Propeller_Mech_Efficiency_gfW'].apply(lambda x: 0 if x >55 else x)
    df.loc[:,'Overall_Efficiency_gfW'] = df['Overall_Efficiency_gfW'].apply(lambda x: 0 if x >55 else x)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    columns = df.columns.values

    if selected_month and selected_date and selected_hour and selected_company_name and selected_product_name and selected_size and selected_config and selected_pitch and selected_voltage:
        st.write("")
        st.write("")
        x_col = st.selectbox("Select X-axis", columns)
        y_col = st.selectbox("Select Y-axis", columns)

        # Get column values
        x = df[x_col]
        y = df[y_col]
        if x_col and y_col:
            # Filter data based on selected ranges for x and y axes
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_range = st.slider("Select range for X-axis", float(x_min), float(x_max), (float(x_min), float(x_max)))
            y_range = st.slider("Select range for Y-axis", float(y_min), float(y_max), (float(y_min), float(y_max)))
            df_filtered = df[(x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])]

            # Add interactive feature to select line or scatter plot
            plot_type = st.selectbox("Select plot type", ["Line", "Scatter"])
            color_scheme = st.selectbox("Select color scheme", ["Qualitative", "Viridis", "Blues"])

            if st.button('Show Plot'):
                plot(plot_type,df_filtered,x_col,y_col,x_range,y_range,color_scheme)


        x =st.button("Select to get all plots")
        if x:

            container2 = st.container()
            with container2:
                col1,col2,col3 = st.columns(3,gap="medium")

            with col1:
                x_col = "Thrust_gf"
                y_col = "Propeller_Mech_Efficiency_gfW"

                df_filtered = df
            
                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)

            with col2:
                x_col = "Motor_Optical_Speed_RPM"
                y_col = "Thrust_gf"
            
                df_filtered = df
            
                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)

            with col3:

                x_col = "Motor_Optical_Speed_RPM"
                y_col = "Torque_Nm"

                df_filtered = df

                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)

            st.write("")

            container3 = st.container()
            with container3:
                col1,col2,col3 = st.columns(3,gap="medium")


            with col1:
                x_col = "Thrust_gf"
                y_col = "Motor_Efficiency_"
            
                df_filtered = df
            
                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)
            
            with col2:
                x_col = "Torque_Nm"
                y_col = "Motor_Efficiency_"
            
                df_filtered = df
            
                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)
            
            with col3:
                
                x_col = "Thrust_gf"
                y_col = "Overall_Efficiency_gfW"
            
                df_filtered = df
            
                plot_normal("Line",df_filtered,x_col,y_col,color_scheme)


    else:
        st.write("Please select the configuration")

            
            
        



def plot(plot_type,df_filtered,x_col,y_col,x_range,y_range,color_scheme):

    if plot_type == "Line":

        # create a list of colors for each unique value
        if color_scheme == "Qualitative":
            colors = px.colors.qualitative.T10
        elif color_scheme == "Viridis":
            colors = px.colors.sequential.Viridis
        else:
            colors = px.colors.sequential.Blues


        # create a line plot with different colors for each line
        fig = go.Figure()
        for group in df_filtered.groupby(['company_name', 'product_name', 'pitch', 'config', 'hour']):
            key, data = group
            color = colors[hash(str(key)) % len(colors)]
            y_max_avg = data.groupby(x_col)[y_col].max().mean() # calculate mean of max y values
            fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='lines', name=f"{str(key)} (Avg. Max {y_max_avg:.2f})", line=dict(color=color)))

        # Set layout
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Grouping",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    else:
        # create a scatter plot with color-coded points
        fig = px.scatter(df_filtered, x=x_col, y=y_col, color='company_name', hover_name='product_name')

        # Set layout
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Company Name",
        )

    # Display plot
    st.plotly_chart(fig)

    line_values = {}

    for group in df_filtered.groupby(['company_name', 'product_name', 'pitch', 'config','hour']):
        key, data = group
        x_data = data[x_col]
        y_data = data[y_col]
        mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
        if mask.any():
            line_values[str(key)] = y_data[mask].max()

    if len(line_values) > 0:
        # Get the line with the highest value within the selected x range
        best_line = max(line_values, key=line_values.get)

        # Display explanation of the plot
        st.write(f"The plot above shows the {x_col} vs {y_col} for different configurations. The lines represent different combinations of company name, product name, pitch, config, and hour. The line with the highest average value for **{y_col}** in the selected **{x_col}** range is **{best_line}**.")
    else:
        st.write("No data within the selected X range.")



def plot_normal(plot_type,df_filtered,x_col,y_col,color_scheme):

    if plot_type == "Line":
        # create a list of colors for each unique value
        if color_scheme == "Qualitative":
            colors = px.colors.qualitative.T10
        elif color_scheme == "Viridis":
            colors = px.colors.sequential.Viridis
        else:
            colors = px.colors.sequential.Blues

        # create a line plot with different colors for each line
        fig = go.Figure()
        for group in df_filtered.groupby(['company_name', 'product_name', 'pitch', 'config', 'hour']):
            key, data = group
            color = colors[hash(str(key)) % len(colors)]
            y_max_avg = data.groupby(x_col)[y_col].max().mean() # calculate mean of max y values
            fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='lines', name=f"{str(key)} (Avg. Max {y_max_avg:.2f})", line=dict(color=color)))

        # Set layout
        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Grouping",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    st.plotly_chart(fig)



               # Add interactive feature to change marker size
        # marker_size = st.slider("Select marker size", 1, 10, 5)
        
        # if st.button('Show Plot'):
        #     if plot_type == "Line":
        #         # create a list of colors for each unique value
        #         colors = px.colors.qualitative.T10

        #         # create a line plot with different colors for each line
        #         fig = go.Figure()
        #         for group in df_filtered.groupby(['company_name', 'product_name', 'pitch', 'config','hour']):
        #             key, data = group
        #             color = colors[hash(str(key)) % len(colors)]
        #             fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='lines', name=str(key), line=dict(color=color)))

        #         # Set layout
        #         fig.update_layout(
        #             title=f"{x_col} vs {y_col}",
        #             xaxis_title=x_col,
        #             yaxis_title=y_col,
        #             legend_title="Grouping",
        #             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        #         )

        #     else:
        #         # create a scatter plot with color-coded points
        #         fig = px.scatter(df_filtered, x=x_col, y=y_col, color='company_name', hover_name='product_name')

        #         # Set layout
        #         fig.update_layout(
        #             title=f"{x_col} vs {y_col}",
        #             xaxis_title=x_col,
        #             yaxis_title=y_col,
        #             legend_title="Company Name",
        #         )


        #     # fig.update_traces(marker=dict(size=marker_size))
        #     # Display plot
        #     st.plotly_chart(fig)




    # df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # columns =df.columns.values

    # if selected_month and selected_date and selected_hour and selected_company_name and selected_product_name and selected_size and selected_config and selected_pitch :

        
    #     x_col = st.selectbox("Select X-axis", columns)
    #     y_col = st.selectbox("Select Y-axis", columns)

    #     # Get column values
    #     x = df[x_col]
    #     y = df[y_col]

    #     # if x_col and y_col:
    #     # Filter data based on selected ranges for x and y axes
    #     x_min, x_max = x.min(), x.max()
    #     y_min, y_max = y.min(), y.max()
    #     x_range = st.slider("Select range for X-axis", float(x_min), float(x_max), (float(x_min), float(x_max)))
    #     y_range = st.slider("Select range for Y-axis", float(y_min), float(y_max), (float(y_min), float(y_max)))
    #     df_filtered = df[(x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])]

    #     if st.button('Show Plot'):
    #         # create a list of colors for each unique value
    #         colors = px.colors.qualitative.T10

    #         # create a line plot with different colors for each line
    #         fig = go.Figure()
    #         for group in df_filtered.groupby(['company_name', 'product_name', 'pitch', 'config','hour']):
    #             key, data = group
    #             color = colors[hash(str(key)) % len(colors)]
    #             fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='lines', name=str(key), line=dict(color=color)))

    #         # Set layout
    #         fig.update_layout(
    #             title=f"{x_col} vs {y_col}",
    #             xaxis_title=x_col,
    #             yaxis_title=y_col,
    #             legend_title="Grouping",
    #             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    #         )

    #         # Display plot
    #         st.plotly_chart(fig)

    # else:
    #     st.write("Please select the configuration") 



   # x =st.selectbox('X_axis', columns)
    # y =st.selectbox('Y-axis',columns)


    # if st.button('Show Plot'):

        # df = df[(df['Overall_Efficiency_gfW'] < 60) & (df["Propeller_Mech_Efficiency_gfW"] <60)]
        # # create a dictionary to map unique values of Company Name, Config, Hour, Pitch, and Product Name to different integers
        # unique_vals = {}
        # for i, val in enumerate(itertools.product(df['company_name'].unique(), df['product_name'].unique(), df['pitch'].unique(), df['config'].unique(),df['hour'].unique())):
        #     unique_vals[val] = i
        # # create a list of colors for each unique value
        # colors = [plt.cm.tab10(unique_vals[val] % 10) for val in unique_vals]

        # # create a line plot with different colors for each line
        # fig, ax = plt.subplots()
        # for group in df.groupby(['company_name', 'product_name', 'pitch', 'config','hour']):
        #     key, data = group
        #     color = colors[unique_vals[key] % 10]
        #     ax.plot(data[x], data[y], color=color, label=key)

        # # add labels and title
        # ax.set_xlabel(x)
        # ax.set_ylabel(y)
        # ax.set_title(f"{x} vs {y}")

        # # add a legend
        # ax.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

        # # display the plot
        # st.pyplot(fig)

        # st.balloons()




    # # filter data
    # df = df[df['Overall_Efficiency_gfW'] < 60]

    # # create a dictionary to map unique values of Company Name, Config, Hour, Pitch, and Product Name to different integers
    # unique_vals = {}
    # for i, val in enumerate(itertools.product(df['company_name'].unique(), df['product_name'].unique(), df['pitch'].unique(), df['config'].unique())):
    #     unique_vals[val] = i

    # # create a list of colors for each unique value
    # colors = [plt.cm.tab10(unique_vals[val] % 10) for val in unique_vals]

    # # create a scatter plot with different colors for each line
    # fig, ax = plt.subplots()
    # ax.scatter(df['Thrust_gf'], df['Overall_Efficiency_gfW'], c=[colors[unique_vals[tuple(row)]] for row in df[['company_name', 'product_name', 'pitch', 'config']].to_records(index=False)])

    # # add labels and title
    # ax.set_xlabel('Thrust (gf)')
    # ax.set_ylabel('Overall Efficiency (gf/W)')
    # ax.set_title('Thrust vs Overall Efficiency')

    # # add a legend with unique values of Company Name, Config, Hour, Pitch, and Product Name
    # handles = []
    # for val, idx in unique_vals.items():
    #     handles.append(ax.scatter([], [], c=colors[idx % 10], label=val))
    # ax.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    # # display the plot
    # st.pyplot(fig)
    # filter data