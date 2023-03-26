import numpy as np
import pandas as pd


class preprocess:

    def __init__(self):
        pass

    def new_data_creation(self,df,column_name,i,j,k,l,min_value, max_value,steps):

        # min_angular_vel =min(df[column_name])
        # max_angular_vel =max(df[column_name])

        min = min_value
        max = max_value

        value_list = [t for t in range(min,max,steps)]


        # Create a list of tuples containing all combinations of sizes and angular velocity
        data = []
        data = [(i,j,k,l, v) for v in value_list]

        # Create a DataFrame from the list of tuples
        new_df = pd.DataFrame(data, columns=['size','product_name','config','pitch',column_name])

        return new_df

        