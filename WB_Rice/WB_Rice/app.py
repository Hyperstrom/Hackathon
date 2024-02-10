import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from test import *
from excel_files import *

import matplotlib.pyplot as plt
import seaborn as sns

# Load your pre-trained model and necessary data here

st.set_page_config(layout="wide")


# Streamlit app
def main():
    st.title('West Bengal Rice Price Prediction :ear_of_rice:')

    col1, col2 = st.columns([1, 2])
    
    with col1: 
        # Input fields
        district = st.selectbox('Select District', list(district_market_variety_mapping.keys()))
        markets = district_market_variety_mapping.get(district, {})
        market = st.selectbox('Market Name', markets)
        varieties = district_market_variety_mapping[district][market]
        variety = st.selectbox('Variety Type', varieties)
        days = st.slider('Number of Predicted Days', min_value=1, max_value=30, value=7)
        if st.button('Predict'):
            prediction_df, price_min_last, price_max_last, price_mod_last = predictions(market, variety, days)
            st.write("Today's Min Price (Rs./Quintal) : "+str(price_min_last))
            st.write("Today's Max Price (Rs./Quintal) : "+str(price_max_last))
            st.write("Today's Mod Price (Rs./Quintal) : "+str(price_mod_last))


    with col2: 
        # Predict and display results
        if 'prediction_df' in locals():
            # Check if prediction_df is not None
            if prediction_df is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(str(len(prediction_df))+" days prediction ")
                    st.write(prediction_df)
                # Create the plot
                with col2:
                    st.subheader(str(len(prediction_df))+" days prediction plot ")
                    fig, ax = plt.subplots(figsize=(6, 4))

                    # Set the x-axis labels to be vertical
                    ax.tick_params(axis='x', rotation=90)

                    # Plot the data
                    sns.lineplot(data=prediction_df, x="Dates", y="Min Price", label="Min Price", ax=ax)
                    sns.lineplot(data=prediction_df, x="Dates", y="Max Price", label="Max Price", ax=ax)
                    sns.lineplot(data=prediction_df, x="Dates", y="Modal Price", label="Modal Price", ax=ax)
                    ax.set_ylabel("Price")
                    # Display Matplotlib plot in Streamlit
                    st.pyplot(plt)
                
if __name__ == "__main__":
    main()
