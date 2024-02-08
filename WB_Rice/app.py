import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from test import *
from excel_files import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


import matplotlib.pyplot as plt
import seaborn as sns

# Load your pre-trained model and necessary data here
big_data = pd.read_excel('West_bengail_Rice_2013-2023.xlsx')
columns_dtype = {'Sl no.':int,
                  'Min Price (Rs./Quintal)':int,
                  'Max Price (Rs./Quintal)':int,
                  'Modal Price (Rs./Quintal)':int,
                 }
big_data = big_data.astype(columns_dtype)

#all District names
dist_name = big_data['District Name'].unique()
dist_name =  list(dist_name)
# all market names 
market_name = big_data['Market Name'].unique()
market_name = list(market_name)

#make a mapping between district , market , variety 
district_market_variety_mapping = {}

for district in dist_name:
  district_market_variety_mapping[district] = {}
  for market in big_data[big_data['District Name'] == district]['Market Name'].unique():
    district_market_variety_mapping[district][market] = big_data[(big_data['District Name'] == district) & (big_data['Market Name'] == market)]['Variety'].unique().tolist()
district_market_variety_mapping['Medinipur(E)'].pop('Egra/contai')


def main_data_frame():
    folder_path = 'variety_13-23'

    index =[]
    market_no = []
    variety_no = []
    market =[]
    variety = []
    file_name =[]
    for file in os.listdir(folder_path):
        index.append(str(file.split('_')[0])+'_'+str(file.split('_')[1]))
        market_no.append(int(file.split('_')[0]))
        variety_no.append(int(file.split('_')[1]))
        market.append(str(file.split('_')[5]))
        variety.append(str(file.split('_')[6]).replace('.xlsx', ''))
        file_name.append(file)
    
    df = pd.DataFrame({'index':index,
                    'market_no':market_no,
                    'variety_no':variety_no,
                    'market':market,
                    'variety':variety,
                    'file':file_name})

    return df

def models_data_frame(): 
    model_folder = 'models'

    models = []
    market_no = []
    variety_no =[]
    price_no = []
    for file in os.listdir(model_folder):
        market_no.append(int(file.split('_')[1]))
        variety_no.append(int(file.split('_')[2]))
        price_no.append(str(file.split('_')[4]).replace('.h5',''))
        models.append(file)

    model_df = pd.DataFrame({'market_no' :market_no,
                            'variety_no': variety_no,
                            'price_no': price_no,
                            'model': models})


    return model_df


df = main_data_frame()
model_df = models_data_frame()


#create a dictionary to map market name and market no
market_dict = {}
for i in range(df.shape[0]):
  market_dict[df['market'].iloc[i]] = df['market_no'].iloc[i]

#create a dictionary to map variety name and variety no
variety_dict = {}
for i in range(df.shape[0]):
  variety_dict[df['variety'].iloc[i]] = df['variety_no'].iloc[i]

def find_market_no(market_name):
  return market_dict[market_name]

def find_variety_no(variety_name):
  return variety_dict[variety_name]

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
sc = MinMaxScaler(feature_range=(0, 1))

def test_data(data):
  x = []
  seq = 45
  for i in range(seq, len(data)):
    x.append(data[i-seq:i])

  x = np.array(x)
  return x
def predictions( market , variety , num):
  market = market
  market_no = find_market_no(market)
  variety = variety
  variety_no = find_variety_no(variety)
  token = num 

  model = list(model_df[ (model_df['market_no'] == market_no) & (model_df['variety_no'] == variety_no)]['model'].values)

  model_path = 'models'
  folder_path = 'variety_13-23'

  model_path_1 = os.path.join(model_path, model[0])
  model_path_2 = os.path.join(model_path, model[1])
  model_path_3 = os.path.join(model_path, model[2])

  model_min  = load_model(model_path_1)
  model_max  = load_model(model_path_2)
  model_mod  = load_model(model_path_3)

  xl_file_path = str(df[(df['market_no'] == market_no)&(df['variety_no']==variety_no)]['file'].values)[2:-2]
  xl_file = os.path.join(folder_path, xl_file_path)

  xl_file = pd.read_excel(xl_file)

  price = xl_file[['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']].values

  price_min = price[:,0]
  price_max = price[:,1]
  price_mod = price[:,2]

  price_min_last = int(price_min[-1: ])
  price_max_last = int(price_max[-1: ])
  price_mod_last = int(price_mod[-1: ])

  y_pred_min_list =np.array([])
  y_pred_max_list =np.array([])
  y_pred_mod_list =np.array([])

  for i in range(token):

    price_min = price_min.reshape(-1,)
    price_max = price_max.reshape(-1,)
    price_mod = price_mod.reshape(-1,)
    price_min = np.concatenate((price_min, y_pred_min_list))
    price_max = np.concatenate((price_max, y_pred_max_list))
    price_mod = np.concatenate((price_mod, y_pred_mod_list))

    test_price_min = price_min[-60:]
    test_price_max = price_max[-60:]
    test_price_mod = price_mod[-60:]

    test_price_min = test_data(test_price_min)
    test_price_max = test_data(test_price_max)
    test_price_mod = test_data(test_price_mod)

    test_price_min = sc.fit_transform(test_price_min)
    test_price_max = sc.fit_transform(test_price_max)
    test_price_mod = sc.fit_transform(test_price_mod)

    min_prediction = model_min.predict(test_price_min)
    max_prediction = model_max.predict(test_price_max)
    mod_prediction = model_mod.predict(test_price_mod)

    price_min = price_min.reshape(-1,1)
    sc.fit_transform(price_min)
    min_prediction = min_prediction.reshape(-1, 1)
    y_pred_min = sc.inverse_transform(min_prediction)
    y1 = int(y_pred_min[-1 :])
    y1 = np.array([y1])
    y_pred_min_list = np.concatenate((y_pred_min_list, y1))

    price_max = price_max.reshape(-1,1)
    sc.fit_transform(price_max)
    max_prediction = max_prediction.reshape(-1, 1)
    y_pred_max = sc.inverse_transform(max_prediction)
    y2 = int(y_pred_max[-1 :])
    y2 = np.array([y2])
    y_pred_max_list = np.concatenate((y_pred_max_list, y2))

    price_mod = price_mod.reshape(-1,1)
    sc.fit_transform(price_mod)
    mod_prediction = mod_prediction.reshape(-1, 1)
    y_pred_mod = sc.inverse_transform(mod_prediction)
    y3 = int(y_pred_mod[-1 :])
    y3 = np.array([y3])
    y_pred_mod_list = np.concatenate((y_pred_mod_list, y3))

  dates = []
  start_date = datetime.date(2024, 1, 1)
  end_date = datetime.date(2024, 1, token)
  for i in range((end_date - start_date).days + 1):
      date = start_date + datetime.timedelta(days=i)
      dates.append(date.strftime("%d/%m/%Y"))

  # create a data frame
  predicted_df = pd.DataFrame({
      "Dates": dates,
      "Min Price": y_pred_min_list,
      "Max Price": y_pred_max_list,
      "Modal Price": y_pred_mod_list
  })
  
  return predicted_df , price_min_last , price_max_last , price_mod_last
# Sample function to get markets and varieties based on district
def get_markets_and_varieties(district):
    if district in district_market_variety_mapping:
        markets = list(district_market_variety_mapping[district].keys())
        return markets, []
    else:
        return [], []


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
