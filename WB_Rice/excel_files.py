import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the directory containing excel_files.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
big_data_path = os.path.join(current_dir, 'West_bengail_Rice_2013-2023.xlsx')
sys.path.append(big_data_path)

# dir = path.Path(__file__).abspath()
# sys.path.append(dir.parent.parent)

# big_data_path = 'West_bengail_Rice_2013-2023.xlsx'

with open(big_data_path,'rb') as file:
    big_data = pd.read_excel(big_data_path)
    
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
    folder_path = os.path.join(current_dir, 'variety_13-23')
    sys.path.append(folder_path)
    
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
    model_folder = os.path.join(current_dir, 'models')
    sys.path.append(model_folder)
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

# df = models_data_frame()
# model_df = main_data_frame()







