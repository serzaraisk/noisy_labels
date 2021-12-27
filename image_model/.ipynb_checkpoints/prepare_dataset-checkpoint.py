from nile.api.v1 import (
    clusters,
    aggregators as na
)

import pandas as pd
import numpy as np

import requests as r
import os
import shutil
from sklearn.model_selection import train_test_split

from PIL import Image
import requests
from io import BytesIO

cluster = clusters.Hahn()


def load_dataset(path):
    image_class = cluster.read(path).as_dataframe()
    
    image_class['aggr_answer'] = image_class['aggr_answer'].str.decode('utf8')
    image_class['assessors_answer'] = image_class['assessors_answer'].str.decode('utf8')
    image_class['some_answer'] = image_class['some_answer'].str.decode('utf8')
    image_class['query'] = image_class['query'].str.decode('utf8')
    image_class = image_class.drop_duplicates(subset=['query'])
    return image_class


def sample_dataset(df, n=None, frac=None):   
    if frac:
        df = df.sample(frac=frac, random_state=1)
    elif n:
        df = df.sample(n=n, random_state=1)
    else:
        df = df
    df = df.reset_index()
    return df


def choose_toloka_answer_column(df, col_name):
    #create defined mapping
    map_ = {'BAD': 0,
           'OK': 1,
           'GOOD': 2}
    df['answer'] = df[col_name].apply(lambda x: map_[x])
    df['label'] = df['assessors_answer'].apply(lambda x: map_[x])
    df = df[['query', 'label', 'answer']]
    return df

def train_test_split_df(df, destination_folder, train_test_ratio, train_valid_ratio, seed=1):
        
    # Split according to label
    df_bad = df[df['label'] == 0]
    df_ok = df[df['label'] == 1]
    df_good = df[df['label'] == 2]
    
    # Train-test split
    df_bad_full_train, df_bad_test = train_test_split(df_bad, train_size = train_test_ratio, random_state = seed)
    df_ok_full_train, df_ok_test = train_test_split(df_ok, train_size = train_test_ratio, random_state = seed)
    df_good_full_train, df_good_test = train_test_split(df_good, train_size = train_test_ratio, random_state = seed)

    # Train-valid split
    df_bad_train, df_bad_valid = train_test_split(df_bad_full_train, train_size = train_valid_ratio, random_state = seed)
    df_ok_train, df_ok_valid = train_test_split(df_ok_full_train, train_size = train_valid_ratio, random_state = seed)
    df_good_train, df_good_valid = train_test_split(df_good_full_train, train_size = train_valid_ratio, random_state = seed)

    # Concatenate splits of different labels
    df_train = pd.concat([df_bad_train, df_ok_train, df_good_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_bad_valid, df_ok_valid, df_good_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_bad_test, df_ok_test, df_good_test], ignore_index=True, sort=False)

    # Write preprocessed data
    if os.path.isdir(destination_folder):
        shutil.rmtree(destination_folder)
    os.mkdir(destination_folder)
    df_train.to_csv(destination_folder + 'train.csv', index=False)
    df_valid.to_csv(destination_folder + 'valid.csv', index=False)
    df_test.to_csv(destination_folder + 'test.csv', index=False)
    

def get_lookup_tables(destination_folder):
    df_train = pd.read_csv(destination_folder + 'train.csv')
    df_test = pd.read_csv(destination_folder + 'test.csv')
    df_val = pd.read_csv(destination_folder + 'valid.csv')
    return df_train, df_val, df_test


def load_pictures(df_train, df_val, df_test):
    if os.path.exists('image_dataset'):
         shutil.rmtree('image_dataset')
            
    load_folder(df_train, '/train')
    load_folder(df_val, '/val')
    load_folder(df_test, '/test')


def load_folder(df, folder_name):
    if os.path.exists('image_dataset' + folder_name):
        shutil.rmtree('image_dataset' + folder_name)
    for row in df.iterrows():
        key = str(row[1]['label'])
        response = requests.get(row[1]['query'])
        try:
            img = Image.open(BytesIO(response.content))
            if not os.path.exists('image_dataset' + folder_name + '/' + key):
                os.makedirs('image_dataset' + folder_name + '/' + key)
            try:
                _ = np.array(img).shape[2]
                img.save('image_dataset' + folder_name + '/' + key + '/' + str(row[0])  +  '.jpg')
            except OSError:
                _ = np.array(img).shape[2]
                img = img.convert('RGB')
                img.save('image_dataset' + folder_name + '/' + key + '/' + str(row[0])  + '.jpg')
        except:
            print(row[1]['query'])
        if row[0] % 100 == 0:
            print(f'Кол-во обработанных строк [{folder_name}]: {str(row[0])}')