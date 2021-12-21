import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
import shutil

def load_data(path, filter_language=False):
    df = pd.read_csv(path)
    df.drop(['Unnamed: 0', 'cnt'], axis=1, inplace=True)
    
    if filter_language:   
        df['english'] = df['query'].apply(lambda x: re.search('[a-zA-Z]', x) != None)
        df['russian'] = df['query'].apply(lambda x: re.search('[А-Яа-я]', x) != None)
        df['ukrain'] = df['query'].apply(lambda x: re.search('[ЇїІіЄєҐґ]', x) != None)
        df = df[(~df['english']) & (df['russian']) & (~df['ukrain'])]
    return df

def choose_toloka_answer_column(df, col_name):
    #create defined mapping
    map_ = {'NOT_COMMERCIAL': 0,
           'COMMERCIAL': 1}
    df['answer'] = df[col_name].apply(lambda x: map_[x])
    df['label'] = df['assessors_answer'].apply(lambda x: map_[x])
    df = df[['query', 'label', 'answer']]
    return df

def train_test_split_df(df, destination_folder, train_test_ratio, train_valid_ratio, seed=1):
        
    # Split according to label
    df_real = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]
    
    # Train-test split
    df_real_full_train, df_real_test = train_test_split(df_real, train_size = train_test_ratio, random_state = seed)
    df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = train_test_ratio, random_state = seed)

    # Train-valid split
    df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size = train_valid_ratio, random_state = seed)
    df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size = train_valid_ratio, random_state = seed)

    # Concatenate splits of different labels
    df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)
    df_all = pd.concat([df_real, df_fake], ignore_index=True, sort=False)

    # Write preprocessed data
    if os.path.isdir('data'):
        shutil.rmtree('data')
    os.mkdir('data')
    df_train.to_csv(destination_folder + 'train.csv', index=False)
    df_valid.to_csv(destination_folder + 'valid.csv', index=False)
    df_test.to_csv(destination_folder + 'test.csv', index=False)
    df_all.to_csv(destination_folder + 'all.csv', index=False)