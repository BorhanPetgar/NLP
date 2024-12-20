"""
Surname dataset class for loading data
"""
from random import sample
from re import X
import torch
from torch.utils.data import Dataset
from vectorizer import SurnameVectorizer
from vocabulary import Vocabulary
from preprocess import split_data, sample_from_countries
import pandas as pd


class SurnameDataset(Dataset):
    def __init__(self, split):
        df = pd.read_csv("hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz")
        sampled_df = sample_from_countries(df, n_samples=20000, min_samples=100)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(sampled_df)
        self.vectorizer = SurnameVectorizer.from_dataframe(sampled_df)
        self.split = {
            'train': (self.X_train, self.y_train),
            'val': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
        self.set_split(split)
        self.build_country_dict(sampled_df)
        
            
    
    def build_country_dict(self, df):
        self.country_dict = {}
        countries = df['nationality'].unique()
        index = 0
        for country in countries:
            self.country_dict[country] = index
            index += 1
        
        
    def lookup_country(self, country):
        return self.country_dict[country]
    
    def set_split(self, split):
        self.X, self.y = self.split[split]
        self.split_size = len(self.y)
    
    def __getitem__(self, index):
        if self.split == 'train':
            vectorized_surname = self.vectorizer.vectorize(self.X_train.iloc[index])
            nationality = self.y_train.iloc[index]
        elif self.split == 'val':
            vectorized_surname = self.vectorizer.vectorize(self.X_val.iloc[index])
            nationality = self.y_val.iloc[index]
        else:
            vectorized_surname = self.vectorizer.vectorize(self.X_test.iloc[index])
            nationality = self.y_test.iloc[index]
        return vectorized_surname, nationality
    
    def __len__(self):
        return len(self.vectorizer.surname_vocab)
    

if __name__ == '__main__':
    # test the dataset
    dataset = SurnameDataset()
    print(dataset.X.iloc[0])
    print(dataset.y.iloc[0])
    # print(dataset[0])
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    for data in dataloader:
        print(data)
        break
    
    

    
        
        
    
