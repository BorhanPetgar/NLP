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
    def __init__(self):
        df = pd.read_csv("hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz")
        sampled_df = sample_from_countries(df, n_samples=1000)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sampled_df)
        self.vectorizer = SurnameVectorizer.from_dataframe(sampled_df)
        self.split = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        self.set_split('train')
    
    def set_split(self, split):
        self.X, self.y = self.split[split]
        self.split_size = len(self.y)
    
    def __getitem__(self, index):

        vectorized_surname = self.vectorizer.vectorize(self.X.iloc[index])
        nationality = self.y.iloc[index]
        return vectorized_surname, nationality
    
    def __len__(self):
        return self.split_size
    

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
        
        
    
