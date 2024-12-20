from os import close, replace
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json


def plot_countries_distribution(df: pd.DataFrame) -> dict:
    """ 
    Plot the distribution of countries in the dataset
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        dict: A dictionary with the counts of each country
    """
    countries = list(df['nationality'].unique())

    country_dict = dict()
    for country in countries:
        country_dict[country] = len(df[df['nationality']==country])

    sns.set_style('darkgrid')
    plt.figure(figsize=(25, 10))
    sns.barplot(country_dict)
    plt.xticks(rotation=90)
    plt.savefig("countries_distribution.png")
    with open("countries.json", "w") as f:
        country_dict_alpha = {k: v for k, v in sorted(  country_dict.items(),
                                                        key=lambda item: item[0])}
        country_dict_total = {k: v for k, v in sorted(  country_dict.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True)}
        final_dict = {
            "country_dict_alpha": country_dict_alpha,
            "country_dict_total": country_dict_total
        }
        json.dump(final_dict, f, indent=4)
        
    return country_dict
    
def sample_from_countries(
    df: pd.DataFrame,
    n_samples: int,
    min_samples: int = 500,
    ) -> pd.DataFrame:
    """ 
    Sample the dataset from the countries with more than min_samples and given n_samples
    
    Args:
        df (pd.DataFrame): The dataset
        n_samples (int): The number of samples to get
        min_samples (int): The minimum number of samples to consider a country
    Returns:
        pd.DataFrame: A sampled dataset
    """
    countries = list(df['nationality'].unique())
    
    # Remove countries with less than min_samples and sample n_samples from the rest
    sampled_df = pd.DataFrame()
    for country in countries:
        nationality_count = len(df[df['nationality']==country])
        if  nationality_count > min_samples:
            if nationality_count < n_samples:
                new_n_sample = nationality_count
                sampled_df = pd.concat([sampled_df, df[df['nationality']==country].sample(new_n_sample, replace=False)])
            else:
                sampled_df = pd.concat([sampled_df, df[df['nationality']==country].sample(n_samples, replace=False)])
    return sampled_df
    
def split_data(df: pd.DataFrame,
               train_size: float = 0.6,
               val_size: float = 0.2,
               test_size: float = 0.2
               ) -> tuple:
    """This function splits the dataset into train, validation and test sets
    
    Args:
        df (pd.DataFrame): The dataset
        train_size (float): The size of the train set
        val_size (float): The size of the validation set
        test_size (float): The size of the test set
        
    Returns:
        tuple: A tuple with the train, validation and test sets
    """
    X = df['surname']
    y = df['nationality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size/(test_size+val_size))
    return X_train, X_val, X_test, y_train, y_val, y_test
    
if __name__ == "__main__":
    
    df = pd.read_csv("hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz")
    # countries_dict = plot_countries_distribution(df)
    sampled_df = sample_from_countries(df, n_samples=1200, min_samples=600)
    print(sampled_df['nationality'].value_counts())
    
    

