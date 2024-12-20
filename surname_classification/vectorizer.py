import string
from vocabulary import Vocabulary
import numpy as np


class SurnameVectorizer(object):
    """Class to vectorize surnames and languages"""
    def __init__(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary):
        """
        Args:
            surname_vocab (Vocabulary): the vocabulary for the surnames
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        
    def vectorize(self, surname: str):
        """Create the vector for the surname
        
        Args:
            surname (str): the surname
        Returns:
            one_hot (np.ndarray): a character level one-hot encoding of the surname
        """
        
        # Create a vector of zeros the length of the vocabulary,
        # the vocabulary is the number of unique characters in the surname because
        # each item in our corpus is a word thus the vocabulary is the number
        # of unique characters in the word.
        one_hot = np.zeros(len(self.surname_vocab), dtype=np.float32)
        
        for token in surname:
            one_hot[self.surname_vocab.lookup_token(token)] = 1
            
        return one_hot
        
    @classmethod
    def from_dataframe(cls, surname_df):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            surname_df (pandas.DataFrame): the surnames dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        surname_vocab = Vocabulary()
        nationality_vocab = Vocabulary()
        
        for index, row in surname_df.iterrows():
            for letter in row.surname:
                if letter not in string.punctuation:
                    surname_vocab.add_token(letter.lower())
            nationality_vocab.add_token(row.nationality)
            
        return cls(surname_vocab, nationality_vocab)
        
    def unvectorize(self, one_hot):
        """Return the string from the one-hot encoding
        
        Args:
            one_hot (np.ndarray): the one-hot encoding
        Returns:
            str: the decoded surname
        """
        return "".join([self.surname_vocab.lookup_index(index.item()) for index in np.where(one_hot == 1)[0]])
    
if __name__ == '__main__':
    """Example usage of the SurnameVectorizer class"""
    import pandas as pd
    df = pd.read_csv("hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz")
    # Create the vectorizer
    vectorizer = SurnameVectorizer.from_dataframe(df)
    print(vectorizer.surname_vocab)
    # token_to_index
    print(vectorizer.surname_vocab._token_to_index)
    # vectorize a surname
    print(vectorizer.vectorize("Smith"))
    # unvectorize a surname
    # create a one-hot encoding first
    one_hot = np.zeros(len(vectorizer.surname_vocab), dtype=np.float32)
    for letter in 'borhan': 
        one_hot[vectorizer.surname_vocab.lookup_token(letter)] = 1
    print(f'Unvectorized surname: {vectorizer.unvectorize(one_hot)}') # -> aohbrn beacuse the order of the letters is not preserved