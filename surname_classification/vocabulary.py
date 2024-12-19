import torch
from collections import Counter, defaultdict
import string
import pandas as pd


class Vocabulary:
    """Class to process text and extract vocabulary for mapping"""
    def __init__(self, special_tokens: list=None):
        """Initialize token-to-index and index-to-token mappings
        
        Args:
            special_tokens (list): A list of special tokens to add to the vocabulary
        """
        
        self._token_to_index = {}
        self._index_to_token = {}
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>']
        self._build_special_tokens()
        
    def _build_special_tokens(self):
        """Build the mappings for the special tokens in the vocabulary"""
        for token in self.special_tokens:
            self.add_token(token)
            
    def add_token(self, token: str):
        """Add a token to the vocabulary
        
        Args:
            token (str): the token to add
        """
        if token not in self._token_to_index:
            index = len(self._token_to_index)
            self._token_to_index[token] = index
            self._index_to_token[index] = token
    
    def add_many(self, tokens: list):
        """Add a list of tokens to the vocabulary
        
        Args:
            tokens (list): a list of tokens to add
        """
        for token in tokens:
            self.add_token(token)
    
    def build_vocab(self, corpus: list[str]):
        """Build the vocabulary from a list of sentences
        
        Args:
            corpus (list): a list of sentences
        """
        tokens = [token for sentence in corpus for token in sentence.split()]
        for token in tokens:
            self.add_token(token)
            
    def lookup_token(self, token: str):
        """Retrieve the index for the token
        
        Args:
            token (str): the token to look up
        Returns:
            int: the index of the token
        """
        return self._token_to_index.get(token, self._token_to_index['<UNK>'])
    
    def lookup_index(self, index: int):
        """Return the token for the index
        
        Args:
            index (int): the index to look up
        Returns:
            str: the token at the index
        """
        assert index < len(self._index_to_token), f"Index {index} out of vocabulary"
        assert type(index) == int, f'Index must be an integer, not {type(index)}'
        
        return self._index_to_token.get(index, '<UNK>')
    
    def __len__(self):
        return len(self._token_to_index)
    
    def to_serializable(self):
        """Return a serializable dictionary for the vocabulary"""
        return {'token_to_index': self._token_to_index,
                'index_to_token': self._index_to_token,
                'special_tokens': self.special_tokens}
        
    @classmethod
    def from_serializable(cls, contents):
        """Create a Vocabulary instance from a serializable dictionary
        
        Args:
            contents (dict): a serializable dictionary
        """
        return cls(**contents)
    
    def __repr__(self):
        return f'<Vocabulary(size={len(self)})>'
    

if __name__ == '__main__':
    """Example usage of the Vocabulary class on Surname dataset"""
    df = pd.read_csv("hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz")
    surnames = list(df['surname'])
    # Create a vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(surnames)
    print(vocab)
    print(vocab.lookup_token('Smith'))
    print(vocab.lookup_index(0))
    print(vocab.lookup_index(1))
    print(vocab.lookup_index(2))
    