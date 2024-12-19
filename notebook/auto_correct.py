import re
from collections import Counter

# Sample corpus
corpus = """
This is a simple example corpus. It contains a few sentences with some common words.
In a real-world scenario, you would use a much larger corpus to build a more accurate model.
"""

# Preprocess the corpus
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Tokenize
    return words

# Build a frequency dictionary
words = preprocess(corpus)
word_freq = Counter(words)
vocab = set(words)

# Generate candidate corrections
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def known(words):
    return set(w for w in words if w in vocab)

# Rank the candidates
def candidates(word):
    return (known([word]) or known(edits1(word)) or [word])

def correct(word):
    return max(candidates(word), key=word_freq.get)

# Test the auto-correct system
misspelled_words = ["exampel", "contans", "sentece", "woudl"]

for word in misspelled_words:
    print(f"Original: {word} -> Corrected: {correct(word)}")