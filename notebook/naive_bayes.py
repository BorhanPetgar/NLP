import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Sample data: list of tuples containing tweets and their corresponding labels
tweets = [
    ("I love sunny days", "positive"),
    ("I hate rain", "negative"),
    ("Sunny days make me happy", "positive"),
    ("Rainy days make me sad", "negative"),
    ("I enjoy walking in the park", "positive"),
    ("I dislike traffic jams", "negative"),
    ("The weather is beautiful today", "positive"),
    ("I am feeling great", "positive"),
    ("I am not feeling well", "negative"),
    ("I love spending time with family", "positive"),
    ("I hate being stuck in traffic", "negative"),
    ("The food was delicious", "positive"),
    ("The service was terrible", "negative"),
    ("I am very happy with the results", "positive"),
    ("I am disappointed with the outcome", "negative"),
    ("I love my new job", "positive"),
    ("I hate my new job", "negative"),
    ("I am excited about the trip", "positive"),
    ("I am worried about the exam", "negative"),
    ("I enjoy reading books", "positive"),
    ("I dislike loud noises", "negative"),
    ("I am thrilled with the news", "positive"),
    ("I am upset about the delay", "negative"),
    ("I love playing sports", "positive"),
    ("I hate losing games", "negative")
]

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))  # Set of English stop words
stemmer = PorterStemmer()  # Initialize the Porter Stemmer
lemmatizer = WordNetLemmatizer()  # Initialize the WordNet Lemmatizer

# Preprocessing function
def preprocess_tweets(tweet):
    # Convert text to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Tokenize the tweet
    tweet_tokens = word_tokenize(tweet)
    # Remove stop words
    tweet_tokens = [word for word in tweet_tokens if word not in stop_words]
    # Apply stemming and lemmatization
    tweet_tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tweet_tokens]
    return tweet_tokens

# Preprocess each tweet in the dataset
processed_tweets = [(preprocess_tweets(tweet), label) for tweet, label in tweets]

# Calculate prior probabilities and likelihoods
class_counts = defaultdict(int)  # Dictionary to count occurrences of each class
word_counts = defaultdict(lambda: defaultdict(int))  # Nested dictionary to count word occurrences per class
total_words = defaultdict(int)  # Dictionary to count total words per class

# Count occurrences of classes and words
for words, label in processed_tweets:
    class_counts[label] += 1  # Increment class count
    for word in words:
        word_counts[label][word] += 1  # Increment word count for the class
        total_words[label] += 1  # Increment total word count for the class

# Calculate prior probabilities for each class
total_tweets = len(tweets)
priors = {label: count / total_tweets for label, count in class_counts.items()}

# Calculate likelihoods with Laplace smoothing
vocab = set(word for words, _ in processed_tweets for word in words)  # Vocabulary set
vocab_size = len(vocab)  # Size of the vocabulary
likelihoods = defaultdict(lambda: defaultdict(float))  # Nested dictionary for likelihoods

# Calculate likelihoods for each word in the vocabulary
for label, words in word_counts.items():
    for word in vocab:
        # Apply Laplace smoothing
        likelihoods[label][word] = (word_counts[label][word] + 1) / (total_words[label] + vocab_size)

# Naive Bayes classifier function
def classify(tweet):
    words = preprocess_tweets(tweet)  # Preprocess the input tweet
    log_probs = {label: log(prior) for label, prior in priors.items()}  # Initialize log probabilities with priors
    
    # Calculate log probabilities for each class
    for label in priors:
        for word in words:
            if word in vocab:  # Only consider words in the vocabulary
                log_probs[label] += log(likelihoods[label][word])
    
    # Debug print to understand log probabilities
    print(f"Tweet: '{tweet}'")
    print("Log probabilities:", log_probs)
    
    # Return the class with the highest log probability
    return max(log_probs, key=log_probs.get)

# Test the classifier with new tweets
new_tweets = [
    "I love rainy days",
    "Rainy days make me sad"
]

# Classify each new tweet and print the result
for tweet in new_tweets:
    print(f"Tweet: '{tweet}' -> Class: {classify(tweet)}")