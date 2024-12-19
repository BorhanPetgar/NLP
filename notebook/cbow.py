import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1)  # Average over context words, keep batch dimension
        print('#' * 30)
        print(f'embeds shape: {embeds.shape}, inputs shape: {inputs.shape}, self.embeddings shape: {self.embeddings(inputs).shape}, self.embeddings(inputs).mean(dim=1) shape: {self.embeddings(inputs).mean(dim=1).shape}')
        print(f'{inputs}')
        exit()
        out = torch.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out

def preprocess_text(text):
    text = text.lower().split()
    return text

def build_vocab(text, min_freq=1):
    word_counts = Counter(text)
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab['<UNK>'] = len(vocab)  # Add a token for unknown words
    return vocab

def create_context_target_pairs(text, vocab, context_size=2):
    data = []
    for i in range(context_size, len(text) - context_size):
        context = [text[i - context_size + j] for j in range(context_size)] + [text[i + j + 1] for j in range(context_size)]
        target = text[i]
        data.append((context, target))
    return data

def text_to_indices(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in text]

def prepare_data(text, context_size=2):
    text = preprocess_text(text)
    vocab = build_vocab(text)
    data = create_context_target_pairs(text, vocab, context_size)
    return data, vocab

# Example text for training
text = """
We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers.
As they evolve, they become more complex and capable of performing a wider range of tasks. The study of these processes is known as computer science.
Computer science is a field that encompasses both theoretical and practical aspects of computing. It involves the study of algorithms, data structures, and the principles of programming languages.
In addition to these core areas, computer science also includes topics such as artificial intelligence, machine learning, and data science.
These fields are rapidly evolving and have a significant impact on many aspects of our daily lives.
"""
data, vocab = prepare_data(text)
# print(data)
# print(vocab)
# exit()
vocab_size = len(vocab)

class CBOWDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_indices = torch.tensor(text_to_indices(context, self.vocab), dtype=torch.long)
        target_index = torch.tensor(self.vocab.get(target, self.vocab['<UNK>']), dtype=torch.long)
        return context_indices, target_index

dataset = CBOWDataset(data, vocab)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

embedding_dim = 10
model = CBOW(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 700
for epoch in range(epochs):
    total_loss = 0
    for context, target in dataloader:
        optimizer.zero_grad()
        print(target)
        output = model(context)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

def predict_center_word(model, context, vocab):
    context_indices = torch.tensor(text_to_indices(context, vocab), dtype=torch.long).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(context_indices)
    _, predicted_index = torch.max(output, 1)
    index_to_word = {i: word for word, i in vocab.items()}
    return index_to_word.get(predicted_index.item(), '<UNK>')

# Example test with a more generalized context
test_context = ["these", "fields", "rapidly", "growing"]
predicted_word = predict_center_word(model, test_context, vocab)
print(f"Predicted center word: {predicted_word}")