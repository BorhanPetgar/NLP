from matplotlib import axes
from seaborn import rugplot
import torch
from torch import float16, nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from argparse import Namespace
import os
from surname_classifier import SurnameClassifier
from surname_dataset import SurnameDataset
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.dataset = SurnameDataset('train')  # Initialize self.dataset here
        self.model = self.build_model(args).to(self.device)
        self.optimizer = self.build_optimizer(self.model)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.dataset.set_split('train')
        self.train_dataset = SurnameDataset('train')
        
        self.dataset.set_split('val')
        self.val_dataset = SurnameDataset('val')
        
        self.dataset.set_split('test')
        self.test_dataset = SurnameDataset('test')
        print(f'########### train size: {len(self.train_dataset.y)}, val size: {len(self.val_dataset.y)}, test size: {len(self.test_dataset.y)}')
        
        
    def compute_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute the accuracy of the model
        
        Args:
            y_pred (torch.Tensor): The predicted values
            y_true (torch.Tensor): The true values
        Returns:
            float: The accuracy of the model"""
        y_pred_indices = y_pred.argmax(dim=1)
        n_correct = torch.eq(y_pred_indices, y_true).sum().item()
        return n_correct / len(y_pred_indices)
        
    def test(self, split: str):
        """Test the model on the dataset for a given split
        
        Args:
            dataset (pd.DataFrame): The dataset
            split (str): The split to test on (train, val, test)
        """
        assert split in ['train', 'val', 'test'], "Split should be one of ['train', 'val', 'test']"
        self.dataset.set_split(split)
        if split == 'test':
            self.dataset = self.test_dataset
        elif split == 'val':
            self.dataset = self.val_dataset
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size)
        print(f'########### {split} size: {len(self.dataset.y)}')
        running_loss = 0.0
        running_acc = 0.0
        for batch in tqdm(dataloader):
            if split == 'train':
                self.model.train()
            else:
                self.model.eval()
            x, y = batch
            y = torch.tensor([self.dataset.lookup_country(country) for country in y]).to(self.device)
            x = x.to(self.device)
            y_pred = self.model(x, apply_softmax=False)
            loss = self.loss_fn(y_pred, y)
            running_loss += loss.item()
            running_acc += (y_pred.argmax(1) == y).sum().item()
        return running_loss / len(self.dataset), running_acc / len(self.dataset)
        
    def build_model(self, args):
        model = SurnameClassifier(
            input_size=len(self.dataset),
            hidden_size=args.hidden_size,
            output_size=len(self.dataset.y)
        )
        return model
    
    def build_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5) 
        return optimizer
    
    def plot_results(self, train_loss, train_acc, val_loss, val_acc):
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes[0, 0].plot(train_loss, label='Train Loss')
        axes[0, 1].plot(val_loss, label='Val Loss')
        axes[1, 0].plot(train_acc, label='Train Acc')
        axes[1, 1].plot(val_acc, label='Val Acc')
        axes[0, 0].set_title('Train Loss')
        axes[0, 1].set_title('Val Loss')
        axes[1, 0].set_title('Train Acc')
        axes[1, 1].set_title('Val Acc')
        plt.savefig('results.png')
        
    
    def train_epoch(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        running_loss = 0.0
        running_acc = 0.0
        
        for batch in tqdm(dataloader):
            self.model.train()
            x, y = batch
            y = torch.tensor([self.train_dataset.lookup_country(country) for country in y]).to(self.device)
            x = x.to(self.device)
            y_pred = self.model(x, apply_softmax=False)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_acc += (y_pred.argmax(1) == y).sum().item()
        epoch_loss = running_loss / len(self.train_dataset)
        epoch_acc = running_acc / len(self.train_dataset)
        return epoch_loss, epoch_acc
    
    def train2(self):
        
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        for epoch in range(self.args.num_epochs):
            epoch_loss, epoch_acc = self.train_epoch()
            # Save the model with the best result on the validation
            val_running_loss = 0.0
            val_running_acc = 0.0
            with torch.no_grad():
                self.model.eval()
                for batch in tqdm(DataLoader(self.val_dataset, batch_size=self.args.batch_size)):
                    x, y = batch
                    y = torch.tensor([self.val_dataset.lookup_country(country) for country in y]).to(self.device)
                    x = x.to(self.device)
                    y_pred = self.model(x, apply_softmax=True)
                    loss = self.loss_fn(y_pred, y)
                    val_running_loss += loss.item()
                    val_running_acc += (y_pred.argmax(1) == y).sum().item()
            print(f"Val loss: {val_running_loss / len(self.val_dataset)}, Val acc: {val_running_acc / len(self.val_dataset)}")
            val_loss = val_running_loss / len(self.val_dataset)
            val_acc = val_running_acc / len(self.val_dataset)
            best_val_loss = float('inf')
            best_val_acc = 0.0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.args.model_state_file)
                print("Model saved")
            
            print(f"Epoch {epoch+1}/{self.args.num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
        self.plot_results(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        # test the model on test set
        # self.model.load_state_dict(torch.load(self.args.model_state_file))
        self.model.eval()
        test_running_loss = 0.0
        test_running_acc = 0.0
        with torch.no_grad():
            for batch in tqdm(DataLoader(self.test_dataset, batch_size=self.args.batch_size)):
                x, y = batch
                y = torch.tensor([self.test_dataset.lookup_country(country) for country in y]).to(self.device)
                x = x.to(self.device)
                y_pred = self.model(x, apply_softmax=True)
                test_loss = self.loss_fn(y_pred, y)
                test_running_loss += test_loss.item()
                test_running_acc += (y_pred.argmax(1) == y).sum().item()
        print(f"Test loss: {test_running_loss / len(self.test_dataset)}, Test acc: {test_running_acc / len(self.test_dataset)}")
        return best_val_loss, best_val_acc
    
    
    def train(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        for epoch in range(self.args.num_epochs):
            print(f'########### train size: {len(self.train_dataset.y)}')
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0
            for batch in tqdm(dataloader):
                x, y = batch
                y = torch.tensor([self.dataset.lookup_country(country) for country in y]).to(self.device)
                x = x.to(self.device)
                y_pred = self.model(x, apply_softmax=False)
                loss = self.loss_fn(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_acc += (y_pred.argmax(1) == y).sum().item()
            epoch_loss = running_loss / len(self.dataset)
            epoch_acc = running_acc / len(self.dataset)
            print(f"Epoch {epoch+1}/{self.args.num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            val_loss, val_acc = self.test('val')
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        test_loss, test_acc = self.test('test')
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")