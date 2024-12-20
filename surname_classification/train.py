from argparse import Namespace
import os
import json
import pandas as pd
from regex import T
import torch
from trainer import Trainer


def train_surname_classifier(args):
    pass

if __name__ == "__main__":
    args = Namespace(
        
        # Data and path information
        surname_csv="hf://datasets/Hobson/surname-nationality/surname-nationality.csv.gz",
        model_state_file="model.pth",
        save_dir="model_storage/ch4/surname_mlp",
        
        # Model hyper parameters
        hidden_size=300,
        
        # Model
        optimizer="Adam",
        metric="accuracy",
        loss='cross_entropy',
        
        # Training hyper parameters
        seed=42,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=100,
        early_stopping_criteria=5,
        
        # Runtime options
        cuda=True,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True,
    )

    trainer = Trainer(args)
    trainer.train()