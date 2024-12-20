import torch
from torch import nn


class SurnameClassifier(nn.Module):
    """ A simple MLP for classifying surnames """
    def __init__(self, input_size, hidden_size, output_size):
        """ Initialize the model 
        
        Args: 
            input_size (int): The size of the input vectors
            hidden_size (int): The size of the hidden layer
            output_size (int): The size of the output layer
        """
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        
    def forward(self, x, apply_softmax):
        """ The forward pass of the classifier
        
        Args:
            x (torch.Tensor): An input data tensor. 
                x.shape should be (batch, input_size)
            apply_softmax (bool): A flag for the softmax activation
                should be false if used with the cross-entropy losses
                use it for inference
                
        Returns:
            torch.Tensor: The resulting tensor. tensor.shape should be (batch, output_size)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        if apply_softmax:
            x = torch.softmax(x, dim=1)
        
        return x
    
if __name__ == "__main__":
    # Test the model
    model = SurnameClassifier(input_size=10, hidden_size=8, output_size=3)
    x = torch.rand(size=(64, 10))
    y = model(x, apply_softmax=True)
    print(y.shape)