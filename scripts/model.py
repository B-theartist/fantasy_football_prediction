import torch.nn as nn
import torch

class FantasyFootballLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super(FantasyFootballLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer with moderate dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Additional dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Define a more complex fully connected layer structure
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass the output through fully connected layers with relu activation
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        
        return out
