import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn

# Define the Autoencoder model
class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim, train_sample_size = None, epochs=200, lr=0.01):
        super(AutoEncoder, self).__init__()
        self.hidden_dim =  hidden_dim
        self.train_sample_size = train_sample_size
        self.epochs = epochs
        self.lr = lr

    def initialize_layers(self, input_dim):
        self.input_dim = input_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def transform(self, x):
        return self.encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    
    def fit(self, X_train, X_test=None, verbosity = 0):
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        self.initialize_layers(X_train.shape[1])

        if self.train_sample_size:
            random_idxs = torch.randperm(X_train.shape[0])[:self.train_sample_size]
            X_train = X_train[random_idxs]
        
        if X_test is not None:
            X_test = torch.tensor(X_test, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        #for verbosity
        print_epochs = np.linspace(1, self.epochs, verbosity+1).astype('int') if verbosity > 0 else []

        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.forward(X_train)
            loss = criterion(outputs, X_train)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (epoch+1) in print_epochs:
                if X_test is not None:
                    with torch.no_grad():
                        test_outputs = self.forward(X_test)
                        test_loss = criterion(test_outputs, X_test)
                    print(f"Epoch {epoch+1}/{self.epochs}, train_loss: {loss.item()}, val_loss: {test_loss.item()}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs}, train_loss: {loss.item()}")

    def fit_transform(self, X_train, X_test=None, verbosity=0):
        self.fit(
            X_train, X_test=X_test, verbosity=verbosity)
        return self.transform(X_train)

# dummy sklearn class for inclusion in sklearn pipelines
class SklearnAutoEncoder(AutoEncoder, BaseEstimator, TransformerMixin):
    def __init__(self, hidden_dim, train_sample_size = None, epochs=200, lr=0.01):
        super().__init__(
            hidden_dim=hidden_dim,
            train_sample_size=train_sample_size,
            epochs=epochs,
            lr=lr
        )
