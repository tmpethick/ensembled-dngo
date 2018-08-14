import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.utils.data
import torch
from torch import nn
import torch.utils.data

device = torch.device("cpu")

class TorchRegressionModel(object):
    def __init__(self, input_dim=1, dim_basis=50, dim_h1=50, dim_h2=50, epochs=100, batch_size=10, lr=0.01, weight_decay=0):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.basis = nn.Sequential(
            nn.Linear(input_dim, dim_h1),
            nn.Tanh(),
            nn.Linear(dim_h1, dim_h2),
            nn.Tanh(),
            nn.Linear(dim_h2, dim_basis),
            nn.Tanh()
        ).to(device)
        
        self.model = nn.Sequential(
            self.basis, 
            nn.Linear(dim_basis, 1)).to(device)

        self.criterion = nn.MSELoss()
        # l2 penalty high (1e-2) leads to little exploration, low (1e-6) leads to more exploration
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


    def fit(self, X, y):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def weights_init(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.model.apply(weights_init)

        # Train the model
        for epoch in range(self.epochs):
            for i, (X, y) in enumerate(dataloader):
                # Move tensors to the configured device
                X = X.to(device)
                y = y.to(device)
                
                # Forward pass
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.from_numpy(X).float()
        X = X.to(device)

        with torch.no_grad():
            return self.model(X).numpy()

    def predict_basis(self, X):
        X = torch.from_numpy(X).float()
        X = X.to(device)

        with torch.no_grad():
            return self.basis(X).numpy()

    def plot_basis_functions(self, bounds):
        if bounds.shape is not (1, 2):
            raise Exception("Only supports 1D")

        X = np.linspace(bounds[:, 0], bounds[:, 1])[:, None]
        D = self.predict_basis(X)
        for i in range(self.dim_basis):
            plt.plot(x, D[:, i])
