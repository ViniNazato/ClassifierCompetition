import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss, CrossEntropyLoss

from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(21, 21),
            nn.Tanh(),
            nn.Linear(21, 14),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(14, 21),
            nn.Tanh(),
            nn.Linear(21, 21),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class AutoEncoderOptimization:
    def __init__(self,X, y, model, batch_size:int=32, lr:float=0.001):
        
        self.model = model
        self.X = X
        self.y = y
        
        self.batch_size = batch_size
        
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=10e-5)
        self.loss_fn = MSELoss() 
        
        torch.manual_seed(42)
    
    def fit(self, epochs=15):
        
        self.model.train()
        
        train_losses = {}
        train_losses['Loss'] = []

        X_train = self.X.values
        
        # Create the DataLoader
        train_loader = self._to_dataloader(X_train, train=True)
        
        for epoch in range(1, epochs + 1):
           
            losses = np.array([])
            for X_train in train_loader:
            
                # ===================forward=====================
                X_pred = self.model(X_train[0])
                loss = self.loss_fn (X_pred, X_train[0])
                losses = np.append(losses, loss.item())
                
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self. optimizer.step()
                
            # ===================log========================
            print(f'Epoch [{epoch}/{epochs}], loss:{np.mean(losses)}')
            train_losses['Loss'].append(np.mean(losses))

        return pd.DataFrame(train_losses)

    def predict(self, X_test:pd.DataFrame, y_test:pd.DataFrame):
        
        test_losses = []
        self.model.eval()
        
        X_test = X_test.values
        y_test = y_test.values
        
        test_loader = self._to_dataloader(X_test,train=False)
        
        with torch.no_grad():

            for X_test in test_loader:

                X_hat = self.model(X_test[0])
                loss = self.loss_fn(X_hat, X_test[0]).data.item()
                test_losses.append(loss)
    
        df_anomaly = pd.DataFrame({'Test_Losses': test_losses})
        df_anomaly['Class'] = y_test
        
        return df_anomaly

    def _to_dataloader(self, X_data ,train:bool):
        
        X_data = TensorDataset(torch.from_numpy(X_data).float())
        
        if train:          
            return DataLoader(X_data, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(X_data,  batch_size=1, shuffle=False)
