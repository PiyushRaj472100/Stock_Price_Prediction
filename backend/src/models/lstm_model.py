import torch
import torch.nn as nn
import numpy as np


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class LSTMForecaster:
    """LSTM Time-Series Forecasting Model."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, epochs: int = 35, lr: float = 0.01):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.model = LSTMNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train) == 0:
            return
        
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

        self.is_trained = True

    def predict(self, X_input: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_input, dtype=torch.float32)
            if X_t.ndim == 2:
                X_t = X_t.unsqueeze(0)
            preds = self.model(X_t)
            return preds.cpu().numpy()
