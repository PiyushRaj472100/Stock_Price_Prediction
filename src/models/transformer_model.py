import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer_encoder(encoded)
        # Aggregate sequence representation (mean pooling over time steps)
        pooled = out.mean(dim=1)
        return self.fc(pooled).squeeze(-1)


class TransformerForecaster:
    """Compact Time-Series Transformer Forecasting Model."""

    def __init__(self, input_dim: int, d_model: int = 32, epochs: int = 35, lr: float = 0.01):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.model = TimeSeriesTransformer(input_dim=input_dim, d_model=d_model)
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
