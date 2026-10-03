from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.lstm_model import LSTMModel
else:
    from .lstm_model import LSTMModel

WINDOW_SIZE = 30
FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("results/models/lstm_model.pth")


class StockDataset(Dataset):
    """Turn a feature matrix into fixed-length training sequences."""

    def __init__(self, data: np.ndarray) -> None:
        features = []
        targets = []

        for index in range(WINDOW_SIZE, len(data)):
            features.append(data[index - WINDOW_SIZE : index])
            targets.append(data[index][0])

        self.x = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def train() -> None:
    """Train and save the baseline LSTM model."""
    data = pd.read_csv(FEATURES_PATH, index_col=0)
    scaled_data = StandardScaler().fit_transform(data)

    dataset = StockDataset(scaled_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LSTMModel(input_size=scaled_data.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(x_batch).squeeze(-1)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model training complete.")


if __name__ == "__main__":
    train()
