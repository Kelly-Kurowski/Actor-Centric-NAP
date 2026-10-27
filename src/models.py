from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn


# Centralized neural-network architecture defaults.

# Evermann, Rehse & Fettke (2017), Predicting Process Behaviour using Deep Learning
LSTM_DEFAULTS = {
    "embedding_dim": 32,
    "hidden_size": 64,
    "dropout": 0.2,
}

# Defaults like in ProcessTransformer (Remco Dijkman)
TRANSFORMER_DEFAULTS = {
    "embedding_dim": 36,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
}


# XGBoost
def create_xgboost(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=1.0,
):
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )


# LightGBM
def create_lightgbm(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
):
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


# Random Forest
def create_random_forest(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=1,
    )


# LSTM
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim,
        hidden_size,
        dropout,
        num_features=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=hidden_size + num_features,
            out_features=num_classes,
        )

    def forward(self, x_sequence, x_features=None):
        embedded = self.embedding(x_sequence)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]

        if x_features is not None:
            combined = torch.cat([last_hidden, x_features.float()], dim=1)
        else:
            combined = last_hidden

        return self.fc(self.dropout(combined))


def create_lstm(
    vocab_size,
    num_classes,
    embedding_dim=None,
    hidden_size=None,
    dropout=None,
    num_features=0,
):
    return LSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=LSTM_DEFAULTS["embedding_dim"] if embedding_dim is None else embedding_dim,
        hidden_size=LSTM_DEFAULTS["hidden_size"] if hidden_size is None else hidden_size,
        dropout=LSTM_DEFAULTS["dropout"] if dropout is None else dropout,
        num_features=num_features,
    )


# Transformer
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        max_length,
        embedding_dim,
        num_heads,
        num_layers,
        dropout,
        num_features=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(
            in_features=embedding_dim + num_features,
            out_features=num_classes,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_sequence, x_features=None):
        positions = torch.arange(
            x_sequence.size(1),
            device=x_sequence.device,
        ).unsqueeze(0)
        embedded = self.embedding(x_sequence) + self.position_embedding(positions)
        encoded = self.transformer(embedded)
        last_position = encoded[:, -1, :]

        if x_features is not None:
            combined = torch.cat([last_position, x_features.float()], dim=1)
        else:
            combined = last_position

        return self.fc(self.dropout(combined))


def create_transformer(
    vocab_size,
    num_classes,
    max_length,
    embedding_dim=None,
    num_heads=None,
    num_layers=None,
    dropout=None,
    num_features=0,
):
    return TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length,
        embedding_dim=TRANSFORMER_DEFAULTS["embedding_dim"] if embedding_dim is None else embedding_dim,
        num_heads=TRANSFORMER_DEFAULTS["num_heads"] if num_heads is None else num_heads,
        num_layers=TRANSFORMER_DEFAULTS["num_layers"] if num_layers is None else num_layers,
        dropout=TRANSFORMER_DEFAULTS["dropout"] if dropout is None else dropout,
        num_features=num_features,
    )
