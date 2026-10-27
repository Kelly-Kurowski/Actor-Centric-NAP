import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models import (
    create_random_forest,
    create_lightgbm,
    create_xgboost,
    create_lstm,
    create_transformer,
    LSTM_DEFAULTS,
    TRANSFORMER_DEFAULTS,
)


def make_tree_model(model_name):
    if model_name == "RandomForest":
        return create_random_forest()
    if model_name == "LightGBM":
        return create_lightgbm()
    if model_name == "XGBoost":
        return create_xgboost()
    raise ValueError(f"Unsupported tree model: {model_name}")


def get_parameter_space(model_name):
    if model_name == "RandomForest":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 40],
            "min_samples_leaf": [1, 2, 5],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [-1, 10, 20],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 40],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        }
    if model_name == "LSTM":
        return {
            "embedding_dim": [16, 32, 64],
            "hidden_size": [32, 64, 128],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005],
            "batch_size": [32, 64],
        }
    if model_name == "Transformer":
        return {
            "embedding_dim": [16, 32, 36, 64],
            "num_heads": [2, 4],
            "num_layers": [1, 2],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005],
            "batch_size": [32, 64],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def get_cv_strategy(y_train, random_state=42, max_folds=3):
    """Use 2-3 stratified folds when each target class has >= 2 samples."""
    y_train = np.asarray(y_train)
    n_samples = len(y_train)
    if n_samples < 3:
        raise ValueError(
            f"Only {n_samples} training samples. "
            "At least three are required for tuning."
        )
    if max_folds < 2:
        raise ValueError("max_folds must be at least 2.")
    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError(
            "At least one target class has fewer than two samples. "
            "Reliable stratified cross-validation is not possible."
        )
    return StratifiedKFold(
        n_splits=min(max_folds, min_class_count),
        shuffle=True,
        random_state=random_state,
    )


class TorchSequenceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model_name,
        vocab_size,
        num_classes,
        sequence_length,
        num_features=0,
        embedding_dim=None,
        hidden_size=None,
        num_heads=None,
        num_layers=None,
        dropout=None,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        early_stopping_patience=5,
        validation_fraction=0.15,
        random_state=42,
    ):
        # Direct assignments are required by sklearn.clone().
        # None means use the architecture default from models.py.
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _build_model(self):
        if self.model_name == "LSTM":
            return create_lstm(
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
                embedding_dim=self.embedding_dim,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                num_features=self.num_features,
            )
        if self.model_name == "Transformer":
            # The create_transformer() factory resolves None to the
            # architecture defaults declared in models.py.
            embedding_dim = (
                TRANSFORMER_DEFAULTS["embedding_dim"]
                if self.embedding_dim is None else self.embedding_dim
            )
            num_heads = (
                TRANSFORMER_DEFAULTS["num_heads"]
                if self.num_heads is None else self.num_heads
            )
            if embedding_dim % num_heads != 0:
                raise ValueError("embedding_dim must be divisible by num_heads.")
            return create_transformer(
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
                max_length=self.sequence_length,
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                num_features=self.num_features,
            )
        raise ValueError(f"Unsupported neural model: {self.model_name}")

    def _split_inputs(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if X.shape[1] < self.sequence_length:
            raise ValueError("X has fewer columns than sequence_length.")
        sequences = X[:, :self.sequence_length].astype(np.int64)
        features = X[:, self.sequence_length:].astype(np.float32)
        if features.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} numerical features, "
                f"received {features.shape[1]}."
            )
        if np.any(sequences < 0):
            raise ValueError("Activity IDs cannot be negative.")
        if np.any(sequences >= self.vocab_size):
            raise ValueError("An activity ID exceeds the configured vocabulary.")
        return sequences, features

    def _forward(self, sequences, features):
        if self.num_features > 0:
            return self.model_(sequences, features)
        return self.model_(sequences)

    def fit(self, X, y):
        """Train with class-preserving internal validation when possible."""
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        sequences, features = self._split_inputs(X)
        y = np.asarray(y, dtype=np.int64)
        if len(sequences) != len(y):
            raise ValueError("X and y have different lengths.")
        if len(y) < 3:
            raise ValueError(
                "At least three samples are required for "
                "training with internal validation."
            )
        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("Target labels are outside the valid class range.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")

        self.classes_ = np.arange(self.num_classes)
        indices = np.arange(len(y))
        classes, counts = np.unique(y, return_counts=True)
        eligible_classes = classes[counts >= 2]
        eligible_indices = indices[np.isin(y, eligible_classes)]
        desired_validation_size = max(
            1, int(np.ceil(len(y) * self.validation_fraction))
        )
        max_validation_size = int(sum(counts - 1))
        validation_size = min(desired_validation_size, max_validation_size)

        rng = np.random.RandomState(self.random_state)
        if validation_size > 0:
            remaining = {cls: int(count) for cls, count in zip(classes, counts)}
            shuffled = rng.permutation(eligible_indices)
            selected = []
            for index in shuffled:
                cls = y[index]
                if remaining[cls] > 1:
                    selected.append(index)
                    remaining[cls] -= 1
                    if len(selected) == validation_size:
                        break
            validation_indices = np.asarray(selected, dtype=int)
        else:
            validation_indices = np.asarray([], dtype=int)

        train_indices = np.setdiff1d(
            indices, validation_indices, assume_unique=True
        )
        self.used_internal_validation_ = len(validation_indices) > 0

        # Standardize additional numerical features using training data only.
        if self.num_features > 0:
            self.feature_scaler_ = StandardScaler()
            self.feature_scaler_.fit(features[train_indices])
            features = self.feature_scaler_.transform(features).astype(np.float32)
        else:
            self.feature_scaler_ = None

        train_sequences = torch.as_tensor(
            sequences[train_indices], dtype=torch.long
        )
        train_features = torch.as_tensor(
            features[train_indices], dtype=torch.float32
        )
        train_targets = torch.as_tensor(y[train_indices], dtype=torch.long)
        train_dataset = TensorDataset(
            train_sequences, train_features, train_targets
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        self.model_ = self._build_model()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.best_validation_loss_ = None
        self.validation_loss_history_ = []

        if self.used_internal_validation_:
            val_sequences = torch.as_tensor(
                sequences[validation_indices], dtype=torch.long
            )
            val_features = torch.as_tensor(
                features[validation_indices], dtype=torch.float32
            )
            val_targets = torch.as_tensor(
                y[validation_indices], dtype=torch.long
            )

        for _ in range(self.epochs):
            self.model_.train()
            for batch_sequences, batch_features, batch_targets in loader:
                optimizer.zero_grad()
                logits = self._forward(batch_sequences, batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

            # If there is no possible internal holdout, train all 50 epochs.
            if not self.used_internal_validation_:
                continue

            self.model_.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for start in range(0, len(val_sequences), self.batch_size):
                    end = start + self.batch_size
                    logits = self._forward(
                        val_sequences[start:end], val_features[start:end]
                    )
                    batch_targets = val_targets[start:end]
                    loss = criterion(logits, batch_targets)
                    batch_count = len(batch_targets)
                    total_val_loss += loss.item() * batch_count
                    total_val_samples += batch_count

            val_loss = total_val_loss / total_val_samples
            self.validation_loss_history_.append(float(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                break

        if self.used_internal_validation_:
            if best_state is None:
                raise RuntimeError("Early stopping failed to save model weights.")
            self.model_.load_state_dict(best_state)
            self.best_validation_loss_ = float(best_val_loss)
        return self

    def predict(self, X):
        sequences, features = self._split_inputs(X)

        # Apply the scaler fitted during training.
        if self.feature_scaler_ is not None:
            features = self.feature_scaler_.transform(features).astype(np.float32)

        sequence_tensor = torch.as_tensor(
            sequences, dtype=torch.long
        )
        feature_tensor = torch.as_tensor(
            features, dtype=torch.float32
        )

        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                end = start + self.batch_size
                logits = self._forward(
                    sequence_tensor[start:end],
                    feature_tensor[start:end]
                )
                predictions.extend(
                    torch.argmax(logits, dim=1).cpu().numpy()
                )

        return np.asarray(predictions, dtype=np.int64)


def _collect_search_results(search):
    """Collect scores and parameters for every search candidate."""
    results = []
    for i, params in enumerate(search.cv_results_["params"]):
        results.append({
            "candidate": i + 1,
            "params": params,
            "mean_cv_accuracy": float(search.cv_results_["mean_test_score"][i]),
            "std_cv_accuracy": float(search.cv_results_["std_test_score"][i]),
            "mean_train_accuracy": float(search.cv_results_["mean_train_score"][i]),
            "rank": int(search.cv_results_["rank_test_score"][i]),
        })
    results.sort(key=lambda item: item["rank"])
    return results


def tune_tree_model(
    model_name,
    X_train,
    y_train,
    n_iter=8,
    random_state=42,
    cv_folds=3,
):
    """Tune Random Forest, LightGBM, or XGBoost."""
    y_train = np.asarray(y_train)
    _, counts = np.unique(y_train, return_counts=True)
    if len(counts) < 2:
        raise ValueError(
            "Training has only one target class; this experiment "
            "requires at least two distinct target classes."
        )
    if counts.min() == 1:
        estimator = make_tree_model(model_name)
        estimator.fit(X_train, y_train)
        best = {
            "params": estimator.get_params(),
            "cv_accuracy": None,
            "best_index": None,
            "cv_folds": 0,
            "cv_strategy": None,
            "scoring": "accuracy",
            "tuning_status": "skipped_rare_class",
            "min_class_count": 1,
        }
        return estimator, best, []

    cv_strategy = get_cv_strategy(
        y_train, random_state=random_state, max_folds=cv_folds
    )
    search = RandomizedSearchCV(
        estimator=make_tree_model(model_name),
        param_distributions=get_parameter_space(model_name),
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv_strategy,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )
    print(f"Tuning {model_name} with RandomizedSearchCV...")
    search.fit(X_train, y_train)
    best = {
        "params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "best_index": int(search.best_index_),
        "cv_folds": cv_strategy.get_n_splits(),
        "cv_strategy": type(cv_strategy).__name__,
        "scoring": "accuracy",
        "tuning_status": "completed",
        "min_class_count": int(counts.min()),
    }
    return search.best_estimator_, best, _collect_search_results(search)


def tune_neural_model(
    model_name,
    X_sequence_train,
    X_features_train,
    y_train,
    vocab_size,
    num_classes,
    n_iter=6,
    random_state=42,
    cv_folds=3,
):
    """
    Tune an LSTM or Transformer by cross-validation accuracy.

    Each fit uses at most 50 epochs and, when an internal validation
    holdout is available, early stopping on validation loss (patience 5).
    Epochs are fixed, not included in hyperparameter search.
    Architecture fallback defaults are defined in models.py.
    """
    if model_name not in {"LSTM", "Transformer"}:
        raise ValueError("model_name must be 'LSTM' or 'Transformer'.")
    X_sequence_train = np.asarray(X_sequence_train, dtype=np.int64)
    X_features_train = np.asarray(X_features_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    if X_sequence_train.ndim != 2:
        raise ValueError("Activity sequences must be a 2D array.")
    if X_features_train.ndim != 2:
        raise ValueError("Numerical features must be a 2D array.")
    if not (
        len(X_sequence_train) == len(X_features_train) == len(y_train)
    ):
        raise ValueError("Sequence, feature, and target row counts differ.")
    if len(y_train) == 0:
        raise ValueError("Training data cannot be empty.")

    X_train = np.concatenate(
        [X_sequence_train.astype(np.float32), X_features_train], axis=1
    )
    sequence_length = X_sequence_train.shape[1]
    num_features = X_features_train.shape[1]
    # Explicitly pull fallback architecture hyperparameters from models.py.
    defaults = LSTM_DEFAULTS if model_name == "LSTM" else TRANSFORMER_DEFAULTS
    estimator = TorchSequenceClassifier(
        model_name=model_name,
        vocab_size=vocab_size,
        num_classes=num_classes,
        sequence_length=sequence_length,
        num_features=num_features,
        embedding_dim=defaults["embedding_dim"],
        hidden_size=defaults.get("hidden_size"),
        num_heads=defaults.get("num_heads"),
        num_layers=defaults.get("num_layers"),
        dropout=defaults["dropout"],
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        early_stopping_patience=5,
        validation_fraction=0.15,
        random_state=random_state,
    )

    _, counts = np.unique(y_train, return_counts=True)
    if len(counts) < 2:
        raise ValueError(
            "Training has only one target class; this experiment "
            "requires at least two distinct target classes."
        )
    if counts.min() == 1:
        estimator.fit(X_train, y_train)
        best = {
            "params": {
                key: estimator.get_params()[key]
                for key in get_parameter_space(model_name)
            },
            "cv_accuracy": None,
            "best_index": None,
            "cv_folds": 0,
            "cv_strategy": None,
            "scoring": "accuracy",
            "tuning_status": "skipped_rare_class",
            "min_class_count": 1,
            "max_epochs": estimator.epochs,
            "early_stopping_patience": estimator.early_stopping_patience,
            "validation_fraction": estimator.validation_fraction,
            "best_validation_loss_final_refit": estimator.best_validation_loss_,
            "used_internal_validation": estimator.used_internal_validation_,
        }
        return estimator, best, []

    cv_strategy = get_cv_strategy(
        y_train, random_state=random_state, max_folds=cv_folds
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=get_parameter_space(model_name),
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv_strategy,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )
    print(f"Tuning {model_name} with RandomizedSearchCV and early stopping...")
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    best = {
        "params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "best_index": int(search.best_index_),
        "cv_folds": cv_strategy.get_n_splits(),
        "cv_strategy": type(cv_strategy).__name__,
        "scoring": "accuracy",
        "tuning_status": "completed",
        "min_class_count": int(counts.min()),
        "max_epochs": best_estimator.epochs,
        "early_stopping_patience": best_estimator.early_stopping_patience,
        "validation_fraction": best_estimator.validation_fraction,
        "best_validation_loss_final_refit": (
            best_estimator.best_validation_loss_
        ),
        "used_internal_validation": best_estimator.used_internal_validation_,
    }
    return best_estimator, best, _collect_search_results(search)


def save_tuning_results(results, output_file):
    """Save tuning results as a JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=str)
