import numpy as np
import xgboost
from xgboost import XGBClassifier

print("XGBoost version:", xgboost.__version__, flush=True)

X = np.random.default_rng(42).random((100, 10))
y = np.array([0, 1, 2, 0, 1, 2] * 16 + [0, 1, 2, 0])

model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=10,
    max_depth=3,
    n_jobs=1,
    tree_method="hist",
    random_state=42,
)

print("Starting fit...", flush=True)
model.fit(X, y)
print("Fit succeeded.", flush=True)

print(model.predict(X[:5]))