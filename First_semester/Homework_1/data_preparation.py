import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import random
random.seed(42)
np.random.seed(42)


def find_damage(area: pd.Series) -> str:
    if area == 0:
        return "None"
    elif area <= 2:
        return "Low"
    elif area <= 30:
        return "Average"
    elif area <= 100:
        return "High"
    else:
        return "Very high"


def get_metrics(real, pred) -> dict:
    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred)

    metrics = {"MSE": mse, "MAE": mae, "R_2": r2}
    return metrics


dataset = pd.read_csv("./data/forestfires.csv")
target = "area"

target_zscores = stats.zscore(dataset[target])
target_outliers = dataset[abs(target_zscores) >= 3]

feature_data = dataset.drop(columns=target)
cat_columns = feature_data.select_dtypes(include="object").columns.tolist()
quan_columns = feature_data.select_dtypes(exclude="object").columns.tolist()

dataset["damage"] = dataset[target].apply(find_damage)

quan_features = dataset.drop(columns=["month", "day", "damage"]).columns

outliers_columns = ["FFMC", "ISI", "rain", "area"]

dataset = pd.get_dummies(dataset, columns=["month", "day"], drop_first=True)

mask = dataset.loc[:, ["FFMC"]].apply(stats.zscore).abs() < 3
dataset["rain"] = dataset["rain"].apply(lambda x: int(x > 0.0))
dataset = dataset[mask.values]

outliers_columns.remove("rain")
dataset.loc[:, outliers_columns] = np.log1p(dataset[outliers_columns])

# Final dataset for ML models.
dataset_ml = dataset.drop(columns=["damage"]).copy()

# Move target column to the last position.
target_column = dataset_ml.pop(target)
dataset_ml.insert(27, target, target_column)

X = dataset_ml.drop(columns=target)
y = dataset_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,
                                                    shuffle=True,
                                                    random_state=42)
X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()

np.save("./data/X_train", X_train_array)
np.save("./data/X_test", X_test_array)
np.save("./data/y_train", y_train_array)
np.save("./data/y_test", y_test_array)