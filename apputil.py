import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_values_ = None
        self.columns_ = None

    def fit(self, X, y):
        """Fits the estimator by computing mean/median per group."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        #combine X and Y
        self.columns_ = list(X.columns)
        df = X.copy()
        df["__y__"] = y

        #group by columns in X and calculate mean/median
        if self.estimate == "mean":
            group_estimates = df.groupby(self.columns_)["__y__"].mean()
        else:
            group_estimates = df.groupby(self.columns_)["__y__"].median()

        self.group_values_ = group_estimates.to_dict()
        return self

    def predict(self, X_):
        """Predicts estimates for new data based on learned group statistics."""
        if self.group_values_ is None:
            raise ValueError("The model must be fitted before calling predict().")

        #convert input to DataFrame
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.columns_)

        predictions = []
        missing_count = 0

        #loop through observations
        for _, row in X_.iterrows():
            key = tuple(row[col] for col in self.columns_)
            value = self.group_values_.get(key, np.nan)
            if pd.isna(value):
                missing_count += 1
            predictions.append(value)

        if missing_count > 0:
            print(f"Warning: {missing_count} observation(s) had unseen group(s); returning NaN for them.")

        return np.array(predictions)
