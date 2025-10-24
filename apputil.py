import pandas as pd

class GroupEstimate:
    def __init__(self, estimate):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_values_ = None

    def fit(self, X, y):
        """Fits the estimator by computing mean/median per group."""
        #make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        #make sure y is the same length as X
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        #combine X and y
        df = X.copy()
        df["__y__"] = y

        #group by all columns in X and calculate mean/median
        if self.estimate == "mean":
            group_estimates = df.groupby(list(X.columns))["__y__"].mean()
        else:
            group_estimates = df.groupby(list(X.columns))["__y__"].median()

        #store results as a dictionary
        self.group_values_ = group_estimates.to_dict()

        return self
