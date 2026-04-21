import numpy as np
import pandas as pd


class OneR:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y):
        self.best_feature = None
        self.rules = {}
        self.error = float("inf")
        self.bins = None
        self._majority_class = int(y.value_counts().idxmax())

        for col in X.columns:
            df = pd.DataFrame({col: X[col], "y": y})
            bins = pd.qcut(df[col], q=self.n_bins, duplicates="drop")
            df["bin"] = bins

            rules = df.groupby("bin", observed=False)["y"].agg(
                lambda x: x.value_counts().idxmax()
            )

            preds = df["bin"].map(rules).fillna(self._majority_class)
            error = (preds != y).mean()

            if error < self.error:
                self.error = error
                self.best_feature = col
                self.rules = rules
                self.bins = bins.cat.categories

    def predict(self, X):
        # Clipa ao intervalo de treino para evitar NaN em pd.cut
        lo = self.bins[0].left
        hi = self.bins[-1].right
        col_clipped = X[self.best_feature].clip(lower=lo, upper=hi)

        bins = pd.cut(col_clipped, bins=self.bins, include_lowest=True)
        preds = bins.map(self.rules).fillna(self._majority_class)
        return preds.to_numpy().astype(int)
