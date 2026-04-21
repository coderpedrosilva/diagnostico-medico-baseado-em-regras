import numpy as np
import pandas as pd


class Prism:
    def __init__(self, n_bins=5, min_support=0.03):
        """
        n_bins      : bins para discretizar colunas contínuas (>10 valores únicos)
        min_support : fração mínima do treino que uma regra deve cobrir (evita overfitting)
        """
        self.n_bins = n_bins
        self.min_support = min_support

    # ------------------------------------------------------------------
    # Discretização interna
    # ------------------------------------------------------------------

    def _fit_bins(self, X):
        self._bin_edges = {}
        for col in X.columns:
            if X[col].nunique() > 10:
                _, edges = pd.qcut(X[col], q=self.n_bins, duplicates="drop", retbins=True)
                # Garante que os extremos cobrem todo o domínio real
                edges[0] = -np.inf
                edges[-1] = np.inf
                self._bin_edges[col] = edges

    def _apply_bins(self, X):
        Xd = X.copy()
        for col, edges in self._bin_edges.items():
            if col in Xd.columns:
                Xd[col] = pd.cut(Xd[col], bins=edges, labels=False, include_lowest=True)
        return Xd

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------

    def fit(self, X, y):
        self.rules = []
        self._majority_class = int(y.value_counts().idxmax())
        min_count = max(1, int(self.min_support * len(X)))

        self._fit_bins(X)
        Xd = self._apply_bins(X)

        df = Xd.copy()
        df["_y"] = y.values

        for target in sorted(df["_y"].unique()):
            subset = df[df["_y"] == target].copy()

            while not subset.empty:
                best_attr = None
                best_value = None
                best_prob = -1.0

                for col in Xd.columns:
                    for val in subset[col].dropna().unique():
                        total = (df[col] == val).sum()
                        if total < min_count:
                            continue
                        prob = (subset[col] == val).sum() / total
                        if prob > best_prob:
                            best_prob = prob
                            best_attr = col
                            best_value = val

                # Nenhuma condição atende o suporte mínimo: encerra esta classe
                if best_attr is None:
                    break

                self.rules.append((best_attr, best_value, target, round(best_prob, 4)))
                subset = subset[subset[best_attr] != best_value]

    # ------------------------------------------------------------------
    # Predição (vetorizada; regra com maior confiança vence)
    # ------------------------------------------------------------------

    def predict(self, X):
        Xd = self._apply_bins(X)
        preds = np.full(len(X), self._majority_class, dtype=int)
        best_conf = np.full(len(X), -1.0)

        for attr, val, target, conf in self.rules:
            mask = (Xd[attr] == val).to_numpy()
            update = mask & (conf > best_conf)
            preds[update] = target
            best_conf[update] = conf

        return preds
