from src.preprocess import load_data
from src.majority import MajorityClassifier
from src.oner import OneR
from src.prism import Prism

# Carregar dados
X_train, X_test, y_train, y_test = load_data()

print("Sistema de Diagnóstico por Regras iniciado")

# Majority
majority = MajorityClassifier()
majority.fit(y_train)
majority_preds = majority.predict(X_test)

print("Majority prediction:", majority_preds[:10])

# OneR
oner = OneR()
oner.fit(X_train, y_train)
oner_preds = oner.predict(X_test)

print("OneR using:", oner.best_feature)
print("OneR predictions:", oner_preds[:10])

# PRISM
prism = Prism()
prism.fit(X_train, y_train)
prism_preds = prism.predict(X_test)

print("PRISM predictions:", prism_preds[:10])

from src.evaluate import evaluate_model

evaluate_model("Majority", y_test, majority_preds)
evaluate_model("OneR", y_test, oner_preds)
evaluate_model("PRISM", y_test, prism_preds)

from src.rule_engine import export_prism_rules
export_prism_rules(prism)
print("Regras PRISM exportadas para reports/rules.txt")
