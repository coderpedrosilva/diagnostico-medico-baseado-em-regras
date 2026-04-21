import os

def export_prism_rules(prism_model, path="reports/rules.txt"):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Total de regras geradas: {len(prism_model.rules)}\n\n")
        for i, (attr, val, target, conf) in enumerate(prism_model.rules, 1):
            label = "diabetes" if target == 1 else "sem diabetes"
            f.write(
                f"REGRA {i}: SE {attr} == {val} "
                f"ENTÃO {label} (confiança: {conf:.2%})\n"
            )
