"""Presets del protocolo para el runner (`--preset`).

Un preset cambia los DEFAULTS del runner, no los fuerza: un flag dado a mano gana
siempre. Los defaults sin preset siguen siendo los históricos (`mean`, λ 10, regla
v1, 15 sesiones), así que las corridas viejas se reproducen igual.

Las claves son los `dest` de argparse (`--l2-form` → `l2_form`).

`paper2`: el protocolo de una sesión del paper 2, común al MLP y a la CNN.
- L2 `sample` con λ 0 (sin regularización).
- Regla de selección v2: balanced accuracy con 1 SE y desempate por CE.
- 1 sesión y 1 ronda; warm-start con el 30 % de las partículas.

Quedan fuera porque el protocolo los fija distinto según el modelo o el experimento:
`g` (MLP 1.2, CNN 1.13), `--patience` (MLP = iteraciones, CNN 1,000,000; ver
sweep.sh y sweep_cnn.sh), `--data-seed`, partículas e iteraciones.
"""
from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "paper2": {
        "l2_form": "sample",
        "lambda_l2": 0.0,
        "selection_metric": "bacc",
        "selection_se": 1.0,
        "n_sessions": 1,
        "n_rounds": 1,
        "warm_percent": 30.0,
    },
}
