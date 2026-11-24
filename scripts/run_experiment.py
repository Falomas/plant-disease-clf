"""Запускает один опыт: exp1_mobilenet_frozen, exp2_mobilenet_finetune или exp3_efficientnet_early_stop."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.training import run_experiment


def main():
    if len(sys.argv) != 2:
        sys.exit(
            "нужно имя опыта: exp1_mobilenet_frozen, "
            "exp2_mobilenet_finetune или exp3_efficientnet_early_stop"
        )
    run_experiment(sys.argv[1])


if __name__ == "__main__":
    main()
