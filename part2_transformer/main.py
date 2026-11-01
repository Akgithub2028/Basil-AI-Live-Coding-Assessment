"""Part 2 submission entry point: the locked temporal-tokenization experiment.

YourModel is implemented in experiment_models.py. The vanilla model in
models.py is unchanged. Run python main.py (GPU if available) or use the
controlled-experiment Colab notebook. See EXPERIMENT_PLAN.md before training.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from experiment_models import YourModel


def main():
    from experiment import main as run_experiment
    run_experiment(prepare_inputs=True)


if __name__ == "__main__":
    main()
