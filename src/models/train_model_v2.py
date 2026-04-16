"""
Train models against the v2 clinically grounded feature matrix.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.train_model import ModelTrainer


def main():
    trainer = ModelTrainer(features_path='data/processed/features_engineered_v2.csv')
    trainer.train_all_models()
    trainer.save_models(output_dir='models')
    print("\n*** v2 model training complete.")


if __name__ == "__main__":
    main()
