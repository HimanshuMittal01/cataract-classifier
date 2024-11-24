"""Entrypoint main script"""

import typer

from cataract_classifier.data import get_dataset_paths
from cataract_classifier.train import train

# pretty_exceptions_show_locals is set to False because locals value
# are long, and sometimes it is difficult to traceback exceptions
cli = typer.Typer(pretty_exceptions_show_locals=False)


@cli.command()
def train_and_evaluate(
    dataset_path: str = "input/processed_images/",
    save_path: str = "models/finetuned_efficientnet_b0.pt",
    batch_size: int = 32,
    num_epochs: int = 10,
):
    # Get training and validation datasets
    data_paths = get_dataset_paths(
        dataset_path, valid_test_split=0.5, random_seed=42
    )

    train(
        train_img_paths=data_paths["train"],
        valid_img_paths=data_paths["valid"],
        save_path=save_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    cli()
