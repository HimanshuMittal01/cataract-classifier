"""Entrypoint main script"""

import typer

from cataract_classifier.data import get_dataset_paths
from cataract_classifier.train import train
from cataract_classifier.predict import predict_on_testset

# pretty_exceptions_show_locals is set to False because locals value
# are long, and sometimes it is difficult to traceback exceptions
cli = typer.Typer(pretty_exceptions_show_locals=False)


@cli.command()
def train_and_evaluate(
    dataset_path: str = "input/processed_images/",
    results_path: str = "results/",
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    random_seed: int = 0,
):
    # Get training and validation datasets
    data_paths = get_dataset_paths(
        dataset_path, valid_test_split=0.5, random_seed=random_seed
    )

    train(
        train_img_paths=data_paths["train"],
        valid_img_paths=data_paths["valid"],
        model_name=model_name,
        results_path=results_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    predict_on_testset(
        test_img_paths=data_paths["test"],
        model_name=model_name,
        results_path=results_path,
    )


if __name__ == "__main__":
    cli()
