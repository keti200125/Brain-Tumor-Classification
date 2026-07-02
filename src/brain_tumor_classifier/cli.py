"""Command-line interface for the legacy experiment workflow."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing ML dependencies."""
    parser = argparse.ArgumentParser(description="Task 02 - Brain tumor multiclass classification")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment YAML path, resolved from the project root",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the selected config and exit without training",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable transfer learning weights")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    cli_args = parse_args(argv)
    if cli_args.validate_config and not cli_args.config:
        build_parser().error("--validate-config requires --config")

    # Keep costly torch/torchvision imports out of the help path.
    import torch

    from brain_tumor_classifier.config import load_experiment_config
    from brain_tumor_classifier.data import (
        INCEPTION_MIN_IMAGE_SIZE,
        BrainTumorDataModule,
    )
    from brain_tumor_classifier.experiments import ExperimentRunner as ConfigExperimentRunner
    from brain_tumor_classifier.legacy import (
        ExperimentRunner as LegacyExperimentRunner,
        get_project_root,
    )
    from brain_tumor_classifier.utils import seed_everything

    root = get_project_root()
    if cli_args.config:
        config = load_experiment_config(cli_args.config, project_root=root)
        args = argparse.Namespace(
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            image_size=config.training.image_size,
            validation_fraction=config.data.validation_fraction,
            num_workers=config.training.num_workers,
            seed=config.seed,
            max_train_samples=config.data.max_train_samples,
            max_val_samples=config.data.max_val_samples,
            max_test_samples=config.data.max_test_samples,
            device=config.device,
            no_pretrained=not config.model.pretrained,
        )
        data_root = config.data.root
        model_names = [config.model.name]
        save_eda_plots = config.outputs.save_eda_plots
        save_training_curves = config.outputs.save_training_curves
        save_prediction_grid = config.outputs.save_prediction_grid
        save_best_checkpoint_enabled = config.outputs.save_best_checkpoint
        save_last_checkpoint_enabled = config.outputs.save_last_checkpoint
        monitor_metric = config.training.monitor_metric
        monitor_mode = config.training.monitor_mode
        experiment_config = config
        requires_inception_size = config.model.name == "inception_v3"
        print(
            f"Loaded config: {config.config_filename} "
            f"(sha256={config.config_hash[:8]})"
        )
        if cli_args.validate_config:
            print(f"Experiment: {config.experiment_name}")
            print(f"Model: {config.model.name}")
            print(f"Data root: {config.data.root}")
            print(f"Output root: {config.outputs.root}")
            print("Config validation passed")
            return
        result = ConfigExperimentRunner(
            config=config,
            config_path=config.config_path,
        ).run()
        print(f"Run ID: {result.run_id}")
        print(f"Run directory: {result.run_dir}")
        print(f"Checkpoint path: {result.checkpoint_path or '(not saved)'}")
        print(
            "Test metrics: "
            f"accuracy={result.test_accuracy:.4f} "
            f"f1_weighted={result.test_f1_weighted:.4f} "
            f"precision_weighted={result.test_precision_weighted:.4f} "
            f"recall_weighted={result.test_recall_weighted:.4f}"
        )
        return
    else:
        args = cli_args
        data_candidates = (
            root / "DATA" / "brain_tumor_dataset",
            root / "DATA",
            root / "data",
        )
        data_root = next(
            (
                candidate
                for candidate in data_candidates
                if (candidate / "Training").is_dir()
                and (candidate / "Testing").is_dir()
            ),
            data_candidates[0],
        )
        output_root = root
        plots_dir = root
        checkpoints_dir = root
        model_names = None
        save_eda_plots = True
        save_training_curves = True
        save_prediction_grid = True
        save_best_checkpoint_enabled = True
        save_last_checkpoint_enabled = True
        monitor_metric = "f1_weighted"
        monitor_mode = "max"
        experiment_config = None
        run_id = "legacy"
        requires_inception_size = True

    seed_everything(args.seed)
    effective_image_size = args.image_size
    if requires_inception_size and args.image_size < INCEPTION_MIN_IMAGE_SIZE:
        print(
            "InceptionV3 requires larger inputs for stable training "
            f"Overriding image size from {args.image_size} to {INCEPTION_MIN_IMAGE_SIZE}"
        )
        effective_image_size = INCEPTION_MIN_IMAGE_SIZE

    output_root.mkdir(parents=True, exist_ok=True)

    data_module = BrainTumorDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        image_size=effective_image_size,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    experiment = LegacyExperimentRunner(
        data_module=data_module,
        args=args,
        device=device,
        output_root=output_root,
        plots_dir=plots_dir,
        checkpoints_dir=checkpoints_dir,
        model_names=model_names,
        save_eda_plots=save_eda_plots,
        save_training_curves=save_training_curves,
        save_prediction_grid=save_prediction_grid,
        save_best_checkpoint_enabled=save_best_checkpoint_enabled,
        save_last_checkpoint_enabled=save_last_checkpoint_enabled,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        experiment_config=experiment_config,
        run_id=run_id,
    )
    experiment.run()
