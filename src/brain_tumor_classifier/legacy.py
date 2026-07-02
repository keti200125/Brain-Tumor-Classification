from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_tumor_classifier.config.schema import ExperimentConfig
from brain_tumor_classifier.data import BrainTumorDataModule, ImageSample
from brain_tumor_classifier.models import (
    build_model,
    get_architecture_name,
)
from brain_tumor_classifier.reporting import (
    build_plot_paths,
    save_class_examples_plot,
    save_distribution_plot,
    save_image_size_plot,
    save_prediction_grid,
    save_training_curves,
)
from brain_tumor_classifier.training import (
    build_checkpoint,
    compute_classification_metrics,
    evaluate_model,
    save_best_checkpoint,
    save_last_checkpoint,
    train_model,
    validate_reloaded_model,
)
from brain_tumor_classifier.utils import seed_everything


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ExperimentRunner:
    def __init__(
        self,
        data_module: BrainTumorDataModule,
        args: argparse.Namespace,
        device: torch.device,
        output_root: Path,
        plots_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        model_names: list[str] | None = None,
        save_eda_plots: bool = True,
        save_training_curves: bool = True,
        save_prediction_grid: bool = True,
        save_best_checkpoint_enabled: bool = True,
        save_last_checkpoint_enabled: bool = True,
        monitor_metric: str = "f1_weighted",
        monitor_mode: str = "max",
        experiment_config: ExperimentConfig | None = None,
        run_id: str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.data_module = data_module
        self.args = args
        self.device = device
        self.output_root = output_root
        self.plots_dir = plots_dir or output_root
        self.checkpoints_dir = checkpoints_dir or output_root
        self.criterion = nn.CrossEntropyLoss()
        self.model_names = model_names or [
            "custom_cnn",
            "inception_v3",
            "vgg11",
            "resnet18",
        ]
        self.save_eda_plots = save_eda_plots
        self.save_training_curves = save_training_curves
        self.save_prediction_grid_enabled = save_prediction_grid
        self.save_best_checkpoint_enabled = save_best_checkpoint_enabled
        self.save_last_checkpoint_enabled = save_last_checkpoint_enabled
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.experiment_config = experiment_config
        self.run_id = run_id or "legacy"
        self.show_progress = show_progress

    def run(self) -> None:
        self.data_module.setup()
        train_samples = self.data_module.train_samples
        val_samples = self.data_module.val_samples
        test_samples = self.data_module.test_samples
        plot_paths = build_plot_paths(self.plots_dir)

        if self.save_eda_plots:
            save_class_examples_plot(
                samples=train_samples,
                class_names=self.data_module.class_names,
                output_path=plot_paths["eda_class_samples_path"],
                seed=self.args.seed,
                per_class=5,
            )
            save_distribution_plot(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                class_names=self.data_module.class_names,
                output_path=plot_paths["eda_class_distribution_path"],
            )
            save_image_size_plot(
                samples=train_samples + val_samples + test_samples,
                output_path=plot_paths["eda_image_sizes_path"],
            )

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()

        trained_results: dict[str, dict[str, float]] = {}
        reloaded_results: dict[str, dict[str, float]] = {}
        used_pretrained: dict[str, bool] = {}
        loaded_models: dict[str, nn.Module] = {}
        histories: dict[str, dict[str, list[float]]] = {}
        model_hyperparams: dict[str, dict[str, str]] = {}

        for model_name in self.model_names:
            (
                model,
                got_pretrained,
                history,
                test_metrics,
                reloaded_metrics,
                reloaded_model,
            ) = self._train_and_evaluate(model_name, train_loader, val_loader, test_loader)

            trained_results[model_name] = test_metrics
            reloaded_results[model_name] = reloaded_metrics
            used_pretrained[model_name] = got_pretrained
            loaded_models[model_name] = reloaded_model
            histories[model_name] = history
            model_hyperparams[model_name] = {
                "architecture": f"{get_architecture_name(model_name)} ({'pretrained' if got_pretrained else 'random init'})",
                "input_size": f"{self.data_module.image_size}x{self.data_module.image_size}",
                "epochs": str(self.args.epochs),
                "batch_size": str(self.data_module.batch_size),
                "learning_rate": f"{self.args.learning_rate}",
                "optimizer": "AdamW",
            }

        if self.save_prediction_grid_enabled:
            self._save_prediction_grid(
                test_samples,
                loaded_models,
                plot_paths["prediction_grid_path"],
            )
        loss_curve_path = plot_paths["train_vs_val_loss_path"]
        metric_curve_path = plot_paths["train_vs_val_f1_path"]
        if self.save_training_curves:
            save_training_curves(
                histories=histories,
                output_loss_path=loss_curve_path,
                output_metric_path=metric_curve_path,
            )

        write_model_report(
            report_path=self.output_root / "data-science-2-model-report.md",
            model_order=self.model_names,
            train_samples=train_samples,
            test_samples=test_samples,
            train_size=len(train_samples),
            val_size=len(val_samples),
            test_size=len(test_samples),
            class_names=self.data_module.class_names,
            initial_results=trained_results,
            reloaded_results=reloaded_results,
            model_hyperparams=model_hyperparams,
            used_pretrained=used_pretrained,
            loss_plot_name=loss_curve_path.name,
            metric_plot_name=metric_curve_path.name,
        )

        print("\nTask 02 completed")
        print(f"Artifacts saved in: {self.output_root}")

    def _train_and_evaluate(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> tuple[nn.Module, bool, dict[str, list[float]], dict[str, float], dict[str, float], nn.Module]:
        print(f"\n===== Training {model_name} =====")
        model, got_pretrained = build_model(
            model_name=model_name,
            num_classes=len(self.data_module.class_names),
            pretrained=not self.args.no_pretrained,
        )
        model = model.to(self.device)
        use_inception_aux = model_name == "inception_v3"

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            trainable_params = list(model.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.learning_rate)
        training_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=self.criterion,
            optimizer=optimizer,
            device=self.device,
            model_name=model_name,
            epochs=self.args.epochs,
            monitor_metric=self.monitor_metric,
            monitor_mode=self.monitor_mode,
            use_inception_aux=use_inception_aux,
            show_progress=self.show_progress,
        )

        model.load_state_dict(training_result.best_state_dict)
        test_result = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=self.criterion,
            device=self.device,
            model_name=f"{model_name}-test",
            use_inception_aux=use_inception_aux,
            show_progress=self.show_progress,
        )
        test_metrics = test_result.metrics
        print(
            f"[{model_name}] test loss={test_result.loss:.4f}, "
            f"metrics={test_metrics}"
        )

        if self.experiment_config is not None:
            experiment_name = self.experiment_config.experiment_name
            trace_name = self.experiment_config.trace_name
            config_file = str(self.experiment_config.config_path)
            config_hash = self.experiment_config.config_hash
            best_path = self.checkpoints_dir / "best_model.pt"
            last_path = self.checkpoints_dir / "last_model.pt"
        else:
            experiment_name = "legacy_all_models"
            trace_name = f"legacy_{model_name}"
            config_file = ""
            config_hash = ""
            best_path = self.checkpoints_dir / f"best_model__{model_name}.pt"
            last_path = self.checkpoints_dir / f"last_model__{model_name}.pt"

        checkpoint_metadata = {
            "experiment_name": experiment_name,
            "trace_name": trace_name,
            "run_id": self.run_id,
            "seed": self.args.seed,
            "config_file": config_file,
            "config_hash": config_hash,
            "model_name": model_name,
            "class_names": self.data_module.class_names,
            "class_to_idx": self.data_module.class_to_idx,
            "image_size": self.data_module.image_size,
            "pretrained_used": got_pretrained,
            "best_epoch": training_result.best_epoch,
            "best_val_metric": training_result.best_val_metric,
            "final_test_metrics": test_metrics,
        }

        if self.save_best_checkpoint_enabled:
            best_checkpoint = build_checkpoint(
                state_dict=training_result.best_state_dict,
                **checkpoint_metadata,
            )
            save_best_checkpoint(best_path, best_checkpoint)
            print(f"Saved best {model_name} checkpoint to {best_path}")
        if self.save_last_checkpoint_enabled:
            last_checkpoint = build_checkpoint(
                state_dict=training_result.last_state_dict,
                **checkpoint_metadata,
            )
            save_last_checkpoint(last_path, last_checkpoint)
            print(f"Saved last {model_name} checkpoint to {last_path}")

        reloaded_model, _ = build_model(
            model_name=model_name,
            num_classes=len(self.data_module.class_names),
            pretrained=False,
        )
        if self.save_best_checkpoint_enabled:
            validation = validate_reloaded_model(
                model=reloaded_model,
                checkpoint_path=best_path,
                loader=test_loader,
                criterion=self.criterion,
                device=self.device,
                model_name=f"{model_name}-reloaded-test",
                expected_metrics=test_metrics,
                use_inception_aux=use_inception_aux,
                show_progress=self.show_progress,
            )
            reloaded_metrics = validation.evaluation.metrics
        else:
            reloaded_model.load_state_dict(training_result.best_state_dict)
            reloaded_model = reloaded_model.to(self.device)
            reloaded_result = evaluate_model(
                model=reloaded_model,
                loader=test_loader,
                criterion=self.criterion,
                device=self.device,
                model_name=f"{model_name}-reloaded-test",
                use_inception_aux=use_inception_aux,
                show_progress=self.show_progress,
            )
            reloaded_metrics = reloaded_result.metrics
        print(f"Reloaded {model_name} metrics: {reloaded_metrics}")

        return (
            model,
            got_pretrained,
            training_result.plot_history(),
            test_metrics,
            reloaded_metrics,
            reloaded_model,
        )

    def _save_prediction_grid(
        self,
        test_samples: list[ImageSample],
        loaded_models: dict[str, nn.Module],
        output_path: Path,
    ) -> None:
        rng = random.Random(self.args.seed)
        sampled_test_images = test_samples[:]
        rng.shuffle(sampled_test_images)
        sampled_test_images = sampled_test_images[:5]

        save_prediction_grid(
            samples=sampled_test_images,
            loaded_models=loaded_models,
            class_names=self.data_module.class_names,
            transform=self.data_module.eval_transform,
            device=self.device,
            output_path=output_path,
        )


def format_metric_with_change(value: float, baseline: float) -> str:
    if baseline == 0:
        return f"{value:.4f} (n/a)"
    delta_pct = ((value - baseline) / baseline) * 100.0
    sign = "+" if delta_pct >= 0 else ""
    return f"{value:.4f} ({sign}{delta_pct:.2f}%)"


def compute_majority_baseline_metrics(
    train_samples: list[ImageSample],
    test_samples: list[ImageSample],
) -> tuple[dict[str, float], int]:
    train_labels = [sample.label for sample in train_samples]
    test_labels = [sample.label for sample in test_samples]

    majority_label = Counter(train_labels).most_common(1)[0][0]
    predictions = [majority_label] * len(test_labels)

    baseline_metrics = compute_classification_metrics(test_labels, predictions)
    return baseline_metrics, majority_label


def write_model_report(
    report_path: Path,
    model_order: list[str],
    train_samples: list[ImageSample],
    test_samples: list[ImageSample],
    train_size: int,
    val_size: int,
    test_size: int,
    class_names: list[str],
    initial_results: dict[str, dict[str, float]],
    reloaded_results: dict[str, dict[str, float]],
    model_hyperparams: dict[str, dict[str, str]],
    used_pretrained: dict[str, bool],
    loss_plot_name: str,
    metric_plot_name: str,
) -> None:
    baseline_metrics, majority_label = compute_majority_baseline_metrics(train_samples, test_samples)

    best_model_name = max(model_order, key=lambda name: initial_results[name]["f1_weighted"])
    best_model_metrics = initial_results[best_model_name]

    lines = [
        "# Model Report - Task 02",
        "",
        (
            "Best model: "
            f"**{best_model_name}** because it has the highest weighted F1 on the test set "
            f"({best_model_metrics['f1_weighted']:.4f}) while maintaining strong accuracy "
            f"({best_model_metrics['accuracy']:.4f})"
        ),
        "",
        "## Dataset Setup",
        f"- Classes: {', '.join(class_names)}",
        f"- Train images: {train_size}",
        f"- Validation images: {val_size}",
        f"- Test images: {test_size}",
        "",
        "## Transfer Learning",
    ]

    for model_name, pretrained_used in used_pretrained.items():
        lines.append(f"- {model_name}: {'pretrained backbone used' if pretrained_used else 'fallback to random initialization'}")

    lines.extend(
        [
            "",
            "## Main Experiment Table",
            "Rows are kept in experiment order. First row is the baseline model",
            "",
            "| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Accuracy (vs baseline) | Test F1 Weighted (vs baseline) | Test Recall Weighted (vs baseline) | Comments |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )

    lines.append(
        "| "
        + " | ".join(
            [
                "baseline_majority_class",
                f"Predict most common train class ({class_names[majority_label]})",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                format_metric_with_change(baseline_metrics["accuracy"], baseline_metrics["accuracy"]),
                format_metric_with_change(baseline_metrics["f1_weighted"], baseline_metrics["f1_weighted"]),
                format_metric_with_change(baseline_metrics["recall_weighted"], baseline_metrics["recall_weighted"]),
                "Greedy statistical baseline; no learning",
            ]
        )
        + " |"
    )

    for model_name in model_order:
        metrics = initial_results[model_name]
        reload_metrics = reloaded_results[model_name]
        params = model_hyperparams[model_name]

        delta_reload = abs(metrics["f1_weighted"] - reload_metrics["f1_weighted"])
        reload_comment = "Reload stable" if delta_reload <= 1e-6 else f"Reload delta F1={delta_reload:.6f}"
        gain_comment = "Improves baseline" if metrics["f1_weighted"] >= baseline_metrics["f1_weighted"] else "Under baseline on F1"

        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    params["architecture"],
                    params["epochs"],
                    params["batch_size"],
                    params["learning_rate"],
                    params["optimizer"],
                    format_metric_with_change(metrics["accuracy"], baseline_metrics["accuracy"]),
                    format_metric_with_change(metrics["f1_weighted"], baseline_metrics["f1_weighted"]),
                    format_metric_with_change(metrics["recall_weighted"], baseline_metrics["recall_weighted"]),
                    f"{gain_comment} {reload_comment}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Diagrams",
            f"- Train vs validation loss: `{loss_plot_name}`",
            f"- Train vs validation main metric (F1 weighted): `{metric_plot_name}`",
            "",
            "## Notes",
            "- The table is not sorted; it follows experiment creation order",
            "- Metrics include value and percentage change vs baseline",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
