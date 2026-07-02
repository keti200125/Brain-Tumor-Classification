"""Config-driven experiment orchestration."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from brain_tumor_classifier.config import ExperimentConfig, load_experiment_config
from brain_tumor_classifier.data import INCEPTION_MIN_IMAGE_SIZE, BrainTumorDataModule
from brain_tumor_classifier.experiments.history import (
    append_experiment_history_row,
    experiment_result_to_row,
    failed_experiment_row,
    write_metrics_json,
    write_run_history_csv,
)
from brain_tumor_classifier.experiments.naming import RunPaths, create_run_paths
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
    evaluate_model,
    save_best_checkpoint,
    save_last_checkpoint,
    train_model,
    validate_reloaded_model,
)
from brain_tumor_classifier.utils import (
    close_logger,
    configure_run_logger,
    seed_everything,
)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    trace_name: str
    run_id: str
    config_file: str
    config_hash: str
    model_name: str
    architecture: str
    pretrained_requested: bool
    pretrained_used: bool
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    image_size: int
    train_size: int
    val_size: int
    test_size: int
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_f1_weighted: float
    test_loss: float
    test_accuracy: float
    test_f1_weighted: float
    test_precision_weighted: float
    test_recall_weighted: float
    reloaded_test_accuracy: float
    reloaded_test_f1_weighted: float
    checkpoint_path: str
    run_dir: str
    created_at: str


class ExperimentRunner:
    """Run one experiment from a validated config."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Path,
        *,
        show_progress: bool = True,
        timestamp: datetime | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.show_progress = show_progress
        self.timestamp = timestamp

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        project_root: str | Path | None = None,
        show_progress: bool = True,
        timestamp: datetime | None = None,
    ) -> "ExperimentRunner":
        config = load_experiment_config(config_path, project_root=project_root)
        return cls(
            config=config,
            config_path=config.config_path,
            show_progress=show_progress,
            timestamp=timestamp,
        )

    def run(self) -> ExperimentResult:
        seed_everything(self.config.seed)
        run_paths = create_run_paths(self.config, timestamp=self.timestamp)
        plot_paths = build_plot_paths(run_paths.plots_dir)
        plot_paths_str = {
            key: str(path)
            for key, path in plot_paths.items()
        }
        run_id = run_paths.run_dir.name
        created_at = self._created_at()
        log_path = run_paths.logs_dir / "run.log"
        logger = configure_run_logger(
            experiment_name=self.config.experiment_name,
            run_id=run_id,
            log_path=log_path,
        )
        logger.info("Starting experiment run")
        logger.info("Config path: %s", self.config.config_path)
        logger.info("Run ID: %s", run_id)
        logger.info("Run directory: %s", run_paths.run_dir)

        try:
            device = self._resolve_device(self.config.device)
            logger.info("Device: %s", device)
            image_size = self._resolve_image_size()
            data_module = self._build_data_module(image_size)
            data_module.setup()
            logger.info(
                "Dataset sizes: train=%s val=%s test=%s",
                len(data_module.train_samples),
                len(data_module.val_samples),
                len(data_module.test_samples),
            )
            logger.info("%s", data_module.formatted_split_balance_report())
            split_balance_report = data_module.split_balance_report()
            for warning_message in split_balance_report.warnings:
                logger.warning("Split balance warning: %s", warning_message)
            self._maybe_generate_eda_plots(data_module, plot_paths)

            model, pretrained_used = build_model(
                self.config.model.name,
                num_classes=len(data_module.class_names),
                pretrained=self.config.model.pretrained,
            )
            model.to(device)
            logger.info(
                "Model: %s (architecture=%s, pretrained_requested=%s, pretrained_used=%s)",
                self.config.model.name,
                get_architecture_name(self.config.model.name),
                self.config.model.pretrained,
                pretrained_used,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = self._build_optimizer(model)
            use_inception_aux = self.config.model.name == "inception_v3"

            training_result = train_model(
                model=model,
                train_loader=data_module.train_dataloader(),
                val_loader=data_module.val_dataloader(),
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                model_name=self.config.model.name,
                epochs=self.config.training.epochs,
                monitor_metric=self.config.training.monitor_metric,
                monitor_mode=self.config.training.monitor_mode,
                use_inception_aux=use_inception_aux,
                show_progress=self.show_progress,
                on_epoch_end=lambda record: logger.info(
                    "Epoch %s/%s | training_loss=%.4f | validation_loss=%.4f | "
                    "training_f1=%.4f | validation_f1=%.4f | duration=%.2fs",
                    record.epoch,
                    self.config.training.epochs,
                    record.train.loss,
                    record.validation.loss,
                    record.train.metrics["f1_weighted"],
                    record.validation.metrics["f1_weighted"],
                    record.duration_seconds,
                ),
            )

            history_payload = {self.config.model.name: training_result.plot_history()}
            if self.config.outputs.save_training_curves:
                save_training_curves(
                    history_payload,
                    plot_paths["train_vs_val_loss_path"],
                    plot_paths["train_vs_val_f1_path"],
                )

            best_record = training_result.epochs[training_result.best_epoch - 1]
            model.load_state_dict(training_result.best_state_dict)
            test_result = evaluate_model(
                model=model,
                loader=data_module.test_dataloader(),
                criterion=criterion,
                device=device,
                model_name=f"{self.config.model.name}-test",
                use_inception_aux=use_inception_aux,
                show_progress=self.show_progress,
            )

            best_checkpoint_path = run_paths.checkpoints_dir / "best_model.pt"
            last_checkpoint_path = run_paths.checkpoints_dir / "last_model.pt"
            best_checkpoint = build_checkpoint(
                experiment_name=self.config.experiment_name,
                trace_name=self.config.trace_name,
                run_id=run_id,
                seed=self.config.seed,
                config_file=str(self.config.config_path),
                config_hash=self.config.config_hash,
                model_name=self.config.model.name,
                state_dict=training_result.best_state_dict,
                class_names=data_module.class_names,
                class_to_idx=data_module.class_to_idx,
                image_size=image_size,
                pretrained_used=pretrained_used,
                best_epoch=training_result.best_epoch,
                best_val_metric=training_result.best_val_metric,
                final_test_metrics=test_result.metrics,
            )
            last_checkpoint = build_checkpoint(
                experiment_name=self.config.experiment_name,
                trace_name=self.config.trace_name,
                run_id=run_id,
                seed=self.config.seed,
                config_file=str(self.config.config_path),
                config_hash=self.config.config_hash,
                model_name=self.config.model.name,
                state_dict=training_result.last_state_dict,
                class_names=data_module.class_names,
                class_to_idx=data_module.class_to_idx,
                image_size=image_size,
                pretrained_used=pretrained_used,
                best_epoch=training_result.best_epoch,
                best_val_metric=training_result.best_val_metric,
                final_test_metrics=test_result.metrics,
            )
            if self.config.outputs.save_best_checkpoint:
                save_best_checkpoint(best_checkpoint_path, best_checkpoint)
            if self.config.outputs.save_last_checkpoint:
                save_last_checkpoint(last_checkpoint_path, last_checkpoint)
            logger.info(
                "Checkpoint paths: best=%s last=%s",
                best_checkpoint_path if self.config.outputs.save_best_checkpoint else "(not saved)",
                last_checkpoint_path if self.config.outputs.save_last_checkpoint else "(not saved)",
            )

            reloaded_metrics = test_result.metrics
            if self.config.outputs.save_best_checkpoint:
                reloaded_model, _ = build_model(
                    self.config.model.name,
                    num_classes=len(data_module.class_names),
                    pretrained=False,
                )
                reloaded_model.to(device)
                reloaded_validation = validate_reloaded_model(
                    model=reloaded_model,
                    checkpoint_path=best_checkpoint_path,
                    loader=data_module.test_dataloader(),
                    criterion=criterion,
                    device=device,
                    model_name=f"{self.config.model.name}-reloaded",
                    expected_metrics=test_result.metrics,
                    use_inception_aux=use_inception_aux,
                    show_progress=self.show_progress,
                )
                reloaded_metrics = reloaded_validation.evaluation.metrics

            best_model_for_grid, _ = build_model(
                self.config.model.name,
                num_classes=len(data_module.class_names),
                pretrained=False,
            )
            best_model_for_grid.to(device)
            best_model_for_grid.load_state_dict(training_result.best_state_dict)
            best_model_for_grid.eval()
            if self.config.outputs.save_prediction_grid:
                self._maybe_generate_prediction_grid(
                    data_module=data_module,
                    model=best_model_for_grid,
                    device=device,
                    prediction_grid_path=plot_paths["prediction_grid_path"],
                )

            checkpoint_path = (
                str(best_checkpoint_path)
                if self.config.outputs.save_best_checkpoint
                else (
                    str(last_checkpoint_path)
                    if self.config.outputs.save_last_checkpoint
                    else ""
                )
            )
            result = ExperimentResult(
                experiment_name=self.config.experiment_name,
                trace_name=self.config.trace_name,
                run_id=run_id,
                config_file=str(self.config.config_path),
                config_hash=self.config.config_hash,
                model_name=self.config.model.name,
                architecture=get_architecture_name(self.config.model.name),
                pretrained_requested=self.config.model.pretrained,
                pretrained_used=pretrained_used,
                epochs=self.config.training.epochs,
                batch_size=self.config.training.batch_size,
                learning_rate=self.config.training.learning_rate,
                optimizer=self.config.training.optimizer,
                image_size=image_size,
                train_size=len(data_module.train_samples),
                val_size=len(data_module.val_samples),
                test_size=len(data_module.test_samples),
                best_epoch=training_result.best_epoch,
                best_val_loss=best_record.validation.loss,
                best_val_accuracy=best_record.validation.metrics["accuracy"],
                best_val_f1_weighted=best_record.validation.metrics["f1_weighted"],
                test_loss=test_result.loss,
                test_accuracy=test_result.metrics["accuracy"],
                test_f1_weighted=test_result.metrics["f1_weighted"],
                test_precision_weighted=test_result.metrics["precision_weighted"],
                test_recall_weighted=test_result.metrics["recall_weighted"],
                reloaded_test_accuracy=reloaded_metrics["accuracy"],
                reloaded_test_f1_weighted=reloaded_metrics["f1_weighted"],
                checkpoint_path=checkpoint_path,
                run_dir=str(run_paths.run_dir),
                created_at=created_at,
            )

            write_run_history_csv(
                path=run_paths.epoch_history_path,
                config=self.config,
                run_id=run_id,
                model_name=self.config.model.name,
                training_result=training_result,
                created_at=created_at,
            )
            metrics_payload = self._build_success_metrics_payload(
                result=result,
                data_module=data_module,
                checkpoint_path=checkpoint_path,
                last_checkpoint_path=(
                    str(last_checkpoint_path)
                    if self.config.outputs.save_last_checkpoint
                    else ""
                ),
                run_paths=run_paths,
                plot_paths=plot_paths_str,
            )
            write_metrics_json(run_paths.metrics_json_path, metrics_payload)
            append_experiment_history_row(
                run_paths.experiment_history_path,
                experiment_result_to_row(
                    result=result,
                    config=self.config,
                    best_val_precision_weighted=best_record.validation.metrics["precision_weighted"],
                    best_val_recall_weighted=best_record.validation.metrics["recall_weighted"],
                    last_checkpoint_path=(
                        str(last_checkpoint_path)
                        if self.config.outputs.save_last_checkpoint
                        else ""
                    ),
                    plots_dir=str(run_paths.plots_dir),
                    plot_paths=plot_paths_str,
                    metrics_json_path=str(run_paths.metrics_json_path),
                    history_csv_path=str(run_paths.epoch_history_path),
                    status="success",
                ),
            )
            logger.info("Final checkpoint path: %s", checkpoint_path or "(not saved)")
            logger.info(
                "Final metrics: test_loss=%.4f accuracy=%.4f f1_weighted=%.4f precision_weighted=%.4f recall_weighted=%.4f reloaded_accuracy=%.4f reloaded_f1_weighted=%.4f",
                result.test_loss,
                result.test_accuracy,
                result.test_f1_weighted,
                result.test_precision_weighted,
                result.test_recall_weighted,
                result.reloaded_test_accuracy,
                result.reloaded_test_f1_weighted,
            )
            self._maybe_regenerate_report(run_paths)
            return result
        except Exception as exc:
            logger.exception("Experiment failed: %s", exc)
            write_metrics_json(
                run_paths.metrics_json_path,
                self._build_failed_metrics_payload(
                    run_id=run_id,
                    created_at=created_at,
                    error_message=str(exc),
                    run_paths=run_paths,
                    plot_paths=plot_paths_str,
                ),
            )
            append_experiment_history_row(
                run_paths.experiment_history_path,
                failed_experiment_row(
                    config=self.config,
                    run_id=run_id,
                    created_at=created_at,
                    run_dir=str(run_paths.run_dir),
                    plots_dir=str(run_paths.plots_dir),
                    plot_paths=plot_paths_str,
                    metrics_json_path=str(run_paths.metrics_json_path),
                    history_csv_path=str(run_paths.epoch_history_path),
                    error_message=str(exc),
                ),
            )
            raise
        finally:
            close_logger(logger)

    def _build_data_module(self, image_size: int) -> BrainTumorDataModule:
        return BrainTumorDataModule(
            data_root=self.config.data.root,
            batch_size=self.config.training.batch_size,
            image_size=image_size,
            validation_fraction=self.config.data.validation_fraction,
            num_workers=self.config.training.num_workers,
            seed=self.config.seed,
            max_train_samples=self.config.data.max_train_samples,
            max_val_samples=self.config.data.max_val_samples,
            max_test_samples=self.config.data.max_test_samples,
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        return torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.training.learning_rate,
        )

    def _resolve_image_size(self) -> int:
        if (
            self.config.model.name == "inception_v3"
            and self.config.training.image_size < INCEPTION_MIN_IMAGE_SIZE
        ):
            return INCEPTION_MIN_IMAGE_SIZE
        return self.config.training.image_size

    def _maybe_generate_eda_plots(
        self,
        data_module: BrainTumorDataModule,
        plot_paths: dict[str, Path],
    ) -> None:
        if not self.config.outputs.save_eda_plots:
            return
        save_class_examples_plot(
            samples=data_module.train_samples,
            class_names=data_module.class_names,
            output_path=plot_paths["eda_class_samples_path"],
            seed=self.config.seed,
        )
        save_distribution_plot(
            train_samples=data_module.train_samples,
            val_samples=data_module.val_samples,
            test_samples=data_module.test_samples,
            class_names=data_module.class_names,
            output_path=plot_paths["eda_class_distribution_path"],
        )
        save_image_size_plot(
            samples=data_module.train_samples + data_module.val_samples + data_module.test_samples,
            output_path=plot_paths["eda_image_sizes_path"],
        )

    def _maybe_generate_prediction_grid(
        self,
        *,
        data_module: BrainTumorDataModule,
        model: nn.Module,
        device: torch.device,
        prediction_grid_path: Path,
    ) -> None:
        if not data_module.test_samples:
            return
        rng = random.Random(self.config.seed)
        sampled_test_images = data_module.test_samples[:]
        rng.shuffle(sampled_test_images)
        sampled_test_images = sampled_test_images[:5]
        save_prediction_grid(
            samples=sampled_test_images,
            loaded_models={self.config.model.name: model},
            class_names=data_module.class_names,
            transform=data_module.eval_transform,
            device=device,
            output_path=prediction_grid_path,
        )

    def _maybe_regenerate_report(self, run_paths: RunPaths) -> None:
        if not self.config.outputs.generate_report_after_run:
            return
        try:
            from brain_tumor_classifier.reporting.excel_report import (
                generate_excel_report,
            )
        except ImportError:
            return

        generate_excel_report(
            history_path=run_paths.experiment_history_path,
            output_path=run_paths.report_path,
        )

    def _build_success_metrics_payload(
        self,
        *,
        result: ExperimentResult,
        data_module: BrainTumorDataModule,
        checkpoint_path: str,
        last_checkpoint_path: str,
        run_paths: RunPaths,
        plot_paths: dict[str, str],
    ) -> dict[str, object]:
        return {
            "seed": self.config.seed,
            "experiment_name": self.config.experiment_name,
            "trace_name": self.config.trace_name,
            "run_id": result.run_id,
            "config_file": str(self.config.config_path),
            "config_hash": self.config.config_hash,
            "model": {
                "name": self.config.model.name,
                "architecture": result.architecture,
                "pretrained_requested": self.config.model.pretrained,
                "pretrained_used": result.pretrained_used,
            },
            "data": {
                "train_size": len(data_module.train_samples),
                "val_size": len(data_module.val_samples),
                "test_size": len(data_module.test_samples),
                "class_names": list(data_module.class_names),
            },
            "training": {
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "optimizer": self.config.training.optimizer,
                "image_size": result.image_size,
                "best_epoch": result.best_epoch,
            },
            "metrics": {
                "best_val_loss": result.best_val_loss,
                "best_val_accuracy": result.best_val_accuracy,
                "best_val_f1_weighted": result.best_val_f1_weighted,
                "test_loss": result.test_loss,
                "test_accuracy": result.test_accuracy,
                "test_f1_weighted": result.test_f1_weighted,
                "test_precision_weighted": result.test_precision_weighted,
                "test_recall_weighted": result.test_recall_weighted,
                "reloaded_test_accuracy": result.reloaded_test_accuracy,
                "reloaded_test_f1_weighted": result.reloaded_test_f1_weighted,
                "reload_f1_delta": (
                    result.reloaded_test_f1_weighted - result.test_f1_weighted
                ),
            },
            "artifacts": {
                "checkpoint_path": checkpoint_path,
                "last_checkpoint_path": last_checkpoint_path,
                "run_dir": str(run_paths.run_dir),
                "plots_dir": str(run_paths.plots_dir),
                "history_csv_path": str(run_paths.epoch_history_path),
                "plot_paths": plot_paths,
                "config_snapshot_path": str(run_paths.config_snapshot_path),
                "metrics_json_path": str(run_paths.metrics_json_path),
                "experiment_history_path": str(run_paths.experiment_history_path),
                "report_path": str(run_paths.report_path),
            },
            "created_at": result.created_at,
            "status": "success",
        }

    def _build_failed_metrics_payload(
        self,
        *,
        run_id: str,
        created_at: str,
        error_message: str,
        run_paths: RunPaths,
        plot_paths: dict[str, str],
    ) -> dict[str, object]:
        return {
            "seed": self.config.seed,
            "experiment_name": self.config.experiment_name,
            "trace_name": self.config.trace_name,
            "run_id": run_id,
            "config_file": str(self.config.config_path),
            "config_hash": self.config.config_hash,
            "model": {
                "name": self.config.model.name,
                "architecture": get_architecture_name(self.config.model.name),
                "pretrained_requested": self.config.model.pretrained,
                "pretrained_used": False,
            },
            "data": {
                "train_size": 0,
                "val_size": 0,
                "test_size": 0,
                "class_names": [],
            },
            "training": {
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "optimizer": self.config.training.optimizer,
                "image_size": self._resolve_image_size(),
                "best_epoch": 0,
            },
            "metrics": {
                "best_val_loss": None,
                "best_val_accuracy": None,
                "best_val_f1_weighted": None,
                "test_loss": None,
                "test_accuracy": None,
                "test_f1_weighted": None,
                "test_precision_weighted": None,
                "test_recall_weighted": None,
                "reloaded_test_accuracy": None,
                "reloaded_test_f1_weighted": None,
                "reload_f1_delta": None,
            },
            "artifacts": {
                "checkpoint_path": "",
                "last_checkpoint_path": "",
                "run_dir": str(run_paths.run_dir),
                "plots_dir": str(run_paths.plots_dir),
                "history_csv_path": str(run_paths.epoch_history_path),
                "plot_paths": plot_paths,
                "config_snapshot_path": str(run_paths.config_snapshot_path),
                "metrics_json_path": str(run_paths.metrics_json_path),
                "experiment_history_path": str(run_paths.experiment_history_path),
                "report_path": str(run_paths.report_path),
            },
            "created_at": created_at,
            "status": "failed",
            "error_message": error_message,
        }

    def _resolve_device(self, configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if configured_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine")
        return torch.device(configured_device)

    def _created_at(self) -> str:
        base_time = self.timestamp or datetime.now().astimezone()
        return base_time.astimezone().isoformat(timespec="seconds")
