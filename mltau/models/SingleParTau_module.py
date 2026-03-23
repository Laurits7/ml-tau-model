import torch
import awkward as ak
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import SigmoidFocalLoss
from mltau.tools.logging import tagging, kinematics, decay_mode, charge_id
from mltau.models.SingleParTau import ParTau

VALID_TASKS = {"is_tau", "charge", "decay_mode", "kinematics"}


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int, task: str):
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got '{task}'")
        self.cfg = cfg
        self.task = task
        self.ParTau = ParTau(
            input_dim=input_dim,
            task=task,
            num_dm_classes=num_dm_classes,
            num_layers=2,
            embed_dims=[256, 512, 256],
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="eta-phi",
        )
        if task == "is_tau":
            self.loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        elif task == "charge":
            self.loss_fn = SigmoidFocalLoss(reduction="none")
        elif task == "decay_mode":
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        elif task == "kinematics":
            self.loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    def _loss_key(self):
        return f"{self.task}_loss"

    def _make_accumulator(self):
        return {key: [] for key in ["loss", self._loss_key()]}

    def training_step(self, batch, batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        for key, value in metrics.items():
            self.training_loss_accumulator[key].append(value.detach())
        inputs = BatchInputs(*batch)
        if batch_idx % 10 == 0:
            self.training_outputs.append(
                {
                    "predictions": predictions,
                    "targets": targets,
                    "gen_jet_p4s": inputs.gen_jet_p4s,
                    "reco_jet_p4s": inputs.reco_jet_p4s,
                    "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
                }
            )
        return metrics["loss"]

    def predict_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            params=self.ParTau.parameters(),
            lr=self.cfg.training.lr,
            betas=(0.95, 0.999),
            eps=1e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20000 * self.cfg.training.trainer.max_epochs,
            eta_min=self.cfg.training.lr * 0.01,
        )
        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        inputs = BatchInputs(*batch)
        model_output = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        # model_output is a tuple (tensor,); wrap in dict with the task key
        predictions = {self.task: model_output[0]}
        return predictions, inputs.target, inputs.weight

    def calculate_metrics(self, targets, predictions, weights):
        pred = predictions[self.task]
        target = targets[self.task]

        if self.task == "kinematics":
            raw_loss = self.loss_fn(pred, target).mean(dim=-1)
            is_tau_mask = targets["is_tau"].bool()
            loss = raw_loss[is_tau_mask].mean()
        elif self.task == "is_tau":
            loss = self.loss_fn(pred, target).mean()
        else:  # "charge" or "decay_mode" — only meaningful for signal taus
            raw_loss = self.loss_fn(pred, target)
            is_tau_mask = targets["is_tau"].bool()
            loss = raw_loss[is_tau_mask].mean()

        return {"loss": loss, self._loss_key(): loss}

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        inputs = BatchInputs(*batch)
        self.validation_outputs.append(
            {
                "predictions": predictions,
                "targets": targets,
                "gen_jet_p4s": inputs.gen_jet_p4s,
                "reco_jet_p4s": inputs.reco_jet_p4s,
                "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
            }
        )
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())
        return metrics["loss"]

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_loss_accumulator = self._make_accumulator()

    def _log_task_metrics(
        self,
        targets,
        predictions,
        gen_jet_p4s,
        gen_jet_tau_p4s,
        reco_jet_p4s,
        tb_logger,
        current_epoch,
        dataset,
    ):
        kwargs = dict(
            targets=targets,
            predictions=predictions,
            tb_logger=tb_logger,
            current_epoch=current_epoch,
        )
        if self.task == "is_tau":
            tagging.log_all_tagging_metrics(
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )
        elif self.task == "charge":
            charge_id.log_charge_id_performance(
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )
        elif self.task == "decay_mode":
            decay_mode.log_all_decay_mode_metrics(**kwargs)
        elif self.task == "kinematics":
            kinematics.log_all_kinematics_metrics(
                reco_jet_p4s=reco_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                cfg=self.cfg,
                dataset=dataset,
                **kwargs,
            )

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = (
            self.validation_outputs if dataset == "val" else self.training_outputs
        )

        if dataset_outputs:
            all_predictions = {}
            all_targets = {}
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}

            for output in dataset_outputs:
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                for key, value in output["gen_jet_p4s"].items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["reco_jet_p4s"].items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["gen_jet_tau_p4s"].items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

            for key in all_predictions:
                all_predictions[key] = ak.concatenate(all_predictions[key], axis=0)
            for key in all_targets:
                all_targets[key] = ak.concatenate(all_targets[key], axis=0)
            for key in all_gen_jet_p4s:
                all_gen_jet_p4s[key] = ak.concatenate(all_gen_jet_p4s[key], axis=0)
            for key in all_reco_jet_p4s:
                all_reco_jet_p4s[key] = ak.concatenate(all_reco_jet_p4s[key], axis=0)
            for key in all_gen_jet_tau_p4s:
                all_gen_jet_tau_p4s[key] = ak.concatenate(
                    all_gen_jet_tau_p4s[key], axis=0
                )

            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            self._log_task_metrics(
                targets=all_targets,
                predictions=all_predictions,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                tb_logger=self.logger.experiment,
                current_epoch=self.current_epoch,
                dataset=dataset,
            )

            dataset_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            epoch_metrics = {
                k: torch.stack(v).mean()
                for k, v in self.validation_loss_accumulator.items()
                if v
            }
            for k, v in epoch_metrics.items():
                self.log(f"val_losses/{k}", v)
        self._log_at_epoch_end(dataset="val")

    def on_train_epoch_start(self):
        self.training_outputs = []
        self.training_loss_accumulator = self._make_accumulator()

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
        self._log_at_epoch_end(dataset="train")
