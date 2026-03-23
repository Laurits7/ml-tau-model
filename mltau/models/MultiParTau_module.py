import torch
import awkward as ak
import torch.nn as nn
import lightning as L
from omegaconf import DictConfig

# from mltau.tools.optimizers.lookahead import Lookahead
from mltau.tools.io.general import BatchInputs
from mltau.tools.losses import SigmoidFocalLoss
from mltau.tools.logging import logger
from mltau.models.MultiParTau import ParTau


class ParTauModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, input_dim: int, num_dm_classes: int):
        super().__init__()
        self.cfg = cfg
        self.ParTau = ParTau(
            input_dim=input_dim,
            num_dm_classes=num_dm_classes,  # Number of decay modes we wish to classify
            num_layers=2,  # cfg.models.ParticleTransformer.hyperparameters.num_layers,
            embed_dims=[
                256,
                512,
                256,
            ],  # cfg.models.ParticleTransformer.hyperparameters.embed_dims,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric="eta-phi",
        )
        self.tagging_loss_fn = SigmoidFocalLoss(
            alpha=0.25, gamma=2.0, reduction="none"
        )  # class imbalance
        self.charge_loss_fn = SigmoidFocalLoss(reduction="none")  # class balance
        self.decay_mode_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.kinematics_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    def training_step(self, batch, batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        for key, value in metrics.items():
            self.training_loss_accumulator[key].append(value.detach())
        inputs = BatchInputs(*batch)
        if batch_idx % 10 == 0:  # Store every 10th batch to save memory
            output = {
                "predictions": predictions,
                "targets": targets,
                # "weights": weights,
                "gen_jet_p4s": inputs.gen_jet_p4s,
                "reco_jet_p4s": inputs.reco_jet_p4s,
                "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
            }
            self.training_outputs.append(output)
        return metrics["loss"]

    def predict_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def test_step(self, batch, _batch_idx):
        return self.forward(batch)[0]

    def configure_optimizers(self):
        # Consider using AdamW
        optimizer = torch.optim.RAdam(
            params=self.ParTau.parameters(),
            lr=self.cfg.training.lr,
            betas=(0.95, 0.999),
            eps=1e-5,
        )
        # if self.cfg.training.optimizer.use_lookahead:
        #     optimizer = Lookahead(base_optimizer=optimizer, k=6, alpha=0.5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20000 * self.cfg.training.trainer.max_epochs,
            # T_max=len(dataloader_train) * cfg.training.num_epochs,
            eta_min=self.cfg.training.lr * 0.01,
        )
        # LR remains constant for the first 70% of iterations, then decays exponentially at an
        # interval of every 20k iterations down to 1% of the inital value at the end of the training
        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        """Both `predictions` and `targets` are defined for the multiple heads"""
        # Unpack batch components
        inputs = BatchInputs(*batch)

        predictions = self.ParTau(
            cand_features=inputs.cand_features,
            cand_kinematics_pxpypze=inputs.cand_kinematics_pxpypze,
            cand_mask=inputs.cand_mask,
        )
        return predictions, inputs.target, inputs.weight

    def calculate_metrics(self, targets, predictions, weights):

        # TODO: Account for weights.

        decay_mode_loss = self.decay_mode_loss_fn(
            predictions["decay_mode"], targets["decay_mode"]  # miks siin pole squeeze?
        )

        # Tau ID
        tau_id_loss = self.tagging_loss_fn(predictions["is_tau"], targets["is_tau"])

        # Charge
        charge_loss = self.charge_loss_fn(predictions["charge"], targets["charge"])

        # Kinematics
        kin_pred = predictions["kinematics"]
        kin_target = targets["kinematics"]

        # Log-ratio terms: independent Huber in log space
        log_pt_loss = self.kinematics_loss_fn(kin_pred[:, 0], kin_target[:, 0])
        log_m_loss = self.kinematics_loss_fn(kin_pred[:, 3], kin_target[:, 3])

        # Angle terms: combined deltaR-like Huber instead of two independent terms
        delta_angle = torch.sqrt(
            (kin_pred[:, 1] - kin_target[:, 1]) ** 2
            + (kin_pred[:, 2] - kin_target[:, 2]) ** 2
            + 1e-8
        )
        angle_loss = self.kinematics_loss_fn(delta_angle, torch.zeros_like(delta_angle))

        kinematics_loss = (log_pt_loss + angle_loss + log_m_loss) / 3.0

        combined_loss = (
            tau_id_loss
            + (decay_mode_loss + charge_loss + kinematics_loss) * targets["is_tau"]
        )  # Here use all losses only if signal sample.

        is_tau_mask = targets["is_tau"].bool()
        metrics = {
            "tau_id_loss": tau_id_loss.mean(),
            "charge_loss": charge_loss[is_tau_mask].mean(),
            "decay_mode_loss": decay_mode_loss[is_tau_mask].mean(),
            "kinematics_loss": kinematics_loss[is_tau_mask].mean(),
            "loss": combined_loss.mean(),
        }

        # TODO: Calculate additional metrics

        return metrics

    def validation_step(self, batch, _batch_idx):
        predictions, targets, weights = self.forward(batch)
        metrics = self.calculate_metrics(
            targets=targets, predictions=predictions, weights=weights
        )
        inputs = BatchInputs(*batch)
        output = {
            "predictions": predictions,
            "targets": targets,
            # "weights": weights,
            "gen_jet_p4s": inputs.gen_jet_p4s,
            "reco_jet_p4s": inputs.reco_jet_p4s,
            "gen_jet_tau_p4s": inputs.gen_jet_tau_p4s,
        }
        self.validation_outputs.append(output)
        for key, value in metrics.items():
            self.validation_loss_accumulator[key].append(value.detach())
        return metrics["loss"]

    def on_validation_epoch_start(self):
        """Initialize storage for validation outputs."""
        self.validation_outputs = []
        self.validation_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
            ]
        }

    def _log_at_epoch_end(self, dataset: str):
        if dataset == "val" and self.trainer.sanity_checking:
            return

        dataset_outputs = (
            self.validation_outputs if dataset == "val" else self.training_outputs
        )

        if dataset_outputs:
            # Aggregate all predictions, targets, and weights
            all_predictions = {}
            all_targets = {}
            # all_weights = []
            all_gen_jet_p4s = {}
            all_gen_jet_tau_p4s = {}
            all_reco_jet_p4s = {}

            for output in dataset_outputs:
                # Concatenate predictions for each head
                for key, pred in output["predictions"].items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.detach().cpu())

                # Concatenate targets
                for key, target in output["targets"].items():
                    if key not in all_targets:
                        all_targets[key] = []
                    all_targets[key].append(target.detach().cpu())

                # Concatenate gen_jet_p4s
                for key, value in output["gen_jet_p4s"].items():
                    if key not in all_gen_jet_p4s:
                        all_gen_jet_p4s[key] = []
                    all_gen_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                # Concatenate reco_jet_p4s
                for key, value in output["reco_jet_p4s"].items():
                    if key not in all_reco_jet_p4s:
                        all_reco_jet_p4s[key] = []
                    all_reco_jet_p4s[key].append(ak.Array(value.detach().cpu()))

                for key, value in output["gen_jet_tau_p4s"].items():
                    if key not in all_gen_jet_tau_p4s:
                        all_gen_jet_tau_p4s[key] = []
                    all_gen_jet_tau_p4s[key].append(ak.Array(value.detach().cpu()))

                # Concatenate weights
                # all_weights.append(output["weights"].detach().cpu())

            # Convert lists to tensors
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

            # Convert dictionaries back to awkward arrays with fields for reinitialize_p4
            gen_jet_p4s = ak.Array(all_gen_jet_p4s)
            reco_jet_p4s = ak.Array(all_reco_jet_p4s)
            gen_jet_tau_p4s = ak.Array(all_gen_jet_tau_p4s)

            # all_weights = ak.concatenate(all_weights, dim=0)

            # Log comprehensive metrics with full validation dataset
            current_epoch = self.current_epoch
            tb_logger = self.logger.experiment
            logger.log_all(
                targets=all_targets,
                gen_jet_p4s=gen_jet_p4s,
                gen_jet_tau_p4s=gen_jet_tau_p4s,
                reco_jet_p4s=reco_jet_p4s,
                predictions=all_predictions,
                cfg=self.cfg,
                tb_logger=tb_logger,
                current_epoch=current_epoch,
                dataset=dataset,
            )

            # Clear outputs to free memory
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
        """Initialize storage for training outputs."""
        self.training_outputs = []
        self.training_loss_accumulator = {
            key: []
            for key in [
                "loss",
                "tau_id_loss",
                "charge_loss",
                "decay_mode_loss",
                "kinematics_loss",
            ]
        }

    def on_train_epoch_end(self):
        epoch_metrics = {
            k: torch.stack(v).mean()
            for k, v in self.training_loss_accumulator.items()
            if v
        }
        for k, v in epoch_metrics.items():
            self.log(f"train_losses/{k}", v)
        self._log_at_epoch_end(dataset="train")
