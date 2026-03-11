import os
import hydra
import lightning as L

from omegaconf import DictConfig
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger  # , CometLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

from mltau.tools.io import ParT_dataloader as dl
from mltau.models import ParTau_module as pm


@hydra.main(config_path="../config", config_name="jetclass", version_base=None)
def train(cfg: DictConfig):
    datamodule = dl.ParTDataModule(cfg=cfg, debug_run=cfg.training.debug_run)
    model = pm.ParTauModule(cfg=cfg, input_dim=13, num_dm_classes=6)
    models_dir = os.path.join(cfg.output_dir, "models")
    log_dir = os.path.join(cfg.output_dir, "logs")
    tb_log_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Configure callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=1000),
        ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=-1,
            save_weights_only=True,
            filename="ParT-{epoch:02d}-{val_loss:.2f}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=callbacks,
        logger=[
            CSVLogger(log_dir, name="ParTau"),
            TensorBoardLogger(
                save_dir=tb_log_dir,
                name="ParTau_experiment",
                log_graph=True,  # Log model graph
                default_hp_metric=False,
            ),
        ],
        overfit_batches=50,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
