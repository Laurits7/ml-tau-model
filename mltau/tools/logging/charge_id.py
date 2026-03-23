import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from mltau.tools.evaluation import charge_id as c
from mltau.tools.evaluation.general import binary_classifier_metrics
from mltau.tools.logging.general import log_metrics_dict


def log_charge_id_performance(
    targets: np.array,
    gen_jet_tau_p4s: np.array,
    reco_jet_p4s: np.array,
    predictions: np.array,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    dataset="train",
):
    # Charge is only meaningful for signal taus — exclude background jets
    signal_mask = targets["is_tau"] == 1
    predictions = predictions["charge"][signal_mask]
    targets = targets["charge"][signal_mask]
    gen_jet_tau_p4s = gen_jet_tau_p4s[signal_mask]
    reco_jet_p4s = reco_jet_p4s[signal_mask]

    evaluator = c.ChargeIdEvaluator(
        predicted=predictions,
        truth=targets,
        gen_jet_tau_p4s=gen_jet_tau_p4s,
        reco_jet_p4s=reco_jet_p4s,
        cfg=cfg,
        sample="all",
        algorithm="all",
    )

    # Classifier plot
    classifier_plot = c.ChargeClassifierPlot()
    classifier_plot.add_line(evaluator, dataset)
    tb_logger.add_figure("charge_id/classifier", classifier_plot.fig, current_epoch)
    plt.close(classifier_plot.fig)

    # ROC plot
    roc_plot = c.ROCPlot(cfg)
    roc_plot.add_line(evaluator)
    tb_logger.add_figure("charge_id/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)

    # Per-metric efficiency and fakerate plots
    metrics = list(cfg.metrics.charge.metrics.keys())
    for metric in metrics:
        eff_plot = c.EfficiencyPlot(cfg, metric)
        eff_plot.add_line(evaluator)
        tb_logger.add_figure(
            f"charge_id/{metric}_efficiency", eff_plot.fig, current_epoch
        )
        plt.close(eff_plot.fig)

        fr_plot = c.FakeRatePlot(cfg, metric)
        fr_plot.add_line(evaluator)
        tb_logger.add_figure(f"charge_id/{metric}_fakerate", fr_plot.fig, current_epoch)
        plt.close(fr_plot.fig)

    # Scalar classification metrics at the ROC-intersection working point
    charge_scalars = binary_classifier_metrics(
        evaluator.predicted, evaluator.truth, evaluator.wp_pos
    )
    charge_scalars["wp_pos"] = evaluator.wp_pos
    charge_scalars["wp_neg"] = evaluator.wp_neg
    log_metrics_dict(tb_logger, charge_scalars, "charge_id", current_epoch)
