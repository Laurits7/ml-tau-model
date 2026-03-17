import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from mltau.tools.general import reinitialize_p4
from mltau.tools.evaluation import tagging as t
from mltau.tools.logging.general import log_metrics_dict


def log_all_tagging_metrics(
    targets: np.array,
    gen_jet_p4s: np.array,
    gen_jet_tau_p4s: np.array,
    reco_jet_p4s: np.array,
    predictions: np.array,
    cfg: DictConfig,
    tb_logger,
    # output_dir: str,
    current_epoch: int,
    dataset="train",
):
    predictions = predictions["is_tau"]  # charge, kinematics, decay_mode
    targets = targets["is_tau"]
    # weights = weight
    sig_mask = targets == 1
    bkg_mask = targets == 0

    sig_gen_tau_p4 = reinitialize_p4(gen_jet_tau_p4s[sig_mask])
    bkg_gen_jet_p4s = reinitialize_p4(gen_jet_p4s[bkg_mask])

    sig_reco_jet_p4 = reinitialize_p4(reco_jet_p4s[sig_mask])
    bkg_reco_jet_p4s = reinitialize_p4(reco_jet_p4s[bkg_mask])

    tagger_evaluator = t.TaggerEvaluator(
        signal_predictions=predictions[sig_mask],
        signal_gen_tau_p4=sig_gen_tau_p4,  # gen_jet_tau
        signal_reco_jet_p4=sig_reco_jet_p4,
        bkg_predictions=predictions[bkg_mask],
        bkg_gen_jet_p4=bkg_gen_jet_p4s,
        bkg_reco_jet_p4=bkg_reco_jet_p4s,
        cfg=cfg,
        sample="all",
        algorithm="all",
    )
    metrics = list(cfg.metrics.tagging.metrics.keys())
    classifier_plot = t.TauClassifierPlot()
    classifier_plot.add_line(tagger_evaluator, dataset)
    tb_logger.add_figure("tagging/classifier", classifier_plot.fig, current_epoch)
    plt.close(classifier_plot.fig)

    roc_plot = t.ROCPlot(cfg)
    roc_plot.add_line(tagger_evaluator)
    tb_logger.add_figure("tagging/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)

    efficiency_plots = {metric: t.EfficiencyPlot(cfg, metric) for metric in metrics}
    fakerate_plots = {metric: t.FakeRatePlot(cfg, metric) for metric in metrics}

    for metric in metrics:
        efficiency_plots[metric].add_line(tagger_evaluator)
        tb_logger.add_figure(
            f"tagging/{metric}_efficiency", efficiency_plots[metric].fig, current_epoch
        )
        plt.close(efficiency_plots[metric].fig)
        fakerate_plots[metric].add_line(tagger_evaluator)
        tb_logger.add_figure(
            f"tagging/{metric}_fakerate", fakerate_plots[metric].fig, current_epoch
        )
        plt.close(fakerate_plots[metric].fig)
    # No need to add wp values probably.

    # TODO: Calculate AUC?

    # Scalar classification metrics at the medium WP threshold
    wp = tagger_evaluator.medium_wp
    tp = float(np.sum(np.array(tagger_evaluator.signal_predictions) > wp))
    fn = float(np.sum(np.array(tagger_evaluator.signal_predictions) <= wp))
    tn = float(np.sum(np.array(tagger_evaluator.bkg_predictions) <= wp))
    fp = float(np.sum(np.array(tagger_evaluator.bkg_predictions) > wp))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    tagging_scalars = {
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": accuracy,
    }
    log_metrics_dict(tb_logger, tagging_scalars, "tagging", current_epoch)

    # Now log all the plots
