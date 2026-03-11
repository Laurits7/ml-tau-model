import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mltau.tools.evaluation import kinematics as k
from mltau.tools.general import reinitialize_p4


def log_all_kinematics_metrics(
    targets: np.array,
    predictions: np.array,
    reco_jet_p4s: np.array,
    cfg: DictConfig,
    tb_logger,
    # output_dir: str,
    current_epoch: int,
    dataset="train",
):
    signal_predictions = predictions["kinematics"][targets["is_tau"] == 1]
    signal_targets = targets["kinematics"][targets["is_tau"] == 1]

    reco_jet_p4s = reinitialize_p4(reco_jet_p4s)[targets["is_tau"] == 1]
    pred_pt = np.exp(signal_predictions[:, 0]) * reco_jet_p4s.pt
    true_pt = np.exp(signal_targets[:, 0]) * reco_jet_p4s.pt

    # PT-energy magnitude evaluation
    evaluator = k.RegressionEvaluator(
        prediction=pred_pt,
        truth=true_pt,
        cfg=cfg,
        sample_name="all",
        algorithm="all",
    )

    response_lineplot = k.LinePlot(
        cfg=cfg,
        xlabel=cfg.metrics.kinematics.ratio_plot.response_plot.xlabel,
        ylabel=cfg.metrics.kinematics.ratio_plot.response_plot.ylabel,
        xscale=cfg.metrics.kinematics.ratio_plot.response_plot.xscale,
        yscale=cfg.metrics.kinematics.ratio_plot.response_plot.yscale,
        ymin=cfg.metrics.kinematics.ratio_plot.response_plot.ylim[0],
        ymax=cfg.metrics.kinematics.ratio_plot.response_plot.ylim[1],
        nticks=cfg.metrics.kinematics.ratio_plot.response_plot.nticks,
    )
    response_lineplot.add_line(
        evaluator.bin_centers,
        evaluator.responses,
        evaluator.algorithm,
        label="",
    )

    tb_logger.add_figure(
        "kinematics/response_vs_pt", response_lineplot.fig, current_epoch
    )
    plt.close(response_lineplot.fig)

    resolution_lineplot = k.LinePlot(
        cfg=cfg,
        xlabel=cfg.metrics.kinematics.ratio_plot.resolution_plot.xlabel,
        ylabel=cfg.metrics.kinematics.ratio_plot.resolution_plot.ylabel,
        xscale=cfg.metrics.kinematics.ratio_plot.resolution_plot.xscale,
        yscale=cfg.metrics.kinematics.ratio_plot.resolution_plot.yscale,
        ymin=cfg.metrics.kinematics.ratio_plot.resolution_plot.ylim[0],
        ymax=cfg.metrics.kinematics.ratio_plot.resolution_plot.ylim[1],
        nticks=cfg.metrics.kinematics.ratio_plot.resolution_plot.nticks,
    )
    resolution_lineplot.add_line(
        evaluator.bin_centers,
        evaluator.resolutions,
        evaluator.algorithm,
        label="",
    )

    tb_logger.add_figure(
        "kinematics/resolution_vs_pt", resolution_lineplot.fig, current_epoch
    )
    plt.close(resolution_lineplot.fig)

    resolution_2d_plot = k.Resolution2DPlot(cfg, "all", evaluator)
    tb_logger.add_figure(
        "kinematics/resolution_2d", resolution_2d_plot.fig, current_epoch
    )
    plt.close(resolution_2d_plot.fig)

    bin_distributions = k.RangeContentPlot(cfg, "all")
    bin_distributions.add_line(evaluator)
    tb_logger.add_figure(
        "kinematics/bin_distributions", bin_distributions.fig, current_epoch
    )
    plt.close(bin_distributions.fig)

    tb_logger.add_scalar("kinematics/resolution", evaluator.resolution, current_epoch)
    tb_logger.add_scalar("kinematics/response", evaluator.response, current_epoch)
