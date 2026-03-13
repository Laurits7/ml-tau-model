import os
import numpy as np
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib import ticker

from omegaconf import DictConfig
from mltau.tools import general as g
from mltau.tools.evaluation.histogram import Histogram


class ChargeIdEvaluator:
    """Class for evaluating charge ID performance."""

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        gen_jet_tau_p4s: ak.Array,
        reco_jet_p4s: ak.Array,
        cfg: DictConfig,
        output_dir: str = "",
        sample: str = "",
        algorithm: str = "",
        n_classifier_cuts: int = 100,
    ):
        self.output_dir = output_dir
        if self.output_dir != "":
            os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        self.predicted = predicted
        self.gen_jet_tau_p4s = g.reinitialize_p4(gen_jet_tau_p4s)
        self.reco_jet_p4s = g.reinitialize_p4(reco_jet_p4s)
        self.cfg = cfg
        self.truth = truth
        self.tagging_cuts = np.linspace(start=0, stop=1, num=n_classifier_cuts + 1)
        self.true_positive_charge_mask = self.truth == 1
        self.true_negative_charge_mask = self.truth == 0
        self.efficiencies, self.eff_denominator_masks = self._calculate_eff_fake(
            eff_fake="eff"
        )
        self.fakerates, self.fake_denominator_masks = self._calculate_eff_fake(
            eff_fake="fake"
        )
        self.pos_charge_predictions = self.predicted[self.true_positive_charge_mask]
        self.neg_charge_predictions = self.predicted[self.true_negative_charge_mask]
        self.wp_value = 0.5
        self.wp_metrics = {}
        for name, metric in cfg.metrics.charge.metrics.items():
            self.wp_metrics[name] = {}
            charge_fr = self._get_working_point_eff_fakes(name, metric, eff_fake="fake")
            charge_eff = self._get_working_point_eff_fakes(name, metric, eff_fake="eff")
            for charge in ["negative", "positive"]:
                fr_bin_centers, fr_data, fr_yerr, fr_xerr = charge_fr[charge]
                eff_bin_centers, eff_data, eff_yerr, eff_xerr = charge_eff[charge]
                self.wp_metrics[name][charge] = {
                    "fakerates": fr_data,
                    "fr_bin_centers": fr_bin_centers,
                    "fr_yerr": fr_yerr,
                    "fr_xerr": fr_xerr,
                    "efficiencies": eff_data,
                    "eff_bin_centers": eff_bin_centers,
                    "eff_yerr": eff_yerr,
                    "eff_xerr": eff_xerr,
                }

    def _calculate_eff_fake(self, eff_fake: str = "eff"):
        _eff_fake = {"positive": [], "negative": []}
        positive_mask = (
            self.true_positive_charge_mask
            if eff_fake == "eff"
            else self.true_negative_charge_mask
        )
        negative_mask = (
            self.true_negative_charge_mask
            if eff_fake == "eff"
            else self.true_positive_charge_mask
        )
        ref_var_pt_mask = self.gen_jet_tau_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        ref_var_theta_mask1 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        ref_var_theta_mask2 = (
            abs(np.rad2deg(self.gen_jet_tau_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        gen_denominator_mask = (
            ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2
        )
        # As we are only assigning charge for tau (candidates) that are tagged, then to have the correct total
        # number of jets, we need to add the cuts on the reco jet also to the denominator
        tau_pt_mask = self.reco_jet_p4s.pt > self.cfg.metrics.tagging.cuts.min_pt
        tau_theta_mask1 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            < self.cfg.metrics.tagging.cuts.max_theta
        )
        tau_theta_mask2 = (
            abs(np.rad2deg(self.reco_jet_p4s.theta))
            > self.cfg.metrics.tagging.cuts.min_theta
        )
        reco_denominator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2
        base_denominator_mask = gen_denominator_mask * reco_denominator_mask

        pos_denominator_mask = base_denominator_mask * positive_mask
        neg_denominator_mask = base_denominator_mask * negative_mask
        denominator_masks = {
            "positive": pos_denominator_mask,
            "negative": neg_denominator_mask,
        }
        neg_all = np.sum(neg_denominator_mask)
        pos_all = np.sum(pos_denominator_mask)
        for cut in self.tagging_cuts:
            pos_passing_cut = np.sum(self.predicted[pos_denominator_mask] >= cut)
            neg_passing_cut = np.sum(self.predicted[neg_denominator_mask] < cut)
            _eff_fake["positive"].append(pos_passing_cut / pos_all)
            _eff_fake["negative"].append(neg_passing_cut / neg_all)
        return _eff_fake, denominator_masks

    ###################################
    ###################################
    ###################################
    ###################################

    def _get_working_point_eff_fakes(self, name, metric, eff_fake="eff"):
        eff_fake_mask = (
            self.eff_denominator_masks
            if eff_fake == "eff"
            else self.fake_denominator_masks
        )
        return_values = {}
        var_values = getattr(self.gen_jet_tau_p4s, name).to_numpy()
        if name == "theta":
            var_values = np.rad2deg(var_values)
        for charge in ["positive", "negative"]:
            if charge == "positive":
                wp_mask = self.predicted > self.wp_value
            else:
                wp_mask = self.predicted < self.wp_value
            eff_var_denom = var_values[eff_fake_mask[charge]]
            eff_var_num = var_values[wp_mask * eff_fake_mask[charge]]
            bin_edges = np.linspace(
                min(eff_var_denom), max(eff_var_denom), metric.n_bins + 1
            )
            denom_hist = Histogram(eff_var_denom, bin_edges, "denominator")
            num_hist = Histogram(eff_var_num, bin_edges, "numerator")
            efficiencies = num_hist / denom_hist
            return_values[charge] = (
                efficiencies.bin_centers,
                efficiencies.data,
                efficiencies.uncertainties,
                efficiencies.bin_halfwidths,
            )
        return return_values


class ChargeClassifierPlot:
    def __init__(self):
        self.bin_edges = np.linspace(start=0, stop=1, num=21)
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator, dataset: str):
        linestyle = "solid" if dataset == "test" else "dashed"
        neg_histogram = np.histogram(
            evaluator.neg_charge_predictions, bins=self.bin_edges
        )[0]
        neg_histogram = neg_histogram / np.sum(neg_histogram)
        pos_histogram = np.histogram(
            evaluator.pos_charge_predictions, bins=self.bin_edges
        )[0]
        pos_histogram = pos_histogram / np.sum(pos_histogram)
        hep.histplot(
            pos_histogram,
            bins=self.bin_edges,
            histtype="step",
            label=r"$\tau^{+}$",
            ls=linestyle,
            color="red",
            ax=self.ax,
        )
        hep.histplot(
            neg_histogram,
            bins=self.bin_edges,
            histtype="step",
            label=r"$\tau^{-}$",
            ls=linestyle,
            color="blue",
            ax=self.ax,
        )
        self.ax.legend(prop={"size": 28})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(r"$\mathcal{D}$", fontdict={"size": 28})
        ax.set_yscale("log")
        ax.set_ylabel("Relative yield / bin")
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class ROCPlot:
    def __init__(self, cfg):
        self.fig, self.ax = self.plot()
        self.cfg = cfg

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel(r"$P_{misid}$", fontsize=30)
        ax.set_xlabel(r"$\varepsilon_{\tau}$", fontsize=30)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        ax.set_ylim((1e-5, 1))
        ax.set_xlim((0, 1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.set_yscale("log")
        plt.grid()
        return fig, ax

    def add_line(self, evaluator):
        self.ax.plot(
            evaluator.efficiencies["positive"],
            evaluator.fakerates["positive"],
            color="r",
            marker="s",
            label=r"$\tau^{+}$",
            ms=15,
            ls="",
        )
        self.ax.plot(
            evaluator.efficiencies["negative"],
            evaluator.fakerates["negative"],
            color="b",
            marker="^",
            label=r"$\tau^{-}$",
            ms=15,
            ls="",
        )
        self.ax.legend(prop={"size": 30})

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class EfficiencyPlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["eff_yerr"],
            ms=20,
            color="b",
            marker="^",
            linestyle="",
            label=r"$\tau^{-}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["eff_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["efficiencies"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["eff_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["eff_yerr"],
            ms=20,
            color="r",
            marker="s",
            linestyle="",
            label=r"$\tau^{+}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.efficiency.ylabel}",
            fontsize=30,
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.efficiency.yscale)
        if self.cfg.metrics.tagging.performances.efficiency.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.efficiency.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.efficiency.ylim
        ax.set_ylim(tuple(ylim))
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")


class FakeRatePlot:
    def __init__(self, cfg: DictConfig, metric: str):
        self.cfg = cfg
        self.metric = metric
        self.fig, self.ax = self.plot()

    def add_line(self, evaluator):
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["negative"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["negative"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["negative"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["negative"]["fr_yerr"],
            ms=20,
            color="b",
            marker="^",
            linestyle="",
            label=r"$\tau^{-}$",
        )
        self.ax.errorbar(
            evaluator.wp_metrics[self.metric]["positive"]["fr_bin_centers"],
            evaluator.wp_metrics[self.metric]["positive"]["fakerates"],
            xerr=evaluator.wp_metrics[self.metric]["positive"]["fr_xerr"],
            yerr=evaluator.wp_metrics[self.metric]["positive"]["fr_yerr"],
            ms=20,
            color="r",
            marker="s",
            linestyle="",
            label=r"$\tau^{+}$",
        )
        self.ax.legend(loc="upper right", prop={"size": 30})

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(
                self.cfg.metrics.tagging.metrics[self.metric].x_maj_tick_spacing
            )
        )
        ax.set_xlabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.xlabel[self.metric]}",
            fontsize=30,
        )
        ax.set_ylabel(
            rf"{self.cfg.metrics.tagging.performances.fakerate.ylabel}", fontsize=30
        )
        ax.set_yscale(self.cfg.metrics.tagging.performances.fakerate.yscale)
        if self.cfg.metrics.tagging.performances.fakerate.ylim is not None:
            ylim = tuple(self.cfg.metrics.tagging.performances.fakerate.ylim)
        else:
            ylim = self.cfg.metrics.tagging.performances.fakerate.ylim
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=30)
        ax.tick_params(axis="y", labelsize=30)
        plt.grid()
        return fig, ax

    def save(self, output_path):
        self.fig.savefig(output_path, format="pdf")
        plt.close("all")
