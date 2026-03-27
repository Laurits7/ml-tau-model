import warnings
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mltau.tools.general import reinitialize_p4

warnings.filterwarnings("ignore", message=".*sumw are zero.*", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message=".*divide by zero encountered in scalar divide.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in multiply.*",
    category=RuntimeWarning,
)


def _wrap_delta_phi(delta_phi):
    return np.arctan2(np.sin(delta_phi), np.cos(delta_phi))


def _finite(values):
    values = np.asarray(values)
    return values[np.isfinite(values)]


def _q50(values):
    values = _finite(values)
    return np.quantile(values, 0.5) if values.size else np.nan


def _iqr(values):
    values = _finite(values)
    if values.size == 0:
        return np.nan
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return q75 - q25


def _iqr_over_q50(values):
    values = _finite(values)
    if values.size == 0:
        return np.nan
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return (q75 - q25) / q50 if q50 != 0 else np.nan


def _compute_binned_stats(x, bins, y, stat_fn):
    x = np.asarray(x)
    y = np.asarray(y)
    centers = 0.5 * (bins[:-1] + bins[1:])
    stats = []
    counts = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (x >= left) & (x < right) & np.isfinite(y) & np.isfinite(x)
        values = y[mask]
        counts.append(int(mask.sum()))
        stats.append(stat_fn(values) if values.size else np.nan)
    return centers, np.asarray(stats), np.asarray(counts)


def _distribution_figure(model_values, reco_values, bins, xlabel, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    reco_values = _finite(reco_values)
    model_values = _finite(model_values)
    if reco_values.size:
        ax.hist(
            reco_values, bins=bins, density=True, histtype='step', linewidth=2.0,
            color='tab:orange', label='Reco jet'
        )
    if model_values.size:
        ax.hist(
            model_values, bins=bins, density=True, histtype='step', linewidth=2.0,
            linestyle='--', color='tab:red', label='Model'
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized entries')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    return fig


def _overlay_curve_figure(x_reco, y_reco, x_model, y_model, bins, stat_fn, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    centers, reco_curve, _ = _compute_binned_stats(x_reco, bins, y_reco, stat_fn)
    _, model_curve, _ = _compute_binned_stats(x_model, bins, y_model, stat_fn)
    ax.plot(centers, reco_curve, marker='o', linewidth=2.0, color='tab:orange', label='Reco jet')
    ax.plot(centers, model_curve, marker='s', linewidth=2.0, linestyle='--', color='tab:red', label='Model')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    return fig


def _log_ratio_family(tb_logger, current_epoch, prefix, pred, truth, reco_pred, truth_x, bins, x_label, ratio_label):
    response_fig = _overlay_curve_figure(
        truth_x, reco_pred / truth, truth_x, pred / truth, bins, _q50,
        x_label, f'$q_{{50}}({ratio_label})$', f'{prefix} response vs truth'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/response_vs_truth', response_fig, current_epoch)
    plt.close(response_fig)

    resolution_fig = _overlay_curve_figure(
        truth_x, reco_pred / truth, truth_x, pred / truth, bins, _iqr_over_q50,
        x_label, r'$(q_{75} - q_{25}) / q_{50}$', f'{prefix} resolution vs truth'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/resolution_vs_truth', resolution_fig, current_epoch)
    plt.close(resolution_fig)

    dist_fig = _distribution_figure(pred / truth, reco_pred / truth, np.linspace(0.5, 1.5, 90), ratio_label, f'{prefix} response distribution')
    tb_logger.add_figure(f'kinematics/{prefix}/distribution', dist_fig, current_epoch)
    plt.close(dist_fig)

    tb_logger.add_scalar(f'kinematics/{prefix}/model_response', float(_q50(pred / truth)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/reco_response', float(_q50(reco_pred / truth)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/model_resolution', float(_iqr_over_q50(pred / truth)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/reco_resolution', float(_iqr_over_q50(reco_pred / truth)), current_epoch)


def _log_residual_family(tb_logger, current_epoch, prefix, pred_residual, reco_residual, truth_x, bins, x_label, residual_label, hist_bins):
    bias_fig = _overlay_curve_figure(
        truth_x, reco_residual, truth_x, pred_residual, bins, _q50,
        x_label, f'$q_{{50}}({residual_label})$', f'{prefix} bias vs truth'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/bias_vs_truth', bias_fig, current_epoch)
    plt.close(bias_fig)

    resolution_fig = _overlay_curve_figure(
        truth_x, reco_residual, truth_x, pred_residual, bins, _iqr,
        x_label, r'$q_{75} - q_{25}$', f'{prefix} resolution vs truth'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/resolution_vs_truth', resolution_fig, current_epoch)
    plt.close(resolution_fig)

    dist_fig = _distribution_figure(pred_residual, reco_residual, hist_bins, residual_label, f'{prefix} residual distribution')
    tb_logger.add_figure(f'kinematics/{prefix}/distribution', dist_fig, current_epoch)
    plt.close(dist_fig)

    tb_logger.add_scalar(f'kinematics/{prefix}/model_bias', float(_q50(pred_residual)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/reco_bias', float(_q50(reco_residual)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/model_resolution', float(_iqr(pred_residual)), current_epoch)
    tb_logger.add_scalar(f'kinematics/{prefix}/reco_resolution', float(_iqr(reco_residual)), current_epoch)




def _range_content_figure(model_values, reco_values, bin_edges, xlabel, data_label, xlim, summary_fn, summary_label):
    fig, rows = plt.subplots(nrows=3, ncols=4, sharex='col', figsize=(16, 9))
    axes = rows.flatten()
    bins = np.linspace(xlim[0], xlim[1], 101)
    for i, ax in enumerate(axes):
        if i >= len(bin_edges) - 1:
            ax.axis('off')
            continue
        left, right = bin_edges[i], bin_edges[i + 1]
        reco_data = _finite(reco_values[i]) if i < len(reco_values) else np.array([])
        model_data = _finite(model_values[i]) if i < len(model_values) else np.array([])
        if reco_data.size:
            ax.hist(reco_data, bins=bins, density=True, histtype='step', linewidth=1.8, color='tab:orange', label='Reco jet')
        if model_data.size:
            ax.hist(model_data, bins=bins, density=True, histtype='step', linewidth=1.8, linestyle='--', color='tab:red', label='Model')
        ax.set_title(f'{xlabel} in [{left:.3g}, {right:.3g}]', fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_xlabel(data_label)
        if model_data.size:
            summary = summary_fn(model_data)
            ax.text(0.03, 0.95, f'Model {summary_label} = {summary:.3f}', transform=ax.transAxes, va='top', fontsize=8)
        if reco_data.size:
            summary = summary_fn(reco_data)
            ax.text(0.03, 0.84, f'Reco {summary_label} = {summary:.3f}', transform=ax.transAxes, va='top', fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', frameon=False)
    fig.tight_layout()
    return fig


def _resolution_2d_figure(prediction, truth, x_bins, y_bins, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    mask = np.isfinite(prediction) & np.isfinite(truth)
    if mask.any():
        hist = ax.hist2d(truth[mask], prediction[mask], bins=[x_bins, y_bins], cmap='viridis')
        fig.colorbar(hist[3], ax=ax, label='Jets')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    return fig


def _log_ratio_diagnostics(tb_logger, current_epoch, prefix, pred, truth, reco_pred, truth_x, bins, x_label, ratio_label, hist_xlim=(0.5, 1.5)):
    binned_model = []
    binned_reco = []
    ratio_model = np.asarray(pred) / np.asarray(truth)
    ratio_reco = np.asarray(reco_pred) / np.asarray(truth)
    truth_x = np.asarray(truth_x)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (truth_x >= left) & (truth_x < right)
        binned_model.append(ratio_model[mask])
        binned_reco.append(ratio_reco[mask])
    content_fig = _range_content_figure(
        binned_model, binned_reco, bins, x_label, ratio_label, hist_xlim, _iqr_over_q50, 'IQR/q50'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/bin_distributions', content_fig, current_epoch)
    plt.close(content_fig)

    left = min(np.min(_finite(truth)), np.min(_finite(reco_pred)), np.min(_finite(pred)))
    right = max(np.max(_finite(truth)), np.max(_finite(reco_pred)), np.max(_finite(pred)))
    diag_bins = np.linspace(left, right, 80)
    model_2d = _resolution_2d_figure(pred, truth, diag_bins, diag_bins, x_label, f'Predicted {prefix}', f'{prefix} model 2D')
    tb_logger.add_figure(f'kinematics/{prefix}/model_resolution_2d', model_2d, current_epoch)
    plt.close(model_2d)
    reco_2d = _resolution_2d_figure(reco_pred, truth, diag_bins, diag_bins, x_label, f'Reco {prefix}', f'{prefix} reco 2D')
    tb_logger.add_figure(f'kinematics/{prefix}/reco_resolution_2d', reco_2d, current_epoch)
    plt.close(reco_2d)


def _log_residual_diagnostics(tb_logger, current_epoch, prefix, pred_residual, reco_residual, truth_x, bins, x_label, residual_label, hist_xlim):
    binned_model = []
    binned_reco = []
    truth_x = np.asarray(truth_x)
    pred_residual = np.asarray(pred_residual)
    reco_residual = np.asarray(reco_residual)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (truth_x >= left) & (truth_x < right)
        binned_model.append(pred_residual[mask])
        binned_reco.append(reco_residual[mask])
    content_fig = _range_content_figure(
        binned_model, binned_reco, bins, x_label, residual_label, hist_xlim, _iqr, 'IQR'
    )
    tb_logger.add_figure(f'kinematics/{prefix}/bin_distributions', content_fig, current_epoch)
    plt.close(content_fig)

    y_bins = np.linspace(hist_xlim[0], hist_xlim[1], 80)
    model_2d = _resolution_2d_figure(pred_residual, truth_x, bins, y_bins, x_label, residual_label, f'{prefix} model 2D')
    tb_logger.add_figure(f'kinematics/{prefix}/model_resolution_2d', model_2d, current_epoch)
    plt.close(model_2d)
    reco_2d = _resolution_2d_figure(reco_residual, truth_x, bins, y_bins, x_label, residual_label, f'{prefix} reco 2D')
    tb_logger.add_figure(f'kinematics/{prefix}/reco_resolution_2d', reco_2d, current_epoch)
    plt.close(reco_2d)

def log_all_kinematics_metrics(
    targets: np.array,
    predictions: np.array,
    reco_jet_p4s: np.array,
    gen_jet_tau_p4s,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    dataset='train',
):
    signal_mask = targets['is_tau'] == 1
    signal_predictions = predictions['kinematics'][signal_mask]
    signal_targets = targets['kinematics'][signal_mask]
    reco = reinitialize_p4(reco_jet_p4s)[signal_mask]
    gen_tau = reinitialize_p4(gen_jet_tau_p4s)[signal_mask]

    # Original theta/mass decode kept for reference:
    # pred_pt = np.exp(signal_predictions[:, 0]) * reco.pt
    # true_pt = np.exp(signal_targets[:, 0]) * reco.pt
    # pred_theta_rad = signal_predictions[:, 1] + reco.theta
    # true_theta_rad = signal_targets[:, 1] + reco.theta
    # pred_phi_rad = signal_predictions[:, 2] + reco.phi
    # true_phi_rad = signal_targets[:, 2] + reco.phi
    # pred_m = np.exp(signal_predictions[:, 3]) * reco.mass
    # true_m = np.exp(signal_targets[:, 3]) * reco.mass

    pred_pt = np.exp(signal_predictions[:, 0]) * np.asarray(reco.pt)
    true_pt = np.asarray(gen_tau.pt)
    reco_pt = np.asarray(reco.pt)

    pred_eta = signal_predictions[:, 1] + np.asarray(reco.eta)
    true_eta = np.asarray(gen_tau.eta)
    reco_eta = np.asarray(reco.eta)

    pred_delta_phi = np.arctan2(signal_predictions[:, 2], signal_predictions[:, 3])
    true_delta_phi = np.arctan2(signal_targets[:, 2], signal_targets[:, 3])
    pred_phi = _wrap_delta_phi(pred_delta_phi + np.asarray(reco.phi))
    true_phi = np.asarray(gen_tau.phi)
    reco_phi = np.asarray(reco.phi)

    pred_mass = np.asarray(reco.mass)
    reco_mass = np.asarray(reco.mass)

    pred_energy = np.sqrt((pred_pt * np.cosh(pred_eta)) ** 2 + pred_mass ** 2)
    true_energy = np.asarray(gen_tau.energy)
    reco_energy = np.asarray(reco.energy)

    pt_bins = np.asarray(cfg.metrics.kinematics.pt.bin_edges['all'])
    eta_bins = np.asarray(cfg.metrics.kinematics.eta.bin_edges['all'])
    phi_bins_deg = np.asarray(cfg.metrics.kinematics.phi.bin_edges['all'])

    _log_ratio_family(
        tb_logger, current_epoch, 'pt', pred_pt, true_pt, reco_pt, true_pt, pt_bins,
        r'True $p_T$ [GeV]', r'$p_T^{pred} / p_T^{gen}$'
    )
    _log_ratio_diagnostics(
        tb_logger, current_epoch, 'pt', pred_pt, true_pt, reco_pt, true_pt, pt_bins,
        r'True $p_T$ [GeV]', r'$p_T^{pred} / p_T^{gen}$', hist_xlim=(0.5, 1.5)
    )

    pred_eta_residual = pred_eta - true_eta
    reco_eta_residual = reco_eta - true_eta
    _log_residual_family(
        tb_logger, current_epoch, 'eta', pred_eta_residual, reco_eta_residual, true_eta, eta_bins,
        r'True $\eta$', r'$\eta^{pred} - \eta^{gen}$', np.linspace(-0.1, 0.1, 90)
    )
    _log_residual_diagnostics(
        tb_logger, current_epoch, 'eta', pred_eta_residual, reco_eta_residual, true_eta, eta_bins,
        r'True $\eta$', r'$\eta^{pred} - \eta^{gen}$', hist_xlim=(-0.1, 0.1)
    )

    pred_phi_residual_rad = _wrap_delta_phi(pred_phi - true_phi)
    reco_phi_residual_rad = _wrap_delta_phi(reco_phi - true_phi)
    pred_phi_residual_deg = np.rad2deg(pred_phi_residual_rad)
    reco_phi_residual_deg = np.rad2deg(reco_phi_residual_rad)
    true_phi_deg = np.rad2deg(true_phi)
    _log_residual_family(
        tb_logger, current_epoch, 'phi', pred_phi_residual_deg, reco_phi_residual_deg, true_phi_deg, phi_bins_deg,
        r'True $\phi$ [deg]', r'$\phi^{pred} - \phi^{gen}$ [deg]', np.linspace(-10.0, 10.0, 90)
    )
    _log_residual_diagnostics(
        tb_logger, current_epoch, 'phi', pred_phi_residual_deg, reco_phi_residual_deg, true_phi_deg, phi_bins_deg,
        r'True $\phi$ [deg]', r'$\phi^{pred} - \phi^{gen}$ [deg]', hist_xlim=(-10.0, 10.0)
    )

    energy_bins = np.asarray(cfg.metrics.kinematics.energy.bin_edges['all'])
    _log_ratio_family(
        tb_logger, current_epoch, 'energy', pred_energy, true_energy, reco_energy, true_energy, energy_bins,
        r'True $E$ [GeV]', r'$E^{pred} / E^{gen}$'
    )

    delta_r_model = np.sqrt((pred_eta - true_eta) ** 2 + pred_phi_residual_rad ** 2)
    delta_r_reco = np.sqrt((reco_eta - true_eta) ** 2 + reco_phi_residual_rad ** 2)
    dr_fig = _overlay_curve_figure(
        true_pt, delta_r_reco, true_pt, delta_r_model, pt_bins, _q50,
        r'True $p_T$ [GeV]', r'Median $\Delta R$', r'$\Delta R$ vs true $p_T$'
    )
    tb_logger.add_figure('kinematics/deltaR/median_vs_truth_pt', dr_fig, current_epoch)
    plt.close(dr_fig)

    dr_dist = _distribution_figure(delta_r_model, delta_r_reco, np.linspace(0.0, 0.25, 100), r'$\Delta R$', r'$\Delta R$ distribution')
    tb_logger.add_figure('kinematics/deltaR/distribution', dr_dist, current_epoch)
    plt.close(dr_dist)

    tb_logger.add_scalar('kinematics/deltaR/model_median', float(_q50(delta_r_model)), current_epoch)
    tb_logger.add_scalar('kinematics/deltaR/reco_median', float(_q50(delta_r_reco)), current_epoch)
