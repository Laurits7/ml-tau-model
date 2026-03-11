import numpy as np
import boost_histogram as bh


def calculate_bin_centers(edges: np.array) -> np.array:
    bin_widths = np.array([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
    bin_centers = []
    for i in range(len(edges) - 1):
        bin_centers.append(edges[i] + (bin_widths[i] / 2))
    return np.array(bin_centers), bin_widths / 2


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1
