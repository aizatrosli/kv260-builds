"""Calibration observers: min/max, percentile, MSE-optimal clipping.

Observers are two-pass and streaming: pass 1 (``observe_range``) tracks min/max, pass 2
(``observe_hist``) fills a fixed histogram over [min, max]. ``range(symmetric)`` returns the float
clipping range (lo, hi); qparams are then derived with :func:`tpuquant.qscheme.act_qparams`.
"""
import numpy as np

NBINS = 4096


class Observer:
    needs_hist = False

    def __init__(self, nbins=NBINS):
        self.nbins = nbins
        self.min = np.inf
        self.max = -np.inf
        self.hist = None
        self.count = 0

    def observe_range(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size:
            self.min = min(self.min, float(x.min()))
            self.max = max(self.max, float(x.max()))
            self.count += x.size

    def observe_hist(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        lo, hi = self.min, self.max
        if hi <= lo:
            hi = lo + 1e-12
        h, _ = np.histogram(x, bins=self.nbins, range=(lo, hi))
        self.hist = h.astype(np.float64) if self.hist is None else self.hist + h

    # -- helpers ------------------------------------------------------------------------------
    def _edges(self):
        hi = self.max if self.max > self.min else self.min + 1e-12
        return np.linspace(self.min, hi, self.nbins + 1)

    def _abs_hist(self):
        """Histogram of |x| on [0, amax] with nbins bins (built from the signed histogram)."""
        e = self._edges(); c = 0.5 * (e[1:] + e[:-1])
        amax = max(abs(self.min), abs(self.max), 1e-12)
        h, ae = np.histogram(np.abs(c), bins=self.nbins, range=(0, amax), weights=self.hist)
        return h, ae

    def range(self, symmetric):
        raise NotImplementedError

    def minmax(self, symmetric):
        if symmetric:
            a = max(abs(self.min), abs(self.max))
            return -a, a
        return min(self.min, 0.0), max(self.max, 0.0)


class MinMaxObserver(Observer):
    name = "minmax"

    def range(self, symmetric):
        return self.minmax(symmetric)


def _hist_quantile(h, edges, q):
    """Linear-interpolated quantile (q in [0,1]) of a histogram."""
    cdf = np.cumsum(h)
    if cdf[-1] <= 0:
        return edges[-1]
    t = q * cdf[-1]
    i = int(np.searchsorted(cdf, t))
    i = min(i, len(h) - 1)
    prev = cdf[i - 1] if i > 0 else 0.0
    frac = (t - prev) / h[i] if h[i] > 0 else 1.0
    return edges[i] + frac * (edges[i + 1] - edges[i])


class PercentileObserver(Observer):
    """Clip at the p-th percentile of |x| (symmetric) or at (100-p, p) percentiles (asymmetric).
    Small tensors (< min_count values) fall back to min/max, as jepa-tpu's quantize.amax does."""
    needs_hist = True
    name = "percentile"

    def __init__(self, pct=99.99, min_count=1000, nbins=NBINS):
        super().__init__(nbins)
        self.pct = pct
        self.min_count = min_count

    def range(self, symmetric):
        if self.count <= self.min_count or self.hist is None:
            return self.minmax(symmetric)
        q = self.pct / 100.0
        if symmetric:
            h, ae = self._abs_hist()
            a = _hist_quantile(h, ae, q)
            return -a, a
        e = self._edges()
        lo = _hist_quantile(self.hist, e, 1 - q); hi = _hist_quantile(self.hist, e, q)
        return min(lo, 0.0), max(hi, 0.0)


def _mse_sym(h, centers, cands, qmax=127):
    """Expected squared error for each symmetric clip candidate (rounding noise d^2/12 inside)."""
    err = []
    for c in cands:
        d = c / qmax
        over = np.maximum(centers - c, 0.0)
        inside = centers <= c
        err.append(float((h * (over ** 2)).sum() + h[inside].sum() * d * d / 12.0))
    return np.array(err)


class MSEObserver(Observer):
    """Choose the clipping range minimising expected quantisation MSE over the histogram."""
    needs_hist = True
    name = "mse"

    def __init__(self, ncand=200, nbins=NBINS):
        super().__init__(nbins)
        self.ncand = ncand

    def range(self, symmetric):
        if self.hist is None:
            return self.minmax(symmetric)
        if symmetric:
            h, ae = self._abs_hist(); c = 0.5 * (ae[1:] + ae[:-1])
            amax = ae[-1]
            cands = amax * np.linspace(0.05, 1.0, self.ncand)
            a = cands[int(np.argmin(_mse_sym(h, c, cands)))]
            return -a, a
        e = self._edges(); c = 0.5 * (e[1:] + e[:-1]); h = self.hist
        lo0, hi0 = min(self.min, 0.0), max(self.max, 0.0)
        fr = np.linspace(0.1, 1.0, 46)
        best, arg = np.inf, (lo0, hi0)
        for a in fr:
            for b in fr:
                lo, hi = lo0 * a, hi0 * b
                if hi - lo <= 0:
                    continue
                d = (hi - lo) / 255.0
                over = np.maximum(c - hi, 0) + np.maximum(lo - c, 0)
                inside = (c >= lo) & (c <= hi)
                err = (h * over ** 2).sum() + h[inside].sum() * d * d / 12
                if err < best:
                    best, arg = err, (lo, hi)
        return arg


def make_observer(method, **kw):
    """method: 'minmax' | 'percentile' | 'percentile:99.9' | 'mse'."""
    if method == "minmax":
        return MinMaxObserver()
    if method.startswith("percentile"):
        pct = float(method.split(":")[1]) if ":" in method else kw.get("pct", 99.99)
        return PercentileObserver(pct=pct)
    if method == "mse":
        return MSEObserver()
    raise ValueError(f"unknown observer {method!r}")
