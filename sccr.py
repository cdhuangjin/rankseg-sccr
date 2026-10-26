"""Frozen one-scan algorithms extracted from the confirmed experiment."""
import numpy as np

def objective(prob, prediction):
    p = np.asarray(prob, dtype=np.float64).reshape(prob.shape[0], -1)
    y = prediction.ravel()
    n = np.bincount(y, minlength=p.shape[0]).astype(float)
    s = np.bincount(y, weights=p[y, np.arange(y.size)], minlength=p.shape[0])
    return float(np.sum(2 * s / (n + p.sum(1) + 1)))

def coordinate(prob, prediction, overlap, tolerance=1e-12, audit=False):
    """One exact-gain coordinate sweep. Auxiliary state is updated after each move."""
    p = np.asarray(prob, dtype=np.float64).reshape(prob.shape[0], -1)
    shape = prediction.shape
    y = prediction.copy().ravel()
    means = p.sum(1)
    n = np.bincount(y, minlength=p.shape[0]).astype(float)
    s = np.bincount(y, weights=p[y, np.arange(y.size)], minlength=p.shape[0])
    before = float(np.sum(2 * s / (n + means + 1)))
    moves = 0
    for j in np.flatnonzero(overlap):
        a = int(y[j])
        old = 2 * s / (n + means + 1)
        gains = 2 * (s + p[:, j]) / (n + means + 2) - old
        removal = 2 * (s[a] - p[a, j]) / (n[a] + means[a]) - old[a]
        gains += removal
        gains[a] = 0.0
        b = int(gains.argmax())
        if audit:
            # Independent full-objective evaluation of every one-pixel alternative.
            actual = []
            initial = objective(p, y)
            for c in range(p.shape[0]):
                alternative = y.copy()
                alternative[j] = c
                actual.append(objective(p, alternative) - initial)
            np.testing.assert_allclose(gains, actual, atol=1e-12, rtol=1e-10)
        if gains[b] > tolerance:
            y[j] = b
            n[a] -= 1
            n[b] += 1
            s[a] -= p[a, j]
            s[b] += p[b, j]
            moves += 1
    after = objective(p, y)
    assert after >= before - 1e-10
    return y.reshape(shape), {"objective_before": before, "objective_after": after, "moves": moves}

coordinate_all = coordinate

def coordinate_pruned(prob, prediction, overlap, allowed, tolerance=1e-12, audit=False):
    p = np.asarray(prob, dtype=np.float64).reshape(prob.shape[0], -1)
    y = prediction.copy().ravel()
    allowed = np.asarray(allowed, dtype=bool)
    assert allowed.shape == (p.shape[0],)
    assert allowed[y[overlap.ravel()]].all(), "Initial overlap assignment outside support; stop, do not project"
    means = p.sum(1)
    n = np.bincount(y, minlength=p.shape[0]).astype(float)
    s = np.bincount(y, weights=p[y, np.arange(y.size)], minlength=p.shape[0])
    before = float(np.sum(2 * s / (n + means + 1)))
    moves = 0
    for j in np.flatnonzero(overlap):
        a = int(y[j])
        old = 2 * s / (n + means + 1)
        gains = 2 * (s + p[:, j]) / (n + means + 2) - old
        removal = 2 * (s[a] - p[a, j]) / (n[a] + means[a]) - old[a]
        gains += removal
        gains[a] = 0.0
        if audit:
            current = objective(p, y)
            for c in np.flatnonzero(allowed):
                alternative = y.copy()
                alternative[j] = c
                assert np.isclose(gains[c], objective(p, alternative)-current, atol=1e-12, rtol=1e-10)
        gains[~allowed] = -np.inf
        b = int(gains.argmax())
        if gains[b] > tolerance:
            y[j] = b
            n[a] -= 1
            n[b] += 1
            s[a] -= p[a, j]
            s[b] += p[b, j]
            moves += 1
    after = objective(p, y)
    assert after >= before - 1e-10
    assert allowed[y[overlap.ravel()]].all()
    return y.reshape(prediction.shape), {"objective_before": before, "objective_after": after, "moves": moves}
