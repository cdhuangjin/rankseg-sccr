import itertools
import json
from pathlib import Path
import numpy as np
from sccr import objective, coordinate_all, coordinate_pruned

rng = np.random.default_rng(20261023)
for case in range(40):
    p = rng.dirichlet(np.ones(3), size=6).T.reshape(3, 2, 3)
    allowed = rng.random(3) > .5
    allowed[0] = True
    y = rng.choice(np.flatnonzero(allowed), size=(2, 3)).astype(np.uint8)
    overlap = rng.random((2, 3)) > .3
    out, stats = coordinate_pruned(p, y, overlap, allowed, audit=True)
    assert np.array_equal(out[~overlap], y[~overlap])
    assert allowed[out].all()
    optimum = -1
    for assignment in itertools.product(np.flatnonzero(allowed), repeat=int(overlap.sum())):
        z = y.copy()
        z[overlap] = assignment
        optimum = max(optimum, objective(p, z))
    assert stats['objective_before'] <= stats['objective_after'] + 1e-12 <= optimum + 1e-10
    all_allowed, _ = coordinate_pruned(p, y, overlap, np.ones(3, bool))
    original, _ = coordinate_all(p, y, overlap)
    assert np.array_equal(all_allowed, original)
    empty, _ = coordinate_pruned(p, y, np.zeros_like(overlap), np.zeros(3, bool))
    assert np.array_equal(empty, y)
try:
    coordinate_pruned(p, np.ones((2, 3), np.uint8), np.ones((2, 3), bool), np.array([True, False, False]))
except AssertionError:
    pass
else:
    raise AssertionError('Invalid starting support was silently accepted')
result = {'status': 'PASS', 'exhaustive_cases': 40, 'full_support_matches_old_solver': True,
          'per_move_gain_independent_check': True, 'invalid_initial_support_rejected': True}

print('PRUNING_MATH_CHECKS_PASS', json.dumps(result))
