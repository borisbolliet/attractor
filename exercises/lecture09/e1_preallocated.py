import numpy as np
state = np.array([1.0, 2.0])
out = np.empty((3, 2))
kept = []
for i in range(3):
    state *= 2
    out[i] = state
    kept.append(state)
print(out.tolist())
print([s.tolist() for s in kept], len({id(s) for s in kept}))
state[0] = 0.0
print(out[2].tolist(), kept[0].tolist())
state, kept = np.array([1.0, 2.0]), []
for i in range(3):
    state *= 2
    kept.append(state.copy())
print([s.tolist() for s in kept])
