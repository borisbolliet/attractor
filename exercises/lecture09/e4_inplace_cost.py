import time
import numpy as np
N, REPS = 10**7, 20
k = np.full(N, 0.05)
y = np.ones(N)
t0 = time.perf_counter()
for _ in range(REPS):
    y = y + k
t_rebind = time.perf_counter() - t0
y = np.ones(N)
t0 = time.perf_counter()
for _ in range(REPS):
    y += k
t_inplace = time.perf_counter() - t0
print(f"rebind   {t_rebind:.3f} s")
print(f"in place {t_inplace:.3f} s")
