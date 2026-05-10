#!/usr/bin/env python3
"""
SCRIPT_TimeTrackerRequest.py

Quick diagnostic: how long does a single `tracker.get_position(...)` call
actually take? Helps disambiguate the source of the low fresh-frame rate
seen in SCRIPT_MeasureTrackerNoise (server's own logs report 7-8 Hz, but
client sees ~0.8 Hz of distinct reads).

If the per-call time is small (≪100 ms), the slowness is server-side or
in the GETALL handler. If it's large (≫100 ms), it's network / TCP setup
overhead in `ServerClient._request` (which opens a new socket per call).

Output: per-call min/median/p95/max + total wall time.
"""
import time
import numpy as np

from Library import LorexTracker


ROBOT_ID = 1
N_CALLS  = 100


def main():
    tracker = LorexTracker.LorexTracker()
    print(f"Calling tracker.get_position({ROBOT_ID}) {N_CALLS}× back-to-back...")
    times = []
    t_start = time.time()
    for _ in range(N_CALLS):
        t0 = time.time()
        tracker.get_position(ROBOT_ID)
        times.append(time.time() - t0)
    total = time.time() - t_start
    a = np.array(times) * 1000.0   # ms
    print(f"\nWall time: {total:.1f} s for {N_CALLS} calls "
          f"(effective rate {N_CALLS/total:.1f} Hz)")
    print(f"Per-call latency:")
    print(f"  min    = {a.min():6.1f} ms")
    print(f"  median = {np.median(a):6.1f} ms")
    print(f"  p95    = {np.percentile(a,95):6.1f} ms")
    print(f"  max    = {a.max():6.1f} ms")
    print(f"  mean   = {a.mean():6.1f} ms")


if __name__ == "__main__":
    main()
