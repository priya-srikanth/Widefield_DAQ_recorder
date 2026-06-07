# Motion-correction sign bug (wfield 0.4.2) — record + remediation

## The bug
`wfield.registration.registration_upsample` (the 2D / `--mode 2d` path) applied the
phase-correlation offset with the **wrong sign**:

```python
(xs, ys), _ = cv2.phaseCorrelate(template, dst)
M = np.float32([[1, 0, xs], [0, 1, ys]])   # WRONG: +; shifts dst AWAY from template
```

`phaseCorrelate` returns how far `dst` is displaced from the template, so aligning
`dst` to the template requires `-(xs, ys)`. The `+` **doubles** the drift instead of
removing it.

Verified on PS93 2026-06-06 (reference vs drifted late frames): raw vessel NCC
**0.738**; with `+` (upstream) **0.393** (worse); with `-` (fixed) **0.903**.

## Impact = magnitude of each session's drift
The error added ≈ 2× the session's drift. It is **invisible** on low-drift sessions
and only material when a session physically drifted. Per-session median frame shift:

| session | median shift | impact |
|---|---|---|
| **PS93 2026-06-06** | **8.66 px** (55% of frames >5 px) | **SEVERE — re-processed with fix** |
| PS94 2026-06-05 | 1.54 px | minor — re-processed with fix |
| all other sessions (6/1–6/6) | < 1.1 px (mostly < 0.7) | negligible (sub-pixel; not re-processed) |

(Large `max_shift` values on a few sessions — e.g. 91 px, 339 px — are isolated
blown-up frames excluded from imagery; `%>5px = 0`, median sub-pixel, so no effect.)

## Fix
`wfield_local/motion_correct_fixed.py` — a sign-corrected, drop-in `motion_correct`
(vendored because `wfield.utils.runpar` uses `multiprocessing`, so a monkeypatch
would not reach the worker processes). `wfield_local/run_wfield_motion.py` imports
`motion_correct` from there. The installed wfield package is left untouched.

## Sessions that used the OLD (buggy) motion correction
**Every session processed before this fix** (committed on 2026-06-07) used the buggy
`+`-sign correction. Status:

- **Re-processed with the fix** (motion → SVD → Allen → maps; outputs refreshed on
  N:/M:): **PS93 2026-06-06**, **PS94 2026-06-05**.
- **NOT re-processed** (buggy but negligible, median drift < 1.1 px): all 6/1, 6/2,
  6/3, 6/4 sessions; PS92/PS94/PS95 6/5; PS92/PS94/PS95 6/6. Their corrected movies
  carry a sub-pixel residual error; re-process only if a future check shows
  meaningful drift.

Upstream: worth reporting to jcouto/wfield.
