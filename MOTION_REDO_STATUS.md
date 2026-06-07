# Motion-correction redo status (sign-fix remediation)

Re-processing every session with the sign-fixed motion correction
(`wfield_local/motion_correct_fixed.py`), newest → oldest. Per session:
fixed motion → SVD → Allen → corrected `.bin` → **M: standby**, non-bin outputs → **N:**.

Legend: **OLD** = still buggy `+`-sign correction; **UPDATED** = re-done with fix.
Standby = updated `.bin` byte-copied to `M:\Widefield\labcams\<date>\<session>\motion_corrected\`.
Source: **E** = re-motion existing E: cleanpairs (fast); **M** = rebuild from M: raw (slow).

| date | session | drift (px) | motion-corr | bin→standby | source | notes |
|---|---|---|---|---|---|---|
| 2026-06-06 | PS93_20260606_180117 | 8.66 | **UPDATING** | pending | E | severe; +maps+deck |
| 2026-06-06 | PS92_20260606_122451 | 0.62 | OLD | OLD bin on M | E | negligible |
| 2026-06-06 | PS94_20260606_140854 | 0.75 | OLD | OLD bin on M | E | negligible |
| 2026-06-06 | PS95_20260606_160806 | 0.90 | OLD | OLD bin on M | E | negligible |
| 2026-06-05 | PS94_20260605_142009 | 1.54 | **QUEUED** | pending | M | minor; +maps+deck+xday-ref |
| 2026-06-05 | PS92_20260605_125023 | 0.71 | OLD | OLD bin on M | M | negligible |
| 2026-06-05 | PS93_20260605_174659 | 0.46 | OLD | OLD bin on M | M | negligible |
| 2026-06-05 | PS95_20260605_163102 | 0.90 | OLD | OLD bin on M | M | negligible |
| 2026-06-04 | PS92_20260604_132934 | 0.49 | OLD | OLD bin on M | M | concat session |
| 2026-06-04 | PS94_20260604_151516 | — | OLD | OLD bin on M | M | negligible |
| 2026-06-04 | PS95_20260604_165712 | — | OLD | OLD bin on M | M | negligible |
| 2026-06-03 | PS92_20260603_104008 | 0.62 | OLD | OLD bin on M | M | negligible |
| 2026-06-03 | PS94_20260603 | 0.34 | OLD | OLD bin on M | M | negligible |
| 2026-06-03 | PS95_20260603_194442 | 0.40 | OLD | OLD bin on M | M | negligible |
| 2026-06-02 | PS92_20260602_151820 | 0.38 | OLD | OLD bin on M | M | negligible |
| 2026-06-01 | PS94_20260601_141614 | 0.19 | OLD | OLD bin on M | M | regime-A full-FOV; no relabel; landmarks/DAQ on E |
| 2026-06-01 | PS95_20260601_153653 | 0.33 | OLD | OLD bin on M | M | regime-A full-FOV; no relabel; landmarks/DAQ on E |

Updated by `_redo_motion_all.py` as each session completes (PS93 6/6 + PS94 6/5
handled by their dedicated drivers since they also need maps/deck/cross-day refresh).
