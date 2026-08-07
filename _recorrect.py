"""Recompute hemodynamic correction with the CORRECT functional channel.

ONE-OFF, local to this PS92 rescue recording only. Evidence (brightness, GCaMP
transient amplitude, structured cue map, user's visual read) shows the DAQ
led415/led470 TTLs were swapped for THIS recording, so the .dat channel order is
[real-470 functional, real-415 isosbestic] = [ch0, ch1]. Codex ran the
correction with functional_channel=1 -> garbage. Correct is functional_channel=0.
The SVD (U, SVT) is channel-agnostic and reused as-is.
"""
import numpy as np, h5py, os, json, shutil
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wfield.hemocorrection import hemodynamic_correction

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
res = os.path.join(base, "motion_corrected", "wfield_local_results")
daq_path = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
fs = 31.23


def main():
    U = np.asarray(np.load(os.path.join(res, "U.npy")), dtype=np.float32)
    SVT = np.asarray(np.load(os.path.join(res, "SVT.npy")), dtype=np.float32)
    H, W, K = U.shape
    U2 = U.reshape(-1, K)

    svt_func = SVT[:, 0::2]   # labeled "415" = real 470 functional
    svt_iso = SVT[:, 1::2]    # labeled "470" = real 415 isosbestic
    SVTcorr_fix, rcoeffs_fix, T_fix = hemodynamic_correction(U, svt_func, svt_iso, fs=fs)
    SVTcorr_old = np.load(os.path.join(res, "SVTcorr.npy"))
    print("SVTcorr_fix", SVTcorr_fix.shape, "old", SVTcorr_old.shape, flush=True)

    fm = np.load(os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz"))
    c0 = fm["original_frame_index_ch0"]
    with h5py.File(daq_path, "r") as f:
        fsd = float(f.attrs["sample_rate_hz"])
        di = [s.decode() for s in f["digital/channel_names"][:]]
        bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")

    def rises(s, t=0.5):
        b = (s > t).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
    pco = rises(bits[:, di.index("pco_exposure")].astype(float))
    cue = rises(bits[:, di.index("cue")].astype(float))
    csample = pco[c0 + 1]
    ins = np.clip(np.searchsorted(csample, cue), 1, len(csample) - 1)
    cue_idx = np.where(np.abs(cue - csample[ins - 1]) <= np.abs(csample[ins] - cue), ins - 1, ins)
    npairs = SVTcorr_fix.shape[1]
    n = int(round(fs))

    def cue_map(SVTc):
        pre = post = 0; used = 0
        for ci in cue_idx:
            if ci - n < 0 or ci + 1 + n > npairs:
                continue
            if (csample[ci + n - 1] - csample[ci - n]) / fsd > 4.0:
                continue
            pre = pre + SVTc[:, ci - n:ci].mean(1); post = post + SVTc[:, ci + 1:ci + 1 + n].mean(1)
            used += 1
        return (U2 @ ((post - pre) / used)).reshape(H, W), used

    map_fix, used = cue_map(SVTcorr_fix)
    map_old, _ = cue_map(SVTcorr_old)
    print(f"cue post-pre: CORRECTED(func=0) [{map_fix.min():.4f},{map_fix.max():.4f}]  "
          f"OLD(func=1) [{map_old.min():.4f},{map_old.max():.4f}]  n={used}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    m = np.percentile(np.abs(map_fix), 99)
    ax[0].imshow(map_fix, cmap="seismic", vmin=-m, vmax=m); ax[0].set_title("CORRECTED (func=ch0) cue post-pre"); ax[0].axis("off")
    m2 = np.percentile(np.abs(map_old), 99)
    ax[1].imshow(map_old, cmap="seismic", vmin=-m2, vmax=m2); ax[1].set_title("OLD (func=ch1, wrong) cue post-pre"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(base, "_recorrect_check.png"), dpi=120)
    print("saved _recorrect_check.png", flush=True)

    for name, arr in [("SVTcorr.npy", SVTcorr_fix), ("rcoeffs.npy", rcoeffs_fix), ("T.npy", T_fix)]:
        src = os.path.join(res, name)
        bak = os.path.join(res, name.replace(".npy", "_functional1_WRONG.npy"))
        if os.path.exists(src) and not os.path.exists(bak):
            shutil.copy2(src, bak)
        np.save(src, arr)
    print("saved corrected SVTcorr/rcoeffs/T (backed up *_functional1_WRONG.npy)", flush=True)

    sp = os.path.join(res, "local_wfield_summary.json")
    s = json.loads(open(sp).read())
    s["functional_channel"] = 0
    s["note"] = ("ONE-OFF: DAQ led415/led470 TTLs swapped for this recording; .dat ch0 is real 470 functional. "
                 "SVTcorr/rcoeffs/T recomputed with functional_channel=0. "
                 "Old (functional_channel=1) outputs saved as *_functional1_WRONG.npy.")
    open(sp, "w").write(json.dumps(s, indent=2))
    print("updated local_wfield_summary.json", flush=True)


if __name__ == "__main__":
    main()
