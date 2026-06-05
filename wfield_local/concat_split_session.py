"""Concatenate a force-split widefield session into one continuous session.

When labcams + the DAQ recorder are force-closed mid-session (e.g. an OS/updater
kill) and restarted, you get N camera ``.dat`` segments and N DAQ ``.h5`` segments
that together cover one recording with short gaps where nothing was acquired. This
tool rejoins them for analysis:

  CAMERA  -> byte-concatenate the ``.dat`` segments back-to-back. The camera was
            OFF during each gap, so there are simply no gap frames; pixel geometry
            (chan,H,W) must match across segments. Output is one ``.dat`` the
            standard pipeline (``mmap_dat`` / ``run_wfield_local``) reads as usual.

  DAQ     -> concatenate the per-sample streams (analog int16 + bit-packed digital)
            with each inter-segment gap ZERO-PADDED, so the sample timeline stays
            wall-clock accurate. This is deliberate: the uninterrupted behavior
            program + behavior camera kept running through the gap, so preserving
            the gap duration is what lets you later align them to the widefield DAQ
            via the shared sync pulse. Gaps carry no pco_exposure pulses (camera was
            off), so the cumulative pco-pulse sequence still maps 1:1 onto the
            cumulative camera frames -- event->frame mapping is unaffected.

Per segment it VERIFIES pco_exposure rising edges == 2 * dat_frame_pairs (the
invariant that the camera and DAQ agree on frame count), and writes a manifest JSON
recording per-segment sample offsets, gap sizes, and the frame/sample boundary
indices for downstream behavior alignment.

Usage:
  python -m wfield_local.concat_split_session \
    --segment "E:\\...\\PS92_20260604_133714.h5::E:\\...\\raw_widefield_data\\pco_edge_run000_00000000_2_460_480_uint16.dat" \
    --segment "E:\\...\\PS92_20260604_140742.h5::E:\\...\\raw_widefield_data_2\\pco_edge_run000_00000000_2_460_480_uint16.dat" \
    --label PS92_20260604_concat \
    --out-cam-dir "E:\\labcams_data\\20260604\\PS92_20260604_132934\\raw_widefield_data_concat" \
    --out-daq    "E:\\DAQ_recorder_output\\20250604\\PS92_20260604_concat.h5"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import h5py
import numpy as np

DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_uint16")
PCO_BIT = 7  # digital channel order: sync,cue,trial_start,spout_strobe,bit0,bit1,bit2,pco_exposure
COPY_BUF = 512 * 1024 * 1024  # 512 MB binary copy buffer


def _dat_dims(path: Path):
    m = DAT_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse dims from {path.name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))  # (chan, H, W)


def _dat_pairs(path: Path, dims) -> int:
    ch, h, w = dims
    bytes_per = ch * h * w * 2
    sz = path.stat().st_size
    if sz % bytes_per:
        raise ValueError(f"{path.name}: size {sz} not a whole number of frames ({bytes_per} B each)")
    return sz // bytes_per


def _parse_ts(s):
    if isinstance(s, bytes):
        s = s.decode()
    return dt.datetime.fromisoformat(s)


def _attr(h, key, default=None):
    v = h.attrs.get(key, default)
    if isinstance(v, bytes):
        v = v.decode()
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Concatenate a force-split widefield session.")
    ap.add_argument("--segment", action="append", required=True,
                    help='"<daq_h5>::<camera_dat>", repeat in chronological order (>=2)')
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-cam-dir", type=Path, required=True)
    ap.add_argument("--out-daq", type=Path, required=True)
    ap.add_argument("--fs", type=float, default=None, help="sample rate override (else read from seg0)")
    ap.add_argument("--dry-run", action="store_true", help="validate + report plan, write nothing")
    args = ap.parse_args()

    segs = []
    for s in args.segment:
        if "::" not in s:
            raise ValueError(f"--segment must be '<h5>::<dat>': {s}")
        h5s, dats = s.split("::", 1)
        segs.append((Path(h5s), Path(dats)))
    if len(segs) < 2:
        raise ValueError("need >= 2 segments")

    # ---- inspect + validate each segment ----
    dims0 = _dat_dims(segs[0][1])
    fs = args.fs
    meta = []
    for i, (h5p, datp) in enumerate(segs):
        if not h5p.exists():
            raise FileNotFoundError(h5p)
        if not datp.exists():
            raise FileNotFoundError(datp)
        dims = _dat_dims(datp)
        if dims != dims0:
            raise ValueError(f"segment {i} dims {dims} != segment0 {dims0}; cannot concatenate")
        pairs = _dat_pairs(datp, dims)
        with h5py.File(h5p, "r") as h:
            nsamp = int(h["analog/samples_int16"].shape[0])
            seg_fs = float(_attr(h, "sample_rate_hz", 5000.0))
            created = _parse_ts(_attr(h, "created_at"))
            closed = _attr(h, "closed_at")
            closed = _parse_ts(closed) if closed else None
            rec_complete = bool(_attr(h, "recording_complete", False))
            dpk = h["digital/packed_samples"][:, 0]
            pco = ((dpk >> PCO_BIT) & 1).astype(np.int8)
            rises = int(np.sum((pco[1:] == 1) & (pco[:-1] == 0)))
        if fs is None:
            fs = seg_fs
        ok = (rises == 2 * pairs)
        meta.append(dict(h5=str(h5p), dat=str(datp), dims=dims, pairs=pairs, nsamp=nsamp,
                         fs=seg_fs, created=created, closed=closed, rec_complete=rec_complete,
                         pco_rises=rises, pulses_match=ok))
        flag = "OK" if ok else "!! MISMATCH"
        print(f"[seg{i}] pairs={pairs} samples={nsamp} ({nsamp/seg_fs:.1f}s) "
              f"pco_rises={rises} (=2x{pairs}? {flag}) created={created} complete={rec_complete}", flush=True)
        if not ok:
            raise ValueError(f"segment {i}: pco rises {rises} != 2*pairs {2*pairs}; refusing to concatenate")

    # ---- gaps (wall-clock) between consecutive segments ----
    gap_samps = [0]
    for i in range(1, len(meta)):
        prev = meta[i - 1]
        prev_end = prev["created"] + dt.timedelta(seconds=prev["nsamp"] / prev["fs"])
        gap_s = (meta[i]["created"] - prev_end).total_seconds()
        if gap_s < 0:
            print(f"[gap{i}] WARNING negative gap {gap_s:.3f}s -> clamping to 0", flush=True)
            gap_s = 0.0
        gsamp = int(round(gap_s * fs))
        gap_samps.append(gsamp)
        print(f"[gap{i}] {gap_s:.3f}s -> {gsamp} zero-padded samples", flush=True)

    total_pairs = sum(m["pairs"] for m in meta)
    total_samps = sum(m["nsamp"] for m in meta) + sum(gap_samps)
    print(f"\n[plan] camera: {total_pairs} frame-pairs ({total_pairs*dims0[0]} single frames)")
    print(f"[plan] DAQ: {total_samps} samples ({total_samps/fs:.1f}s incl. {sum(gap_samps)} gap samples)")
    print(f"[plan] out .dat: {args.out_cam_dir}")
    print(f"[plan] out .h5 : {args.out_daq}")

    # boundary indices (for behavior alignment): cumulative sample + frame at each seg start
    boundaries = []
    s_off = 0
    f_off = 0
    for i, m in enumerate(meta):
        s_off += gap_samps[i]
        boundaries.append(dict(segment=i, sample_start=s_off, frame_pair_start=f_off,
                               single_frame_start=f_off * dims0[0],
                               created_at=m["created"].isoformat(), n_samples=m["nsamp"],
                               n_frame_pairs=m["pairs"], gap_samples_before=gap_samps[i]))
        s_off += m["nsamp"]
        f_off += m["pairs"]

    if args.dry_run:
        print("\n[dry-run] no files written.")
        print(json.dumps(boundaries, indent=2))
        return 0

    args.out_cam_dir.mkdir(parents=True, exist_ok=True)
    args.out_daq.parent.mkdir(parents=True, exist_ok=True)
    ch, H, W = dims0
    out_dat = args.out_cam_dir / f"pco_edge_run000_00000000_{ch}_{H}_{W}_uint16.dat"

    # ---- camera: binary concatenate ----
    print(f"\n[camera] writing {out_dat} ...", flush=True)
    written = 0
    with open(out_dat, "wb") as fout:
        for i, m in enumerate(meta):
            with open(m["dat"], "rb") as fin:
                while True:
                    buf = fin.read(COPY_BUF)
                    if not buf:
                        break
                    fout.write(buf)
                    written += len(buf)
            print(f"   seg{i} appended ({written/1e9:.1f} GB total)", flush=True)
    got_pairs = _dat_pairs(out_dat, dims0)
    assert got_pairs == total_pairs, f"output .dat has {got_pairs} pairs, expected {total_pairs}"
    print(f"[camera] done: {got_pairs} frame-pairs", flush=True)

    # ---- DAQ: concatenate with zero-padded gaps ----
    print(f"[daq] writing {args.out_daq} ...", flush=True)
    with h5py.File(meta[0]["h5"], "r") as h0:
        n_an = h0["analog/samples_int16"].shape[1]
        with h5py.File(args.out_daq, "w") as out:
            an = out.create_dataset("analog/samples_int16", shape=(total_samps, n_an),
                                    dtype="int16", fillvalue=0,
                                    chunks=(min(1_000_000, total_samps), n_an), compression="gzip", compression_opts=1)
            dg = out.create_dataset("digital/packed_samples", shape=(total_samps, 1),
                                    dtype="uint8", fillvalue=0,
                                    chunks=(min(1_000_000, total_samps), 1), compression="gzip", compression_opts=1)
            # copy metadata sub-datasets verbatim from seg0
            for grp in ("analog", "digital"):
                for name in h0[grp]:
                    src = h0[f"{grp}/{name}"]
                    if name in ("samples_int16", "packed_samples"):
                        continue
                    out.create_dataset(f"{grp}/{name}", data=src[...])
            # root attrs from seg0, then override the session-level ones
            for k, v in h0.attrs.items():
                out.attrs[k] = v
        # stream each segment into its slice
        off = 0
        for i, m in enumerate(meta):
            off += gap_samps[i]  # leave gap as zero (fillvalue)
            with h5py.File(m["h5"], "r") as hi:
                with h5py.File(args.out_daq, "a") as out:
                    out["analog/samples_int16"][off:off + m["nsamp"], :] = hi["analog/samples_int16"][...]
                    out["digital/packed_samples"][off:off + m["nsamp"], :] = hi["digital/packed_samples"][...]
            off += m["nsamp"]
            print(f"   seg{i} written at sample {off - m['nsamp']}..{off}", flush=True)

    with h5py.File(args.out_daq, "a") as out:
        out.attrs["sample_count"] = total_samps
        out.attrs["sample_rate_hz"] = fs
        out.attrs["created_at"] = meta[0]["created"].isoformat()
        out.attrs["closed_at"] = (meta[-1]["closed"].isoformat() if meta[-1]["closed"] else "")
        out.attrs["recording_complete"] = True
        out.attrs["sample_index_is_contiguous"] = False
        out.attrs["concat_has_padded_gaps"] = True
        out.attrs["concat_n_segments"] = len(meta)
        out.attrs["concat_source_h5"] = json.dumps([m["h5"] for m in meta])
        out.attrs["concat_source_dat"] = json.dumps([m["dat"] for m in meta])
        out.attrs["concat_segment_samples"] = json.dumps([m["nsamp"] for m in meta])
        out.attrs["concat_gap_samples"] = json.dumps(gap_samps)
        out.attrs["concat_boundaries"] = json.dumps(boundaries)
        out.attrs["concat_note"] = ("Camera segments concatenated back-to-back (no gap frames); "
                                    "DAQ gaps zero-padded to preserve wall-clock for sync-pulse "
                                    "alignment to the uninterrupted behavior session + camera.")

    manifest = dict(label=args.label, fs=fs, dims=dims0, total_frame_pairs=total_pairs,
                    total_samples=total_samps, gap_samples=gap_samps,
                    out_dat=str(out_dat), out_daq=str(args.out_daq),
                    segments=[dict(h5=m["h5"], dat=m["dat"], pairs=m["pairs"], nsamp=m["nsamp"],
                                   created=m["created"].isoformat(), recording_complete=m["rec_complete"],
                                   pco_rises=m["pco_rises"]) for m in meta],
                    boundaries=boundaries,
                    notes=[
                        "Camera concatenated back-to-back; camera was off during gaps so no gap frames.",
                        "DAQ gaps zero-padded -> sample timeline stays wall-clock accurate.",
                        "pco_exposure pulses verified == 2*frame_pairs per segment (camera/DAQ agree).",
                        "For behavior alignment: use the 'sync' digital channel; boundaries[] gives the "
                        "sample+frame index of each segment start in the concatenated streams.",
                        "part1 (crash, recording_complete=False) camlog text is truncated ~hundreds of "
                        "frames vs the .dat/DAQ; .dat + DAQ pco pulses are the authoritative frame record.",
                    ])
    man_path = args.out_cam_dir / f"{args.label}_concat_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] manifest -> {man_path}", flush=True)
    print(json.dumps(boundaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
