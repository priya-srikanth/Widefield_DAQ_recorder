"""Sign-corrected drop-in for wfield.registration.motion_correct (2D translation).

BUG (wfield 0.4.2, registration.py::registration_upsample): the phase-correlation
offset is applied with a ``+`` sign::

    (xs, ys), _ = cv2.phaseCorrelate(template, dst)
    M = np.float32([[1, 0, xs], [0, 1, ys]])     # <-- WRONG: doubles the drift

phaseCorrelate returns how far ``dst`` is displaced from the template, so aligning
``dst`` TO the template requires shifting by ``-(xs, ys)``. The ``+`` version
*doubles* the drift instead of removing it. This is invisible on sub-pixel sessions
but catastrophic on large drifts -- e.g. PS93 2026-06-06 drifted ~8.5 px and the bug
pushed the corrected movie to ~17 px residual (blurry mean, vessel halos).
Verified on PS93 6/6 reference vs late frames: raw NCC 0.74; +sign 0.39 (worse);
-sign 0.90 (correct).

This module vendors the minimal 2D path with the sign FIXED, reusing wfield's
runpar / chunk_indices / registration_ecc. ``runpar`` uses multiprocessing, so the
fix must live in an importable module (workers import this), not a monkeypatch.
``run_wfield_motion`` imports ``motion_correct`` from here.
"""
import numpy as np
import cv2
from tqdm import tqdm

from wfield.utils import runpar, chunk_indices
from wfield.registration import registration_ecc, _xy_rot_from_affine


def registration_upsample(frame, template):
    h, w = frame.shape
    dst = frame.astype('float32')
    (xs, ys), sf = cv2.phaseCorrelate(template.astype('float32'), dst)
    # SIGN FIX: shift dst by -(xs, ys) to align it TO the template (upstream used +).
    M = np.float32([[1, 0, -xs], [0, 1, -ys]])
    dst = cv2.warpAffine(dst, M, (w, h))
    return (xs, ys), (np.clip(dst, 0, (2 ** 16 - 1))).astype('uint16')


def _register_multichannel_stack(frames, templates, mode='2d', niter=100, eps0=1e-3,
                                 warp_mode=cv2.MOTION_EUCLIDEAN):
    nframes, nchannels, h, w = frames.shape
    if mode == 'ecc':
        hann = cv2.createHanningWindow((w, h), cv2.CV_32FC1)
        hann = (hann * 255).astype('uint8')
    ys = np.zeros((nframes, nchannels), dtype=np.float32)
    xs = np.zeros((nframes, nchannels), dtype=np.float32)
    rot = np.zeros((nframes, nchannels), dtype=np.float32)
    stack = np.zeros_like(frames, dtype='uint16')
    for ichan in range(nchannels):
        chunk = frames[:, ichan].squeeze()
        if mode == '2d':
            res = runpar(registration_upsample, chunk, template=templates[ichan])
            ys[:, ichan] = np.array([r[0][1] for r in res], dtype='float32')
            xs[:, ichan] = np.array([r[0][0] for r in res], dtype='float32')
        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk, template=templates[ichan],
                         hann=hann, niter=niter, eps0=eps0, warp_mode=warp_mode)
            xy, rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:, ichan] = xy[:, 1]
            xs[:, ichan] = xy[:, 0]
            rot[:, ichan] = rots
        stack[:, ichan, :, :] = np.stack([r[1] for r in res])
    return (xs, ys, rot), stack


def motion_correct(dat, out=None, refs=None, chunksize=512, nreference=60,
                   mode='2d', apply_shifts=True):
    """Sign-corrected motion correction. Same signature/returns as wfield's."""
    nframes, nchan, h, w = dat.shape
    if out is None:
        out = dat
    chunks = chunk_indices(nframes, chunksize)
    xshifts, yshifts, rshifts = [], [], []
    # reference: average of frames [nreference:nreference*2] (start of file), self-registered
    if refs is None:
        nreference = int(nreference)
        chunk = np.array(dat[nreference:nreference * 2])
        refs = chunk[0].astype('float32')
        _, refs = _register_multichannel_stack(chunk, refs, mode=mode)
        refs = np.mean(refs, axis=0).astype('float32')
    for c in tqdm(chunks, desc='Motion correction (sign-fixed)'):
        localchunk = np.array(dat[c[0]:c[-1]])
        (xs, ys, rot), corrected = _register_multichannel_stack(localchunk, refs, mode=mode)
        if apply_shifts:
            out[c[0]:c[-1]] = corrected[:]
            if hasattr(out, 'flush'):
                out.flush()
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)
    return (np.vstack(yshifts), np.vstack(xshifts)), np.vstack(rshifts)
