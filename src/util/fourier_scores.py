import numpy as np
from scipy.fft import rfft


def fourier_scores(dense, watermarking_fn, legacy=False):
    wf = watermarking_fn
    if hasattr(wf, "num_fns") and not legacy:
        f = rfft(dense, axis=-1)
        f = (f[:, 1:-1] if wf.N % 2 == 0 else f[:, 1:]).astype(np.complex64)
        return np.concatenate((f.real, -f.imag), axis=1) * wf.scaling_factor
    f = rfft(dense, axis=-1)[:, 1:-1].astype(np.complex64)
    return np.concatenate((f.real, f.imag), axis=1) * wf.scaling_factor
