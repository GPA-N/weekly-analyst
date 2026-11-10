"""Apply a sparse weight patch (edited down_proj columns) to an f16 GGUF.

The patch is a .npz with, for each edited layer L, `L{L}_cols` (column indices of the MLP down projection) and
`L{L}_vals` (the new column values, shape [hidden, n_cols]). The patched file is a copy; the input is never modified.
"""

import re
import shutil
from pathlib import Path

import numpy as np


def apply_patch(gguf_in: Path, patch_npz: Path, gguf_out: Path) -> list[int]:
    """Copy gguf_in to gguf_out, write the patched columns into it, and return the patched layer numbers."""
    from gguf import GGUFReader  # lazy: only the patch step needs it

    shutil.copyfile(gguf_in, gguf_out)
    patch = np.load(patch_npz)
    layers = sorted({int(m.group(1)) for k in patch.files if (m := re.fullmatch(r"L(\d+)_cols", k))})
    reader = GGUFReader(str(gguf_out), "r+")
    by_name = {t.name: t for t in reader.tensors}
    for layer in layers:
        tensor = by_name[f"blk.{layer}.ffn_down.weight"]
        cols, vals = patch[f"L{layer}_cols"], patch[f"L{layer}_vals"]
        data = tensor.data  # row-major [hidden, intermediate], f16
        if data.dtype != np.float16 or data.shape[0] != vals.shape[0] or cols.max() >= data.shape[1]:
            raise ValueError(f"layer {layer}: tensor {data.dtype}{data.shape} does not fit the patch {vals.shape}")
        data[:, cols] = vals.astype(np.float16)
    if hasattr(reader.data, "flush"):
        reader.data.flush()
    return layers
