"""TorchScript where it is supported.

PyTorch has deprecated TorchScript (torch.jit.script); on Python 3.14+ it warns that it "is not
supported in Python 3.14+ and may break". The Python helpers of this package are therefore
compiled with torch.jit.script on Python < 3.14 (as before) and stay plain Python functions,
calling the same compiled custom ops, on 3.14+. FGC_NO_TORCHSCRIPT=1 forces plain Python on
any version (for testing and timing).
"""

import os
import sys

import torch

USE_TORCHSCRIPT = sys.version_info < (3, 14) and os.environ.get("FGC_NO_TORCHSCRIPT", "0") != "1"


def script(fn):
    """torch.jit.script(fn) where TorchScript is used, otherwise fn unchanged."""
    return torch.jit.script(fn) if USE_TORCHSCRIPT else fn
