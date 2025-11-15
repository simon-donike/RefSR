import sys
import types

import pytest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("PyTorch is required to run the model tests.", allow_module_level=True)
else:

    def _ensure_torchvision_stub() -> None:
        """Provide a minimal torchvision.ops.deform_conv2d fallback for tests."""

        try:
            import torchvision  # type: ignore  # noqa: F401
        except ModuleNotFoundError:
            torchvision = types.ModuleType("torchvision")
            ops = types.ModuleType("torchvision.ops")

            def deform_conv2d(
                x,
                offset,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                mask=None,
            ):
                return F.conv2d(x, weight, bias, stride, padding, dilation)

            ops.deform_conv2d = deform_conv2d
            torchvision.ops = ops
            sys.modules["torchvision"] = torchvision
            sys.modules["torchvision.ops"] = ops

    _ensure_torchvision_stub()
