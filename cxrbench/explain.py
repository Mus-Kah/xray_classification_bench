from __future__ import annotations

import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor, layer, cls: int | None = None):
    """Return Grad-CAM maps and predicted classes for a ConvNeXt-style model."""
    store = {}

    def forward_hook(module, inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(forward_hook)
    model.zero_grad()
    logits = model(x)
    target = logits.argmax(1) if cls is None else torch.tensor([cls], device=x.device)
    logits[torch.arange(len(target)), target].sum().backward()
    handle.remove()

    weights = store["gradients"].mean((2, 3), keepdim=True)
    cam = F.relu((weights * store["activations"]).sum(1, keepdim=True))
    cam = F.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    cam = (cam - cam.amin((1, 2), keepdim=True)) / (
        cam.amax((1, 2), keepdim=True) - cam.amin((1, 2), keepdim=True) + 1e-8
    )
    return (
        cam.detach().cpu().numpy(),
        logits.argmax(1).detach().cpu().numpy(),
    )
