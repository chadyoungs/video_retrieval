"""Normalise the many shapes ``transformers`` can return for features.

``CLIPModel.get_text_features`` historically returned a tensor, but depending on
the transformers version and the checkpoint family it can return a model-output
object (``CLIPTextModelOutput`` / ``BaseModelOutputWithPooling``) instead. Both
image-side helpers and the text encoder must accept either form, otherwise we
crash on ``features.detach()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

#: Attributes carrying an *already projected* embedding, in priority order.
_PROJECTED_ATTRS = ("text_embeds", "image_embeds", "pooler_output")


def to_feature_tensor(features: Any) -> torch.Tensor:
    """Return a 2-D ``[batch, dim]`` tensor from a tensor or a model output."""
    if isinstance(features, torch.Tensor):
        tensor = features
    else:
        tensor = None
        for name in _PROJECTED_ATTRS:
            candidate = getattr(features, name, None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
        if tensor is None:
            hidden = getattr(features, "last_hidden_state", None)
            if isinstance(hidden, torch.Tensor):
                # No projection head available: mean-pool as a last resort.
                tensor = hidden.mean(dim=1)
        if tensor is None:
            raise TypeError(
                "Unsupported feature output: "
                f"{type(features).__name__} has none of "
                f"{', '.join(_PROJECTED_ATTRS + ('last_hidden_state',))}"
            )

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def features_to_array(features: Any) -> np.ndarray:
    """Detached CPU numpy array for a tensor or model-output object."""
    return to_feature_tensor(features).detach().cpu().numpy()
