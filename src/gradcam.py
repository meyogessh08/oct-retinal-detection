"""Grad-CAM implemented from scratch with PyTorch hooks (no extra library needed).

Idea: take the gradient of the predicted class score with respect to the feature maps of
the last conv layer, average it per channel to get importance weights, then build a
weighted sum of the feature maps. ReLU keeps only regions that push the score UP.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """Return (heatmap in [0,1] sized like the input, predicted class, probabilities)."""
        input_tensor = input_tensor.clone().requires_grad_(True)  # ensures backward works
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].detach()
        if class_idx is None:
            class_idx = int(logits.argmax(1))

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs.cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend the heatmap on top of the original image. Returns an RGB uint8 array."""
    img = np.array(image.convert("RGB").resize((cam.shape[1], cam.shape[0])))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img + alpha * heat)
