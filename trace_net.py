"""TRACE-Net: model and loss in one self-contained PyTorch source file.

Traceable Spectral--Spatial Decision Formation Network for Interpretable
Hyperspectral Image Classification.


"""

from __future__ import annotations

import argparse
import json
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SCALE_NAMES = ("Low", "Mid", "High")
ROLE_NAMES = ("Cen", "Con", "Het", "Arr")
EVIDENCE_NAMES = tuple(f"{scale}-{role}" for scale in SCALE_NAMES for role in ROLE_NAMES)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class HaarSpectralSplit(nn.Module):
    """Three-level average--difference Haar split, independently per pixel.

    Low = R(A3; 0, 0, 0), Mid = R(0; 0, D2, D3), High = R(0; D1, 0, 0).
    Replicate the last band to 8*ceil(C/8), then restore the original C bands.
    No parameters, spatial filtering, interpolation, or band selection.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.pad_channels = (-self.in_channels) % 8

    @staticmethod
    def _dwt(x: Tensor) -> tuple[Tensor, Tensor]:
        even, odd = x[:, 0::2], x[:, 1::2]
        return (even + odd) / 2.0, (even - odd) / 2.0

    @staticmethod
    def _idwt(approx: Tensor, detail: Tensor) -> Tensor:
        # Interleave even and odd spectral positions. No division in synthesis.
        return torch.stack((approx + detail, approx - detail), dim=2).flatten(1, 2)

    @classmethod
    def _reconstruct(cls, approx: Tensor, details: tuple[Tensor, ...]) -> Tensor:
        for detail in reversed(details):
            approx = cls._idwt(approx, detail)
        return approx

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B, {self.in_channels}, H, W], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Haar input must be a floating-point tensor")
        if self.pad_channels:
            tail = x[:, -1:].expand(-1, self.pad_channels, -1, -1)
            padded = torch.cat((x, tail), dim=1)
        else:
            padded = x
        a1, d1 = self._dwt(padded)
        a2, d2 = self._dwt(a1)
        a3, d3 = self._dwt(a2)
        zero_a3 = torch.zeros_like(a3)
        zero_d1, zero_d2, zero_d3 = (torch.zeros_like(v) for v in (d1, d2, d3))
        low = self._reconstruct(a3, (zero_d1, zero_d2, zero_d3))
        mid = self._reconstruct(zero_a3, (zero_d1, d2, d3))
        high = self._reconstruct(zero_a3, (d1, zero_d2, zero_d3))
        return tuple(v[:, :self.in_channels].contiguous() for v in (low, mid, high))


class ChannelLayerNorm2d(nn.Module):
    """Normalize only latent channels at each position, never across pixels."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PointwiseEncoder(nn.Module):
    """Exactly the paper's t1, t2, phi sequence (two 1x1 maps, three LNs).

    `spatial_bn` is a historical state-dict name: it is channel LayerNorm,
    NOT BatchNorm, and introduces neither spatial mixing nor batch statistics.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            ChannelLayerNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_bn = ChannelLayerNorm2d(hidden_channels)
        self.spatial_act = nn.ReLU(inplace=True)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            ChannelLayerNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        t1 = self.proj(x)
        t2 = self.spatial_act(self.spatial_bn(t1))
        return self.refine(t2)


class SpectralBranch(nn.Module):
    """One pointwise encoder and one bias-free classifier shared by four roles."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = PointwiseEncoder(in_channels, hidden_channels)
        self.shared_classifier = nn.Linear(hidden_channels, num_classes, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = x.shape
        if min(height, width) < 3 or height % 2 == 0 or width % 2 == 0:
            raise ValueError("Patch height and width must be odd and >= 3 (paper: 9x9)")
        center_index = (height // 2) * width + width // 2
        indices = torch.arange(height * width, device=x.device)
        neighbors = indices[indices != center_index]
        pixels = x.flatten(2).transpose(1, 2)
        neighbor_raw = pixels[:, neighbors]

        # Apply the SAME phi_f to center, individual neighbors, and their raw mean.
        center = self.encoder(pixels[:, center_index, :, None, None]).flatten(1)
        encoded_neighbors = self.encoder(
            neighbor_raw.reshape(batch * neighbors.numel(), channels, 1, 1)
        ).flatten(1).reshape(batch, neighbors.numel(), self.hidden_channels)
        consensus = self.encoder(neighbor_raw.mean(dim=1)[:, :, None, None]).flatten(1)
        neighbor_mean = encoded_neighbors.mean(dim=1)
        heterogeneity = neighbor_mean - consensus

        # D_f(c)=0; the center spectrum never enters this path.
        deviation_flat = encoded_neighbors.new_zeros(batch, height * width, self.hidden_channels)
        deviation_flat[:, neighbors] = encoded_neighbors - neighbor_mean[:, None, :]
        deviation = deviation_flat.transpose(1, 2).reshape(
            batch, self.hidden_channels, height, width
        )
        local_mean = F.avg_pool2d(F.pad(deviation, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        arrangement = F.relu(deviation - local_mean).mean(dim=(2, 3))

        roles = (center, consensus, heterogeneity, arrangement)
        features = torch.stack(roles, dim=1)  # [B, 4, d]
        role_logits = torch.stack(tuple(self.shared_classifier(role) for role in roles), dim=1)
        return features, role_logits


class TraceOutput(NamedTuple):
    """Live forward tensors; detach explicitly when exporting or logging.

    logits: [B,K]; scale_logits: [B,3,K], without global bias;
    evidence: [B,3,4,K]; role_features: [B,3,4,d]; global_bias: [K];
    alphas: [3]; arrangement_scale: scalar (signed, shared across scales).
    """

    logits: Tensor
    scale_logits: Tensor
    evidence: Tensor
    role_features: Tensor
    global_bias: Tensor
    alphas: Tensor
    arrangement_scale: Tensor


class TRACENet(nn.Module):
    """Paper's full TRACE-Net, with no ablation gates or auxiliary classifiers.

    Default d=64, three Haar levels, three independent alpha_f, one signed s.
    Use hidden_channels=48 explicitly only for compatible historical weights.
    The fixed alpha initialization 1/81 corresponds to the default 9x9 patch;
    alternate odd patch sizes do NOT automatically change that initialization.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 64):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.num_classes = _positive_integer(num_classes, "num_classes")
        self.hidden_channels = _positive_integer(hidden_channels, "hidden_channels")
        self.freq_split = HaarSpectralSplit(in_channels)
        self.branches = nn.ModuleDict({
            name: SpectralBranch(in_channels, hidden_channels, num_classes)
            for name in SCALE_NAMES
        })
        self.global_bias = nn.Parameter(torch.zeros(num_classes))
        self.center_context_logits = nn.Parameter(torch.full((3,), -math.log(80.0)))
        self.shared_spatial_scale = nn.Parameter(torch.zeros(()))

    @property
    def alphas(self) -> Tensor:
        return torch.sigmoid(self.center_context_logits)

    def forward(self, x: Tensor, *, return_details: bool = False) -> Tensor | TraceOutput:
        if x.ndim != 4 or x.shape[0] < 1:
            raise ValueError(f"Expected a nonempty [B,C,H,W] batch, got {tuple(x.shape)}")
        components = self.freq_split(x)
        branch_outputs = tuple(
            self.branches[name](part) for name, part in zip(SCALE_NAMES, components)
        )
        raw_logits = torch.stack(tuple(item[1] for item in branch_outputs), dim=1)
        alpha = self.alphas
        role_weights = torch.stack(
            (alpha, 1.0 - alpha, 1.0 - alpha, self.shared_spatial_scale.expand_as(alpha)), dim=1
        )
        evidence = raw_logits * role_weights[None, :, :, None]
        scale_logits = evidence.sum(dim=2)  # z_f = sum_r E[f,r], no b/3 here.
        logits = self.global_bias + scale_logits.sum(dim=1)  # b is added ONCE.
        if not return_details:
            return logits
        return TraceOutput(
            logits=logits,
            scale_logits=scale_logits,
            evidence=evidence,
            role_features=torch.stack(tuple(item[0] for item in branch_outputs), dim=1),
            global_bias=self.global_bias,
            alphas=alpha,
            arrangement_scale=self.shared_spatial_scale,
        )


class CSDCLoss(nn.Module):
    """Mean squared deviation from a detached per-sample mean-logit target.

    For B samples and K classes: sum((z_f - stopgrad(mean_f(z_f)))**2)
    divided by B*3*K. No softmax, KL divergence, temperature, or extra heads.
    """

    def forward(self, scale_logits: Tensor) -> Tensor:
        if scale_logits.ndim != 3 or scale_logits.shape[1] != 3:
            raise ValueError(f"Expected scale_logits [B,3,K], got {tuple(scale_logits.shape)}")
        if scale_logits.shape[0] < 1 or scale_logits.shape[2] < 1:
            raise ValueError("CSDC requires a nonempty batch and at least one class")
        target = scale_logits.mean(dim=1, keepdim=True).detach()
        return (scale_logits - target).square().mean()


class TRACENetLoss(nn.Module):
    """CE(z,y) + lambda_csdc*CSDC(z_l,z_m,z_h); constant lambda=0.05 by default.

    The default return is a scalar tensor suitable for loss.backward().
    return_parts=True returns a dict with live tensors 'loss', 'ce', 'csdc'.
    Labels must be zero-based class IDs; exclude unlabeled/background centers
    when building the training set. No class weighting or label smoothing.
    """

    def __init__(self, lambda_csdc: float = 0.05):
        super().__init__()
        if not math.isfinite(lambda_csdc) or lambda_csdc < 0:
            raise ValueError("lambda_csdc must be finite and nonnegative")
        self.lambda_csdc = float(lambda_csdc)
        self.csdc = CSDCLoss()

    def forward(
        self, logits: Tensor, targets: Tensor, scale_logits: Tensor, *, return_parts: bool = False
    ) -> Tensor | dict[str, Tensor]:
        if logits.ndim != 2 or targets.shape != logits.shape[:1]:
            raise ValueError("Expected logits [B,K] and targets [B]")
        if scale_logits.shape != (logits.shape[0], 3, logits.shape[1]):
            raise ValueError("scale_logits must match logits: expected [B,3,K]")
        ce = F.cross_entropy(logits, targets, reduction="mean")
        csdc = self.csdc(scale_logits)
        total = ce + self.lambda_csdc * csdc
        if return_parts:
            return {"loss": total, "ce": ce, "csdc": csdc}
        return total


def make_optimizer(model: nn.Module) -> torch.optim.Adam:
    """Paper's Adam (not AdamW), constant lr, weight decay on all parameters."""
    return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)


def _smoke_test() -> None:
    """Synthetic CPU integration check only; not a Houston accuracy experiment."""
    torch.set_num_threads(1)
    torch.manual_seed(101)
    model = TRACENet(144, 15).cpu()
    criterion = TRACENetLoss()
    optimizer = make_optimizer(model)
    x = torch.randn(4, 144, 9, 9)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    for _ in range(3):
        output = model(x, return_details=True)
        loss = criterion(output.logits, y, output.scale_logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError("Non-finite loss/gradient or a disconnected parameter")
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(x, return_details=True)
        reconstructed = output.global_bias + output.evidence.sum(dim=2).sum(dim=1)
        torch.testing.assert_close(reconstructed, output.logits, rtol=0, atol=0)
        spectral_error = (sum(model.freq_split(x)) - x).abs().max().item()
    print(json.dumps({
        "test": "synthetic_cpu_smoke_test", "torch": torch.__version__,
        "device": "cpu", "hidden_channels": model.hidden_channels,
        "parameters": sum(p.numel() for p in model.parameters()),
        "logits_shape": list(output.logits.shape), "evidence_shape": list(output.evidence.shape),
        "loss_after_third_forward": loss.item(),
        "haar_max_abs_reconstruction_error": spectral_error,
        "csdc_weight": criterion.lambda_csdc, "status": "PASS",
    }, indent=2))


__all__ = [
    "TRACENet", "TRACENetLoss", "CSDCLoss", "TraceOutput", "HaarSpectralSplit",
    "SCALE_NAMES", "ROLE_NAMES", "EVIDENCE_NAMES", "make_optimizer",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a synthetic CPU forward/backward check"
    )
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()
