import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips

    _LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    _LPIPS_AVAILABLE = False

_LPIPS_NET = None


def get_lpips_net(device: str):
    global _LPIPS_NET
    if not _LPIPS_AVAILABLE:
        return None
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips.LPIPS(net="alex").to(device)
        _LPIPS_NET.eval()
    return _LPIPS_NET


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """Converts a tensor in [-1, 1] with shape 1x3xHxW to HxWx3 in [0, 1]."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("to_numpy_image expects a tensor of shape 1x3xHxW.")
    return ((x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)


def compute_psnr(x_true: torch.Tensor, x_rec: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((x_true - x_rec) ** 2).item()
    if mse < 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_ssim(x_true: torch.Tensor, x_rec: torch.Tensor) -> float:
    a = to_numpy_image(x_true)
    b = to_numpy_image(x_rec)
    return float(ssim_fn(a, b, data_range=1.0, channel_axis=2))


def compute_lpips(x_true: torch.Tensor, x_rec: torch.Tensor, device: str) -> float:
    net = get_lpips_net(device)
    if net is None:
        return float("nan")
    with torch.no_grad():
        return float(net(x_true.to(device), x_rec.to(device)).item())


def compute_metrics(
        x_true: torch.Tensor,
        x_rec: torch.Tensor,
        device: str,
        use_lpips: bool = False,
) -> Dict[str, float]:
    metrics = {
        "PSNR": compute_psnr(x_true, x_rec),
        "SSIM": compute_ssim(x_true, x_rec),
    }
    if use_lpips:
        metrics["LPIPS"] = compute_lpips(x_true, x_rec, device)
    return metrics


def compute_metrics_batch(
        x_true: torch.Tensor,
        x_rec: torch.Tensor,
        device: str,
        use_lpips: bool = False,
) -> List[Dict[str, float]]:
    if x_true.shape != x_rec.shape:
        raise ValueError("x_true and x_rec must have the same shape.")
    if x_true.ndim != 4:
        raise ValueError("Expected tensors with shape BxCxHxW.")

    batch_metrics: List[Dict[str, float]] = []
    for b in range(x_true.shape[0]):
        batch_metrics.append(
            compute_metrics(
                x_true[b:b + 1],
                x_rec[b:b + 1],
                device=device,
                use_lpips=use_lpips,
            )
        )
    return batch_metrics


def show_reconstruction_panel(
        x_true: torch.Tensor,
        y_vis: torch.Tensor,
        x_dps: torch.Tensor = None,
        x_pigdm: torch.Tensor = None,
        metrics_dps: Dict[str, float] = None,
        metrics_pigdm: Dict[str, float] = None,
        title: str = "",
        figsize: Tuple[int, int] = (16, 4),
        method=None
) -> None:
    if method is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        images = [x_true, y_vis]
        if method == 'DPS':
            images += [x_dps]
            titles = [
                "Ground truth",
                "Measurement / display",
                _metrics_title("DPS", metrics_dps),
            ]
        elif method == 'PiGDM':
            images += [x_pigdm]
            titles = [
                "Ground truth",
                "Measurement / display",
                _metrics_title("PiGDM", metrics_pigdm),
            ]
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        images = [x_true, y_vis, x_dps, x_pigdm]
        titles = [
            "Ground truth",
            "Measurement / display",
            _metrics_title("DPS", metrics_dps),
            _metrics_title("PiGDM", metrics_pigdm),
        ]
    for ax, img, txt in zip(axes, images, titles):
        ax.imshow(to_numpy_image(img))
        ax.set_title(txt, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def _metrics_title(prefix: str, metrics: Dict[str, float]) -> str:
    keys = ["PSNR", "SSIM"]
    if "LPIPS" in metrics:
        keys.append("LPIPS")
    lines = [prefix]
    for k in keys:
        v = metrics[k]
        if np.isnan(v):
            lines.append(f"{k}=nan")
        elif k == "SSIM":
            lines.append(f"{k}={v:.3f}")
        else:
            lines.append(f"{k}={v:.2f}")
    return "\n".join(lines)


def plot_metric_curves(
        x_values: Sequence[float],
        results_dps: Sequence[Dict[str, float]],
        results_pigdm: Sequence[Dict[str, float]],
        x_label: str,
        title: str,
) -> None:
    metric_names = [k for k in results_dps[0].keys() if k in results_pigdm[0]]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]
    for ax, metric_name in zip(axes, metric_names):
        dps_vals = [m[metric_name] for m in results_dps]
        pigdm_vals = [m[metric_name] for m in results_pigdm]
        ax.plot(x_values, dps_vals, "o-", label="DPS")
        ax.plot(x_values, pigdm_vals, "s-", label="PiGDM")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ===========
# Operators
# ===========

class InpaintingOperator:
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask * x

    def H_pinv(self, y: torch.Tensor) -> torch.Tensor:
        return self.mask * y

    def noisy_pinv(self, residual: torch.Tensor, rt2: float, sigma_y2: float) -> torch.Tensor:
        # mat = H^T (H H^T + sigma_y^2 / r_t^2 I)^(-1) residual
        # for H = diag(mask), this becomes:
        # mat = r_t^2 * mask * residual / (r_t^2 * mask + sigma_y^2)
        denom = rt2 * self.mask + sigma_y2
        return rt2 * self.mask * residual / denom.clamp_min(1e-12)

    def display_measurement(self, y: torch.Tensor) -> torch.Tensor:
        return y


class GaussianBlurOperator:
    def __init__(
            self,
            kernel_size: int = 9,
            sigma: float = 3.0,
            img_h: int = 256,
            img_w: int = 256,
            device: str = "cpu",
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device

        ax = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        padded = torch.zeros(img_h, img_w, device=device)
        padded[:kernel_size, :kernel_size] = kernel
        padded = torch.roll(padded, shifts=(-kernel_size // 2, -kernel_size // 2), dims=(0, 1))

        self.K_fft = torch.fft.fft2(padded)
        self.K_fft_conj = self.K_fft.conj()
        self.K_abs2 = (self.K_fft * self.K_fft_conj).real

    def H(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x)
        return torch.fft.ifft2(X * self.K_fft).real

    def H_pinv(self, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        Y = torch.fft.fft2(y)
        return torch.fft.ifft2(Y * self.K_fft_conj / (self.K_abs2 + eps)).real

    def noisy_pinv(self, residual: torch.Tensor, rt2: float, sigma_y2: float) -> torch.Tensor:
        R = torch.fft.fft2(residual)
        filt = (rt2 * self.K_fft_conj) / (rt2 * self.K_abs2 + sigma_y2)
        return torch.fft.ifft2(R * filt).real

    def display_measurement(self, y: torch.Tensor) -> torch.Tensor:
        return y


class SuperResolutionOperator:
    def __init__(self, factor=4):
        self.factor = factor

    def H(self, x):
        return F.avg_pool2d(x, self.factor)

    def H_pinv(self, y):
        return F.interpolate(y, scale_factor=self.factor, mode='nearest')

    def noisy_pinv(self, residual, rt2, sigma_y2):
        f2 = self.factor ** 2
        scale = rt2 / (rt2 + f2 * sigma_y2)
        return F.interpolate(residual, scale_factor=self.factor, mode='nearest') * scale

    def display_measurement(self, y: torch.Tensor) -> torch.Tensor:
        return y


class ComposedOperator:
    def __init__(self, operators: Sequence[object]):
        self.operators = list(operators)

    def H(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for op in self.operators:
            out = op.H(out)
        return out

    def H_pinv(self, y: torch.Tensor) -> torch.Tensor:
        out = y
        for op in reversed(self.operators):
            out = op.H_pinv(out)
        return out

    def display_measurement(self, y: torch.Tensor) -> torch.Tensor:
        return self.H_pinv(y)

class BlurInpaintSR4Operator:
    """
    Measurement model:
        x -> blur -> inpaint (HR mask) -> downsample x4

    Pseudoinverse used by PiGDM:
        y -> upsample x4 -> re-apply HR mask -> regularized deblur
    """
    def __init__(
        self,
        mask_hr,
        sigma_blur=3.0,
        kernel_size=9,
        deblur_eps=1e-2,
        device="cpu",
    ):
        self.mask_hr = mask_hr
        self.device = device
        self.deblur_eps = deblur_eps

        _, _, H, W = mask_hr.shape

        self.blur_op = GaussianBlurOperator(
            kernel_size=kernel_size,
            sigma=sigma_blur,
            img_h=H,
            img_w=W,
            device=device,
        )
        self.inpaint_op = InpaintingOperator(mask_hr)
        self.sr_op = SuperResolutionOperator(factor=4)

    def H(self, x):
        x = self.blur_op.H(x)
        x = self.inpaint_op.H(x)
        x = self.sr_op.H(x)
        return x

    def H_pinv(self, y):
        # upsample low-res measurement
        x = self.sr_op.H_pinv(y)

        # keep consistency with the HR mask used before downsampling
        x = self.inpaint_op.H_pinv(x)

        # regularized inverse blur (important for stability)
        x = self.blur_op.H_pinv(x, eps=self.deblur_eps)
        return x

    def display_measurement(self, y):
        # for display only: just show the low-res observation upsampled
        return self.sr_op.H_pinv(y)

# ==============
# Mask helpers
# ==============

def make_center_mask(imgshape, crop_h: int = 128, crop_w: int = 128, device: str = "cpu") -> torch.Tensor:
    mask = torch.ones(imgshape, device=device)
    h, w = imgshape[2], imgshape[3]
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    mask[:, :, top:top + crop_h, left:left + crop_w] = 0.0
    return mask


def make_random_mask(imgshape, missing_ratio: float = 0.5, device: str = "cpu") -> torch.Tensor:
    mask = (torch.rand(1, 1, imgshape[2], imgshape[3], device=device) > missing_ratio).float()
    return mask.expand(imgshape)


# =====================
# Diffusion helpers
# =====================

def predict_x0_from_eps(x_t: torch.Tensor, eps_t: torch.Tensor, alphabar_t: float) -> torch.Tensor:
    return (1.0 / math.sqrt(alphabar_t)) * x_t - math.sqrt(1.0 / alphabar_t - 1.0) * eps_t


def ddpm_posterior_mean(x_t: torch.Tensor, eps_t: torch.Tensor, alpha_t: float, beta_t: float,
                        alphabar_t: float) -> torch.Tensor:
    return (1.0 / math.sqrt(alpha_t)) * (x_t - (beta_t / math.sqrt(1.0 - alphabar_t)) * eps_t)


def make_uniform_sampling_schedule(reversed_time_steps: Sequence[int], num_steps: Optional[int] = None) -> List[int]:
    schedule = list(reversed_time_steps)
    if num_steps is None or num_steps >= len(schedule):
        return schedule
    indices = np.linspace(0, len(schedule) - 1, num_steps, dtype=int)
    indices = sorted(set(indices.tolist()))
    return [schedule[i] for i in indices]


def make_measurement(
        x_true: torch.Tensor,
        operator,
        sigma_noise: float = 0.0,
        seed: Optional[int] = None,
) -> torch.Tensor:
    y = operator.H(x_true)
    if sigma_noise > 0:
        if seed is not None:
            gen = torch.Generator(device=x_true.device)
            gen.manual_seed(seed)
            noise = torch.randn(y.shape, generator=gen, device=x_true.device)
        else:
            noise = torch.randn_like(y)
        y = y + sigma_noise * noise
    return y


# =======================
# DPS
# =======================

def dps_reconstruct(
        y: torch.Tensor,
        operator,
        get_eps_with_grad: Callable[[torch.Tensor, int], torch.Tensor],
        imgshape: Sequence[int],
        alpha: np.ndarray,
        beta: np.ndarray,
        alphabar: np.ndarray,
        reversed_time_steps: Sequence[int],
        zeta_scale: float = 0.1,
        x_init: Optional[torch.Tensor] = None,
        verbose: bool = True,
) -> torch.Tensor:
    if x_init is None:
        x = torch.randn(imgshape, device=y.device, requires_grad=True)
    else:
        x = x_init.detach().clone().to(y.device).requires_grad_(True)

    batch_size = x.shape[0]

    for i, t in enumerate(reversed_time_steps):
        eps = get_eps_with_grad(x, t)
        xhat = predict_x0_from_eps(x, eps, alphabar[t])

        residual = operator.H(xhat) - y
        residual_sq = residual.reshape(batch_size, -1).pow(2).sum(dim=1)
        loss = residual_sq.sum()
        grad = torch.autograd.grad(loss, x)[0]

        residual_norm = residual_sq.sqrt()
        zeta_t = (zeta_scale / residual_norm.clamp_min(1e-8)).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            mu = ddpm_posterior_mean(x, eps.detach(), alpha[t], beta[t], alphabar[t])
            drift = zeta_t * grad
            if t > 0:
                x_new = mu + math.sqrt(beta[t]) * torch.randn_like(x) - drift
            else:
                x_new = mu - drift
        x = x_new.detach().requires_grad_(True)

        if verbose and (i == 0 or t % 100 == 0 or t == 0):
            mean_residual = residual_norm.mean().item()
            print(f"DPS  t={t:4d}   mean ||A xhat - y||={mean_residual:.4f}")

    return x.detach()


# =====================
# PiGDM
# =====================

def pigdm_reconstruct(
        y: torch.Tensor,
        operator,
        get_eps_with_grad,
        imgshape,
        alphabar: np.ndarray,
        reversed_time_steps,
        num_steps: int = 100,
        eta: float = 0.0,
        sigma_noise: float = 0.0,
        x_init: Optional[torch.Tensor] = None,
        clip_xhat: bool = True,
        verbose: bool = True,
) -> torch.Tensor:
    """
    PiGDM sampler for VP-SDE.

    Follows Algorithm 1 from the paper:
        x_s = sqrt(alpha_s) * xhat_t + c1 * eps + c2 * eps_theta + sqrt(alpha_t) * g
    """
    device = y.device

    schedule = make_uniform_sampling_schedule(reversed_time_steps, num_steps=num_steps)
    if schedule[-1] != 0:
        schedule.append(0)

    pairs = [(schedule[k], schedule[k + 1]) for k in range(len(schedule) - 1)]

    if x_init is None:
        x = torch.randn(imgshape, device=device)
    else:
        x = x_init.to(device).clone()

    x = x.detach().requires_grad_(True)
    is_noisy = sigma_noise > 1e-12

    for step_idx, (t_cur, t_next) in enumerate(pairs):
        abar_t = float(alphabar[t_cur])
        abar_s = float(alphabar[t_next]) if t_next > 0 else 1.0

        eps_theta = get_eps_with_grad(x, t_cur)
        xhat = (x - math.sqrt(max(1.0 - abar_t, 0.0)) * eps_theta) / math.sqrt(max(abar_t, 1e-12))

        if clip_xhat:
            xhat = xhat.clamp(-1.0, 1.0)

        if is_noisy:
            if not hasattr(operator, "noisy_pinv"):
                raise AttributeError("operator must implement noisy_pinv for sigma_noise > 0.")
            residual = y - operator.H(xhat)
            sigma_t2 = (1.0 - abar_t) / max(abar_t, 1e-12)
            rt2 = sigma_t2 / (sigma_t2 + 1.0)
            mat = operator.noisy_pinv(residual, rt2=rt2, sigma_y2=sigma_noise ** 2)
        else:
            mat = operator.H_pinv(y) - operator.H_pinv(operator.H(xhat))

        mat_x = (mat.detach() * xhat).sum()
        guidance = torch.autograd.grad(mat_x, x, retain_graph=False)[0]
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

        if t_next > 0:
            c1_sq_term = (1.0 - abar_t / max(abar_s, 1e-12)) * (1.0 - abar_s) / max(1.0 - abar_t, 1e-12)
            c1_sq_term = max(c1_sq_term, 0.0)
            c1 = eta * math.sqrt(c1_sq_term)
            c2 = math.sqrt(max(1.0 - abar_s - c1 ** 2, 0.0))
            noise = torch.randn_like(x)
        else:
            c1 = 0.0
            c2 = 0.0
            noise = torch.zeros_like(x)

        with torch.no_grad():
            x_new = (
                    math.sqrt(max(abar_s, 0.0)) * xhat.detach()
                    + c1 * noise
                    + c2 * eps_theta.detach()
                    + math.sqrt(max(abar_t, 1e-12)) * guidance
            )
            x_new = torch.nan_to_num(x_new, nan=0.0, posinf=1.0, neginf=-1.0)
            x_new = x_new.clamp(-1.5, 1.5)

        if verbose and (step_idx == 0 or step_idx == len(pairs) - 1 or (step_idx % 10 == 0)):
            finite_xhat = torch.isfinite(xhat).all().item()
            finite_guid = torch.isfinite(guidance).all().item()
            finite_xnew = torch.isfinite(x_new).all().item()
            print(
                f"PiGDM step {step_idx + 1:3d}/{len(pairs)} | "
                f"t={t_cur:4d}->{t_next:4d} | "
                f"finite(xhat)={finite_xhat} "
                f"finite(g)={finite_guid} "
                f"finite(xnew)={finite_xnew}"
            )

        x = x_new.detach().requires_grad_(True)

    return x.detach()


# ==============================
# Evaluation helpers
# ==============================

def _package_metrics(x_true: torch.Tensor, x_rec: torch.Tensor, device: str, use_lpips: bool):
    if x_true.shape[0] == 1:
        return compute_metrics(x_true, x_rec, device=device, use_lpips=use_lpips)
    return compute_metrics_batch(x_true, x_rec, device=device, use_lpips=use_lpips)


def run_dps_or_pigdm(
        x_true: torch.Tensor,
        y: torch.Tensor,
        operator,
        get_eps_with_grad: Callable[[torch.Tensor, int], torch.Tensor],
        imgshape: Sequence[int],
        alpha: np.ndarray,
        beta: np.ndarray,
        alphabar: np.ndarray,
        reversed_time_steps: Sequence[int],
        dps_kwargs: Optional[Dict] = None,
        pigdm_kwargs: Optional[Dict] = None,
        use_lpips: bool = False,
        method: str = "PiGDM",
        show: bool = False,
        title: str = "",
) -> Dict[str, Dict]:
    dps_kwargs = {} if dps_kwargs is None else dict(dps_kwargs)
    pigdm_kwargs = {} if pigdm_kwargs is None else dict(pigdm_kwargs)
    device = str(x_true.device)

    if method == 'DPS':
        x_dps = dps_reconstruct(
            y=y,
            operator=operator,
            get_eps_with_grad=get_eps_with_grad,
            imgshape=imgshape,
            alpha=alpha,
            beta=beta,
            alphabar=alphabar,
            reversed_time_steps=reversed_time_steps,
            **dps_kwargs,
        )
        x_pigdm = None
        metrics_pigdm = None
        metrics_dps = compute_metrics(x_true, x_dps, device=device, use_lpips=use_lpips)
    elif method == 'PiGDM':
        x_pigdm = pigdm_reconstruct(
            y=y,
            operator=operator,
            get_eps_with_grad=get_eps_with_grad,
            imgshape=imgshape,
            alphabar=alphabar,
            reversed_time_steps=reversed_time_steps,
            **pigdm_kwargs,
        )
        x_dps = None
        metrics_dps = None
        metrics_pigdm = compute_metrics(x_true, x_pigdm, device=device, use_lpips=use_lpips)

    if show:
        y_vis = operator.display_measurement(y) if hasattr(operator, "display_measurement") else operator.H_pinv(y)
        show_reconstruction_panel(
            x_true=x_true,
            y_vis=y_vis,
            x_dps=x_dps,
            x_pigdm=x_pigdm,
            metrics_dps=metrics_dps,
            metrics_pigdm=metrics_pigdm,
            title=title,
            method=method,
        )
    if method == 'DPS':
        key_dict = "metrics_dps"
        metrics = metrics_dps
    elif method == 'PiGDM':
        key_dict = "metrics_pigdm"
        metrics = metrics_pigdm
    return {f"{key_dict}": metrics}


def compare_dps_and_pigdm(
        x_true: torch.Tensor,
        y: torch.Tensor,
        operator,
        get_eps_with_grad: Callable[[torch.Tensor, int], torch.Tensor],
        imgshape: Sequence[int],
        alpha: np.ndarray,
        beta: np.ndarray,
        alphabar: np.ndarray,
        reversed_time_steps: Sequence[int],
        dps_kwargs: Optional[Dict] = None,
        pigdm_kwargs: Optional[Dict] = None,
        use_lpips: bool = False,
        show: bool = True,
        title: str = "",
) -> Dict[str, Union[torch.Tensor, Dict[str, float], List[Dict[str, float]]]]:
    dps_kwargs = {} if dps_kwargs is None else dict(dps_kwargs)
    pigdm_kwargs = {} if pigdm_kwargs is None else dict(pigdm_kwargs)

    x_dps = dps_reconstruct(
        y=y,
        operator=operator,
        get_eps_with_grad=get_eps_with_grad,
        imgshape=imgshape,
        alpha=alpha,
        beta=beta,
        alphabar=alphabar,
        reversed_time_steps=reversed_time_steps,
        **dps_kwargs,
    )
    x_pigdm = pigdm_reconstruct(
        y=y,
        operator=operator,
        get_eps_with_grad=get_eps_with_grad,
        imgshape=imgshape,
        alphabar=alphabar,
        reversed_time_steps=reversed_time_steps,
        **pigdm_kwargs,
    )

    device = str(x_true.device)
    metrics_dps = _package_metrics(x_true, x_dps, device=device, use_lpips=use_lpips)
    metrics_pigdm = _package_metrics(x_true, x_pigdm, device=device, use_lpips=use_lpips)

    y_vis = operator.display_measurement(y) if hasattr(operator, "display_measurement") else operator.H_pinv(y)
    if show:
        if x_true.shape[0] != 1:
            raise ValueError("show=True is only supported for batch size 1.")
        show_reconstruction_panel(
            x_true=x_true,
            y_vis=y_vis,
            x_dps=x_dps,
            x_pigdm=x_pigdm,
            metrics_dps=metrics_dps,
            metrics_pigdm=metrics_pigdm,
            title=title,
        )

    return {
        "x_dps": x_dps,
        "x_pigdm": x_pigdm,
        "metrics_dps": metrics_dps,
        "metrics_pigdm": metrics_pigdm,
        "y_vis": y_vis,
    }

def run_task_on_indices(
    indices,
    task_name,
    dps_kwargs,
    pigdm_kwargs,
    seed_base=1234,
    show_example=False,
    example_idx=None,
    method=None
):
    rows = []

    for k, idx in enumerate(indices):
        set_all_seeds(seed_base + idx)
        x_true = load_image_by_idx(idx)
        imgshape = x_true.shape

        if task_name == "inpaint_center_128":
            mask = make_center_mask(imgshape, crop_h=128, crop_w=128, device=device)
            operator = InpaintingOperator(mask)
            y = make_measurement(x_true, operator, sigma_noise=0.0)
            title = f'Inpainting center 128 x 128 — idx={idx:05d}'

        elif task_name.startswith('inpaint_'):
            ratio = float(task_name.split('_')[-1])
            mask = make_random_mask(imgshape, missing_ratio=ratio, device=device)
            operator = InpaintingOperator(mask)
            y = make_measurement(x_true, operator, sigma_noise=0.0)
            title = f'Inpainting random {ratio:.2f} — idx={idx:05d}'

        elif task_name == 'deblur_clean':
            operator = GaussianBlurOperator(
                kernel_size=9,
                sigma=SIGMA_BLUR,
                img_h=imgshape[2],
                img_w=imgshape[3],
                device=device,
            )
            y = make_measurement(x_true, operator, sigma_noise=0.0)
            title = f'Deblur clean — idx={idx:05d}'

        elif task_name.startswith('deblur_noisy_'):
            sigma_y = float(task_name.split('_')[-1])
            operator = GaussianBlurOperator(
                kernel_size=9,
                sigma=SIGMA_BLUR,
                img_h=imgshape[2],
                img_w=imgshape[3],
                device=device,
            )
            y = make_measurement(x_true, operator, sigma_noise=sigma_y)
            title = f'Deblur noisy sigma_y={sigma_y:.2f} — idx={idx:05d}'

        elif task_name.startswith('compose_blur_inpaint_sr4_'):
            ratio = float(task_name.split('_')[-1])

            mask_hr = make_random_mask(imgshape, missing_ratio=ratio, device=device)
            operator = BlurInpaintSR4Operator(
                mask_hr=mask_hr,
                sigma_blur=SIGMA_BLUR,
                kernel_size=9,
                deblur_eps=1e-2,   # augmente à 5e-2 si PiGDM reste instable
                device=device,
            )
            y = make_measurement(x_true, operator, sigma_noise=0.0)
            title = f'Blur clean + inpaint {ratio:.2f} + x4 SR meas. — idx={idx:05d}'

        else:
            raise ValueError(f'Unknown task {task_name}')

        local_show = show_example and (example_idx is not None) and (idx == example_idx)

        if method is not None:
            out = run_dps_or_pigdm(
                x_true=x_true,
                y=y,
                operator=operator,
                get_eps_with_grad=get_eps_from_model_withgrad,
                imgshape=imgshape,
                alpha=alpha,
                beta=beta,
                alphabar=alphabar,
                reversed_time_steps=reversed_time_steps,
                dps_kwargs=dps_kwargs,
                pigdm_kwargs=pigdm_kwargs,
                use_lpips=False,
                method=method,
                show=local_show,
                title=title,
            )

            rows.append({
                'idx': idx,
                'task': task_name,
                f'{method}_PSNR': out[f'metrics_{method.lower()}']['PSNR'],
                f'{method}_SSIM': out[f'metrics_{method.lower()}']['SSIM'],
            })

        else:
            out = compare_dps_and_pigdm(
                x_true=x_true,
                y=y,
                operator=operator,
                get_eps_with_grad=get_eps_from_model_withgrad,
                imgshape=imgshape,
                alpha=alpha,
                beta=beta,
                alphabar=alphabar,
                reversed_time_steps=reversed_time_steps,
                dps_kwargs=dps_kwargs,
                pigdm_kwargs=pigdm_kwargs,
                use_lpips=False,
                show=local_show,
                title=title,
            )

            rows.append({
                'idx': idx,
                'task': task_name,
                'DPS_PSNR': out['metrics_dps']['PSNR'],
                'PiGDM_PSNR': out['metrics_pigdm']['PSNR'],
                'DPS_SSIM': out['metrics_dps']['SSIM'],
                'PiGDM_SSIM': out['metrics_pigdm']['SSIM'],
            })

    return pd.DataFrame(rows)

def summarize_two_method_df(df, task_label):
    return pd.DataFrame([
        {
            'Task': task_label,
            'Method': 'DPS',
            'PSNR': format_mean_std(df['DPS_PSNR'].mean(), df['DPS_PSNR'].std(ddof=1)),
            'SSIM': format_mean_std(df['DPS_SSIM'].mean(), df['DPS_SSIM'].std(ddof=1), nd=3),
        },
        {
            'Task': task_label,
            'Method': 'PiGDM',
            'PSNR': format_mean_std(df['PiGDM_PSNR'].mean(), df['PiGDM_PSNR'].std(ddof=1)),
            'SSIM': format_mean_std(df['PiGDM_SSIM'].mean(), df['PiGDM_SSIM'].std(ddof=1), nd=3),
        },
    ])
