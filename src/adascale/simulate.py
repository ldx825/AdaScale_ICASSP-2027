"""
simulate.py
-----------
在缓存的 per-scale 特征上离线模拟「动态多尺度推理」的执行过程，
保证与真实 dynamic execution 的逻辑一致：

  样本按 scale 顺序累积特征；在每个 scale 末尾，策略决定样本是否退出；
  已退出的样本不再参与后续 scale 的计算（其最终预测=退出时的预测）。

设计要点（正确性）:
  - 策略为无状态实现：top1 历史、JS(prev) 等由 simulate() 统一维护并传入，
    避免活跃集收缩导致的状态错位。
  - 聚合权重默认 triangular w_s = s（与官方一致；L2-normalize 消除标量因子）。

所有评估均为 training-free（无任何拟合参数）。
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.load_features import SCALE_WEIGHTS

# CLIP 兼容温度（logit_scale=100 → τ=0.01）；保证 margin/JS 在合理尺度
LOGIT_TAU = 0.01


@dataclass
class SimResult:
    pred: np.ndarray            # (N,) 最终预测
    scales_used: np.ndarray     # (N,) 每样本实际使用的 scale 数
    acc: float
    avg_scales: float
    exit_hist: dict = field(default_factory=dict)


def _normalize(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log(p.clamp_min(eps) / m.clamp_min(eps))).sum(-1)
    kl_qm = (q * torch.log(q.clamp_min(eps) / m.clamp_min(eps))).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


class ExitPolicy:
    """无状态策略接口。decide 返回 (len(idx),) bool: True=退出, False=继续。"""

    def decide(self, t, idx, feat, logits, prob, js_prev, top1_hist):
        raise NotImplementedError


class FixedKPolicy(ExitPolicy):
    def __init__(self, k: int):
        self.k = k

    def decide(self, t, idx, feat, logits, prob, js_prev, top1_hist):
        return np.full(len(idx), t >= self.k)


class StabilityPolicy(ExitPolicy):
    """稳定性 + 置信度早停：连续 r 个 scale top-1 一致 & margin>=delta & JS<=eps。"""

    def __init__(self, r: int = 3, delta: float = 0.5, eps: float = 0.05,
                 min_scale: int = 2, use_js: bool = True, use_margin: bool = True):
        self.r = r
        self.delta = delta
        self.eps = eps
        self.min_scale = min_scale
        self.use_js = use_js
        self.use_margin = use_margin

    def decide(self, t, idx, feat, logits, prob, js_prev, top1_hist):
        n = len(idx)
        if t < max(self.min_scale, self.r):
            return np.zeros(n, dtype=bool)
        ok = np.ones(n, dtype=bool)
        cur = top1_hist[idx, t]
        for j in range(1, self.r):
            ok &= (cur == top1_hist[idx, t - j])
        if self.use_margin:
            top2 = torch.topk(prob, 2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            ok &= margin >= self.delta
        if self.use_js and t >= 2:
            ok &= js_prev.cpu().numpy() <= self.eps
        return ok


class MarginOnlyPolicy(ExitPolicy):
    """仅 margin 阈值（消融用）。"""

    def __init__(self, delta=0.5, min_scale=2):
        self.delta = delta
        self.min_scale = min_scale

    def decide(self, t, idx, feat, logits, prob, js_prev, top1_hist):
        if t < self.min_scale:
            return np.zeros(len(idx), dtype=bool)
        top2 = torch.topk(prob, 2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        return margin >= self.delta


def simulate(feats, prototypes, policy: ExitPolicy, num_scales=None,
             weights=None, device="cpu"):
    """模拟动态执行。

    Args:
        feats: list[Tensor(N,D)] per-scale 特征（原始顺序）
        prototypes: Tensor(C,D)
        policy: ExitPolicy（无状态）
        weights: list[float] 每 scale 权重（默认 w_s = s）
    Returns:
        SimResult（acc 未填充，见 simulate_and_eval）
    """
    T = num_scales or len(feats)
    if weights is None:
        weights = SCALE_WEIGHTS[:T]
    feats = [f.to(device) for f in feats[:T]]
    N = feats[0].shape[0]
    D = feats[0].shape[1]
    C = prototypes.shape[0]
    prototypes = prototypes.to(device)

    prefix_sum = torch.zeros(N, D, device=device)
    pred_final = np.full(N, -1, dtype=np.int64)
    scales_used = np.zeros(N, dtype=np.int64)
    active = np.ones(N, dtype=bool)
    top1_hist = np.full((N, T + 1), -1, dtype=np.int64)
    prev_prob = torch.zeros(N, C, device=device)

    for t in range(1, T + 1):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        prefix_sum[idx] = prefix_sum[idx] + float(weights[t - 1]) * feats[t - 1][idx]
        f_norm = _normalize(prefix_sum[idx])
        logits = f_norm @ prototypes.T
        prob = F.softmax(logits / LOGIT_TAU, dim=-1)
        top1 = logits.argmax(-1).cpu().numpy()
        top1_hist[idx, t] = top1

        if t == 1:
            js_prev = torch.zeros(len(idx), device=device)
        else:
            js_prev = js_divergence(prob, prev_prob[idx])
        prev_prob[idx] = prob.detach()

        pred_final[idx] = top1
        scales_used[idx] = t

        exit_mask = policy.decide(t, idx, f_norm, logits, prob, js_prev, top1_hist)
        active[idx[exit_mask]] = False

    exit_hist = {int(k): int(v) for k, v in
                 zip(*np.unique(scales_used, return_counts=True))}
    return SimResult(pred=pred_final, scales_used=scales_used,
                     acc=float("nan"), avg_scales=float(scales_used.mean()),
                     exit_hist=exit_hist)


def simulate_and_eval(feats, prototypes, policy, labels, num_scales=None,
                      weights=None, device="cpu"):
    res = simulate(feats, prototypes, policy, num_scales, weights, device)
    labels = np.asarray(labels)
    res.acc = float((res.pred == labels).mean())
    return res
