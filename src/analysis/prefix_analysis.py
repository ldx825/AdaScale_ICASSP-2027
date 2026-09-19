"""
prefix_analysis.py
------------------
Scale Utility Audit 的核心统计：

对每个样本，基于 per-scale 特征缓存计算「prefix 累计预测」动态：
    f_t = normalize( sum_{s<=t} w_s * f_s ),  t = 1..10
    logits_t = f_t @ prototypes.T
    y_t = argmax(logits_t)

统计量（全部 label-free 或仅诊断用 GT）：
  - Acc(t), flip rate, t_eq (minimum final-equivalent scale)
  - prediction flip / harmful / rescue 过渡
  - margin / entropy / JS divergence / feature cosine 轨迹
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.load_features import prefix_aggregate

# CLIP 兼容温度（logit_scale=100 → τ=0.01）；cosine logits 差异被放大到合适尺度
LOGIT_TAU = 0.01


# ---------------------------------------------------------------------------
# 基础计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def prefix_logits(feats, prototypes, num_scales=None, device="cpu"):
    """计算所有 prefix 的 logits。

    Args:
        feats: list[Tensor(N,D)]，per-scale L2-normalized 特征（cpu 或 gpu）
        prototypes: Tensor(C,D)
    Returns:
        logits: Tensor(T,N,C)  T=num_scales
        feats_prefix: list[Tensor(N,D)] 每个 prefix 的归一化特征
    """
    T = num_scales or len(feats)
    feats = [f.to(device) for f in feats]
    prototypes = prototypes.to(device)
    all_logits, all_feats = [], []
    for t in range(1, T + 1):
        f = prefix_aggregate(feats, t)
        all_logits.append(f @ prototypes.T)
        all_feats.append(f)
    return torch.stack(all_logits, dim=0), all_feats


@torch.no_grad()
def softmax_stats(logits, tau=LOGIT_TAU):
    """对 logits (T,N,C) 计算温度缩放的 softmax 概率及 top1/top2/margin/entropy。

    Args:
        tau: softmax 温度（default 0.01，即 logit_scale=100，与 CLIP 训练一致）
    Returns dict with arrays (T,N):
        p, top1, top2, top1_score(raw), top2_score(raw), margin(prob), entropy
    """
    T, N, C = logits.shape
    p = F.softmax(logits / tau, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1)
    p1 = top2.values[..., 0]
    p2 = top2.values[..., 1]
    top1 = top2.indices[..., 0]
    raw = torch.topk(logits, k=2, dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return {
        "p": p,
        "top1": top1,
        "top1_class": top1,
        "top1_prob": p1,
        "top2_prob": p2,
        "margin": p1 - p2,
        "top1_raw": raw.values[..., 0],
        "top2_raw": raw.values[..., 1],
        "raw_margin": raw.values[..., 0] - raw.values[..., 1],
        "entropy": entropy,
    }


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence, 对 (..., C) 概率求和，返回 (...)。"""
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log(p.clamp_min(eps) / m.clamp_min(eps))).sum(-1)
    kl_qm = (q * torch.log(q.clamp_min(eps) / m.clamp_min(eps))).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


# ---------------------------------------------------------------------------
# 稳定性 / 过渡统计
# ---------------------------------------------------------------------------

def min_final_equivalent_scale(preds):
    """t_eq: 最小 t（1-based），使得从 t 起所有预测都等于最终预测。

    Args:
        preds: np.ndarray (N, T) int
    Returns:
        np.ndarray (N,) int，值域 [1, T]
    """
    N, T = preds.shape
    teq = np.full(N, T, dtype=np.int64)
    for i in range(N):
        final = preds[i, -1]
        t = T
        for j in range(T - 1, -1, -1):
            if preds[i, j] == final:
                t = j
            else:
                break
        teq[i] = t + 1
    return teq


def flip_statistics(preds):
    """flip 相关统计。

    Args:
        preds: (N, T)
    Returns:
        dict:
          flips: (N, T-1) bool 每个 transition 是否 flip
          n_flips: (N,) 每样本 flip 次数
          flip_rate: (T-1,) 每个 transition 的 flip rate
    """
    flips = preds[:, 1:] != preds[:, :-1]
    return {
        "flips": flips,
        "n_flips": flips.sum(axis=1),
        "flip_rate": flips.mean(axis=0),
    }


def harmful_rescue(labels, preds):
    """基于 GT 的诊断统计。

    harmful: 某 t<10 预测正确，最终（t=10）错误
    rescue : 最终正确，但存在某 t<10 预测错误

    Returns:
        dict:
          harmful: (N,) bool
          rescue: (N,) bool
          harmful_transition: (T-1,) 正确->错误的 transition 次数
          rescue_transition: (T-1,) 错误->正确的 transition 次数
    """
    N, T = preds.shape
    corr = preds == labels[:, None]
    final_ok = corr[:, -1]
    harmful = (corr[:, :-1].any(axis=1)) & (~final_ok)
    rescue = (~final_ok) & final_ok  # placeholder, replaced below
    # rescue: 最终正确，且早期存在错误
    rescue = final_ok & (~corr[:, :-1].all(axis=1))
    # transitions
    harmful_tr = ((corr[:, :-1]) & (~corr[:, 1:])).sum(axis=0)
    rescue_tr = ((~corr[:, :-1]) & (corr[:, 1:])).sum(axis=0)
    return {
        "harmful": harmful,
        "rescue": rescue,
        "harmful_transition": harmful_tr,
        "rescue_transition": rescue_tr,
        "correct": corr,
    }


def first_last_correct_scale(labels, preds):
    """每样本首次/末次预测正确的 scale（无正确则 NaN）。"""
    N, T = preds.shape
    corr = preds == labels[:, None]
    first = np.full(N, np.nan)
    last = np.full(N, np.nan)
    for i in range(N):
        idx = np.where(corr[i])[0]
        if len(idx):
            first[i] = idx[0] + 1
            last[i] = idx[-1] + 1
    return first, last


def prefix_accuracy(preds, labels):
    """Acc(t) for t=1..T。"""
    corr = preds == labels[:, None]
    return corr.mean(axis=0)


def oracle_min_scales(labels, preds):
    """诊断用 oracle：每样本达到最终预测所需最少 scale 的分布（= t_eq）。

    注意这只是「预测一致性」的 oracle；此外报告一个更弱的 oracle:
    从 t=1 开始逐渐增加，检查何时累计预测等于最终预测（即 t_eq）。
    label-aware oracle（需要 GT）另行计算（只在报告中作为上界说明）：
        t_oracle_label[i] = 最小的 t 使得 pred_t == label 且之后不再改变？
    这里提供两个版本。
    """
    teq = min_final_equivalent_scale(preds)
    # label-aware "perfect early exit" oracle: 最少 scales 使得所有已预测样本都正确
    # 定义为：对每个样本，最小的 t 使 pred_t = label（若存在）
    N, T = preds.shape
    corr = preds == labels[:, None]
    t_gt = np.full(N, np.nan)
    for i in range(N):
        idx = np.where(corr[i])[0]
        if len(idx):
            t_gt[i] = idx[0] + 1
    return teq, t_gt
