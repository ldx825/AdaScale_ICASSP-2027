"""
adascale.py
-----------
AdaScale：Training-Free Adaptive Multi-Scale Inference（正式方法接口）。

组件
----
1. AdaScalePolicy — 稳定性驱动的早停策略（无状态，可由 simulate.py 或
   runtime_bench.py 驱动）：
       在 scale t，样本退出当且仅当
         (a) 最近 r 个 scale 的 top-1 预测一致（跨尺度稳定性）
         (b) 当前 margin >= delta（置信度充足）
         (c) JS(p_t, p_{t-1}) <= eps（分布漂移小；可选）
   已退出样本以退出时刻的预测为最终结果；未退出样本到达 T=10 自然结束。

2. 聚合规则 — 保持与 LG-CLIP 官方一致（triangular w_s = s，prefix 归一化）；
   L2-normalize 使权重仅相对比例有意义。

默认超参（development setting = PET + ViT-B/32，将跨数据集冻结）：
    r = 3, delta = 0.25, eps = 0.01（text-proto 协议）
    见 notes/method_decisions.md 的最终冻结记录。

Training-free 保证：无任何可学习参数；不使用 ground truth；无每数据集调参。
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.load_features import SCALE_WEIGHTS

LOGIT_TAU = 0.01  # CLIP 兼容温度（logit_scale=100）


@dataclass
class AdaScaleConfig:
    r: int = 3
    delta: float = 0.25
    eps: float = 0.01
    min_scale: int = 1
    use_js: bool = True
    use_margin: bool = True
    num_scales: int = 10

    def as_dict(self):
        return dict(r=self.r, delta=self.delta, eps=self.eps,
                    min_scale=self.min_scale, use_js=self.use_js,
                    use_margin=self.use_margin, num_scales=self.num_scales)


class AdaScalePolicy:
    """稳定性 + 置信度自适应早停（无状态接口，兼容 simulate.py / runtime_bench.py）。"""

    def __init__(self, config: AdaScaleConfig = None):
        self.cfg = config or AdaScaleConfig()

    def decide(self, t, idx, feat, logits, prob, js_prev, top1_hist):
        c = self.cfg
        n = len(idx)
        if t < max(c.min_scale, c.r):
            return np.zeros(n, dtype=bool)
        ok = np.ones(n, dtype=bool)
        cur = top1_hist[idx, t]
        for j in range(1, c.r):
            ok &= (cur == top1_hist[idx, t - j])
        if c.use_margin:
            top2 = torch.topk(prob, 2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            ok &= margin >= c.delta
        if c.use_js and t >= 2:
            ok &= js_prev.cpu().numpy() <= c.eps
        return ok


def softmax_probs(logits, tau=LOGIT_TAU):
    return F.softmax(logits / tau, dim=-1)
