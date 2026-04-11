import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import bisect
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Triple = Tuple[int, int, int, int]


def inverse_softplus(x: float) -> float:
    x = float(x)
    if x <= 0:
        raise ValueError("inverse_softplus expects a positive value")
    return math.log(math.expm1(x))


def read_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            s, r, o, t = map(int, parts[:4])
            triples.append((s, r, o, t))
    return triples


def augment_with_inverse(triples: List[Triple], num_rels: int) -> List[Triple]:
    aug: List[Triple] = []
    for s, r, o, t in triples:
        aug.append((s, r, o, t))
        aug.append((o, r + num_rels, s, t))
    return aug


def build_sr_history(triples: List[Triple]) -> Dict[Tuple[int, int], Dict[int, List[int]]]:
    sr_hist: Dict[Tuple[int, int], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        sr_hist[(s, r)][o].append(t)

    for sr_key in sr_hist:
        for o in sr_hist[sr_key]:
            sr_hist[sr_key][o].sort()

    return sr_hist


def build_so_history(triples: List[Triple]):
    so_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        so_hist[s][o].append(t)
    for s in so_hist:
        for o in so_hist[s]:
            so_hist[s][o].sort()
    return so_hist


def build_ro_history(triples: List[Triple]):
    ro_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        ro_hist[r][o].append(t)
    for r in ro_hist:
        for o in ro_hist[r]:
            ro_hist[r][o].sort()
    return ro_hist


def last_time_before(times: List[int], t: int):
    idx = bisect.bisect_left(times, t) - 1
    if idx < 0:
        return None
    return times[idx]


def freq_before(times: List[int], t: int) -> int:
    return bisect.bisect_left(times, t)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_topk_candidate_ids(base_scores: torch.Tensor, gold_ids: torch.Tensor, topk_cands: int) -> torch.Tensor:
    k = min(topk_cands, base_scores.size(1))
    topk_ids = torch.topk(base_scores, k=k, dim=1).indices

    rows = []
    for i in range(topk_ids.size(0)):
        row = topk_ids[i].tolist()
        g = int(gold_ids[i].item())
        if g not in row:
            row[-1] = g
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long, device=base_scores.device)


def scatter_topk_back(full_scores: torch.Tensor, candidate_ids: torch.Tensor, adjusted_topk_scores: torch.Tensor):
    out = full_scores.clone()
    out.scatter_(1, candidate_ids, adjusted_topk_scores)
    return out


def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
):
    cand_np = _to_numpy(candidate_ids)
    query_np = _to_numpy(query_triples)

    batch_size, k = cand_np.shape

    seen_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_np[i][:4])

        cand_map_sr = sr_hist.get((s, r), {})
        cand_map_so = so_hist.get(s, {})
        cand_map_ro = ro_hist.get(r, {})

        for j, cand_o in enumerate(cand_np[i]):
            cand_o = int(cand_o)

            times_sr = cand_map_sr.get(cand_o, [])
            if times_sr:
                lt = last_time_before(times_sr, t)
                if lt is not None:
                    seen_sr[i, j] = 1.0
                    dt_sr[i, j] = float(t - lt)
                    freq_sr[i, j] = float(freq_before(times_sr, t))

            times_so = cand_map_so.get(cand_o, [])
            if times_so:
                lt = last_time_before(times_so, t)
                if lt is not None:
                    seen_so[i, j] = 1.0
                    dt_so[i, j] = float(t - lt)
                    freq_so[i, j] = float(freq_before(times_so, t))

            times_ro = cand_map_ro.get(cand_o, [])
            if times_ro:
                lt = last_time_before(times_ro, t)
                if lt is not None:
                    seen_ro[i, j] = 1.0
                    dt_ro[i, j] = float(t - lt)
                    freq_ro[i, j] = float(freq_before(times_ro, t))

    return seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro


def novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist):
    times_sr = sr_hist.get((s, r), {}).get(o, [])
    lt_sr = last_time_before(times_sr, t)
    if lt_sr is not None:
        return "repeat"

    times_so = so_hist.get(s, {}).get(o, [])
    lt_so = last_time_before(times_so, t)

    times_ro = ro_hist.get(r, {}).get(o, [])
    lt_ro = last_time_before(times_ro, t)

    if lt_so is not None or lt_ro is not None:
        return "near_repeat"

    return "novel"


def stale_exact_bucket(s, r, o, t, sr_hist):
    times = sr_hist.get((s, r), {}).get(o, [])
    lt = last_time_before(times, t)
    if lt is None:
        return "novel"
    gap = t - lt
    if gap <= 1:
        return "recent"
    if gap <= 10:
        return "mid"
    return "stale"


class RelationHistoryValidityCalibrator(nn.Module):
    """
    Stronger post-hoc relation-aware history-validity calibrator.

    Main ideas preserved:
    - exact branch on (s, r, o)
    - near branches on (s, o) and (r, o)
    - recency / frequency
    - stale-repeat suppression
    - top-K residual calibration

    Strengthening additions:
    - score-aware candidate MLP
    - uncertainty gate
    - relation-conditioned branch parameters
    - bounded bias with learnable base score scaling
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "full",
        rel_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        init_gamma_exact: float = 0.02,
        init_gamma_near: float = 0.10,
        stale_init: float = 0.40,
        init_base_scale: float = 1.0,
        max_bias: float = 2.5,
        use_score_mlp: bool = True,
        use_uncertainty_gate: bool = True,
    ):
        super().__init__()
        assert mode in {"full", "recency_only", "frequency_only", "exact_only"}

        self.mode = mode
        self.max_bias = float(max_bias)
        self.use_score_mlp = bool(use_score_mlp)
        self.use_uncertainty_gate = bool(use_uncertainty_gate)

        # exact branch
        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_recent_sr = nn.Embedding(num_relations, 1)
        self.rel_w_mid_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        # near branches
        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_w_presence_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_w_presence_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        self.rel_context = nn.Embedding(num_relations, rel_emb_dim)

        self.gamma_exact_raw = nn.Parameter(torch.tensor(inverse_softplus(init_gamma_exact), dtype=torch.float32))
        self.gamma_near_raw = nn.Parameter(torch.tensor(inverse_softplus(init_gamma_near), dtype=torch.float32))
        self.base_scale_raw = nn.Parameter(torch.tensor(inverse_softplus(init_base_scale), dtype=torch.float32))

        feature_dim = 17
        gate_dim = 5

        if self.use_score_mlp:
            self.score_mlp = nn.Sequential(
                nn.Linear(feature_dim + rel_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.score_mlp = None

        if self.use_uncertainty_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_dim + rel_emb_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.gate_mlp = None

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_recent_sr.weight, 0.45)
        nn.init.constant_(self.rel_w_mid_sr.weight, 0.12)
        nn.init.constant_(self.rel_w_stale_sr.weight, float(stale_init))

        nn.init.constant_(self.rel_w_presence_so.weight, 0.18)
        nn.init.constant_(self.rel_w_presence_ro.weight, 0.18)

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

        nn.init.normal_(self.rel_context.weight, mean=0.0, std=0.02)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        denom = freq_feat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        freq_feat = (freq_feat / denom) * seen
        return freq_feat

    def _score_context(self, base_scores):
        mean = base_scores.mean(dim=1, keepdim=True)
        std = base_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        centered = (base_scores - mean) / std

        top1 = base_scores.max(dim=1, keepdim=True).values
        gap_from_top1 = top1 - base_scores

        order = torch.argsort(base_scores, dim=1, descending=True)
        rank = torch.argsort(order, dim=1).float()
        denom = max(base_scores.size(1) - 1, 1)
        rank_norm = rank / denom
        return centered, gap_from_top1, rank_norm

    def _bucket_flags(self, seen, dt):
        recent = ((seen > 0) & (dt <= 1)).float()
        mid = ((seen > 0) & (dt > 1) & (dt <= 10)).float()
        stale = ((seen > 0) & (dt > 10)).float()
        return recent, mid, stale

    def _branch_exact(self, rel_ids, seen, dt, freq):
        lam = F.softplus(self.rel_lambda_sr(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = self.rel_w_rec_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = self.rel_w_freq_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wrecent = self.rel_w_recent_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wmid = self.rel_w_mid_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wstale = self.rel_w_stale_sr(rel_ids).squeeze(-1).unsqueeze(1)
        b = self.rel_bias_sr(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)
        recent, mid, stale = self._bucket_flags(seen, dt)

        if self.mode == "recency_only":
            score = wrec * rec + wrecent * recent + 0.5 * wmid * mid - wstale * stale + b
        elif self.mode == "frequency_only":
            score = wfreq * freq_feat + 0.25 * wrecent * recent - wstale * stale + b
        else:
            score = (
                wrec * rec
                + wfreq * freq_feat
                + wrecent * recent
                + 0.5 * wmid * mid
                - wstale * stale
                + b
            )

        return torch.tanh(score) * seen, rec, freq_feat, recent, mid, stale

    def _branch_near(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_wpresence, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        wpres = emb_wpresence(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)

        if self.mode == "recency_only":
            score = wrec * rec + wpres * seen + b
        elif self.mode == "frequency_only":
            score = wfreq * freq_feat + wpres * seen + b
        else:
            score = wrec * rec + wfreq * freq_feat + wpres * seen + b

        return torch.tanh(score) * seen, rec, freq_feat

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr, rec_sr, freq_sr_norm, recent_sr, mid_sr, stale_sr = self._branch_exact(
            rel_ids, seen_sr, dt_sr, freq_sr
        )

        if self.mode == "exact_only":
            g_so = torch.zeros_like(g_sr)
            g_ro = torch.zeros_like(g_sr)
            rec_so = torch.zeros_like(g_sr)
            rec_ro = torch.zeros_like(g_sr)
            freq_so_norm = torch.zeros_like(g_sr)
            freq_ro_norm = torch.zeros_like(g_sr)
            near_presence = torch.zeros_like(g_sr)
            branch_raw = F.softplus(self.gamma_exact_raw) * g_sr
        else:
            g_so, rec_so, freq_so_norm = self._branch_near(
                rel_ids,
                seen_so, dt_so, freq_so,
                self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_w_presence_so, self.rel_bias_so
            )
            g_ro, rec_ro, freq_ro_norm = self._branch_near(
                rel_ids,
                seen_ro, dt_ro, freq_ro,
                self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_w_presence_ro, self.rel_bias_ro
            )
            near_presence = torch.clamp(seen_so + seen_ro, min=0.0, max=2.0) / 2.0
            branch_raw = (
                F.softplus(self.gamma_exact_raw) * g_sr
                + F.softplus(self.gamma_near_raw) * 0.5 * (g_so + g_ro)
            )

        centered_score, gap_from_top1, rank_norm = self._score_context(base_scores)
        hist_presence = torch.clamp(seen_sr + 0.5 * (seen_so + seen_ro), min=0.0, max=1.0)

        rel_ctx = self.rel_context(rel_ids).unsqueeze(1).expand(-1, base_scores.size(1), -1)

        feature_stack = torch.stack(
            [
                base_scores,
                centered_score,
                gap_from_top1,
                rank_norm,
                seen_sr,
                rec_sr,
                freq_sr_norm,
                recent_sr,
                mid_sr,
                stale_sr,
                near_presence,
                seen_so,
                rec_so,
                freq_so_norm,
                seen_ro,
                rec_ro,
                freq_ro_norm,
            ],
            dim=-1,
        )

        if self.score_mlp is not None:
            score_delta = self.score_mlp(torch.cat([feature_stack, rel_ctx], dim=-1)).squeeze(-1)
        else:
            score_delta = torch.zeros_like(base_scores)

        combined_raw = branch_raw + score_delta

        if self.gate_mlp is not None:
            gate_features = torch.stack(
                [
                    centered_score,
                    gap_from_top1,
                    rank_norm,
                    hist_presence,
                    stale_sr,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_mlp(torch.cat([gate_features, rel_ctx], dim=-1)).squeeze(-1))
        else:
            gate = torch.ones_like(base_scores)

        hist_bias = self.max_bias * torch.tanh(gate * combined_raw)
        base_scale = F.softplus(self.base_scale_raw) + 1e-4
        logits = base_scale * base_scores + hist_bias

        return logits, hist_bias
import torch
import torch.nn as nn
import torch.nn.functional as F


Triple = Tuple[int, int, int, int]


def read_triples(path: str) -> List[Triple]:
    triples = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            s, r, o, t = map(int, parts[:4])
            triples.append((s, r, o, t))
    return triples


def augment_with_inverse(triples: List[Triple], num_rels: int) -> List[Triple]:
    aug = []
    for s, r, o, t in triples:
        aug.append((s, r, o, t))
        aug.append((o, r + num_rels, s, t))
    return aug


def build_sr_history(triples: List[Triple]) -> Dict[Tuple[int, int], Dict[int, List[int]]]:
    sr_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        sr_hist[(s, r)][o].append(t)

    for sr_key in sr_hist:
        for o in sr_hist[sr_key]:
            sr_hist[sr_key][o].sort()

    return sr_hist


def build_so_history(triples: List[Triple]):
    so_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        so_hist[s][o].append(t)
    for s in so_hist:
        for o in so_hist[s]:
            so_hist[s][o].sort()
    return so_hist


def build_ro_history(triples: List[Triple]):
    ro_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        ro_hist[r][o].append(t)
    for r in ro_hist:
        for o in ro_hist[r]:
            ro_hist[r][o].sort()
    return ro_hist


def last_time_before(times: List[int], t: int):
    idx = bisect.bisect_left(times, t) - 1
    if idx < 0:
        return None
    return times[idx]


def freq_before(times: List[int], t: int) -> int:
    return bisect.bisect_left(times, t)


def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
):
    if torch.is_tensor(candidate_ids):
        cand_np = candidate_ids.detach().cpu().numpy()
    else:
        cand_np = np.asarray(candidate_ids)

    batch_size, k = cand_np.shape

    seen_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_sr   = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_sr = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_so   = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_so = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    seen_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    dt_ro   = torch.zeros((batch_size, k), dtype=torch.float32, device=device)
    freq_ro = torch.zeros((batch_size, k), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_triples[i])

        cand_map_sr = sr_hist.get((s, r), {})
        cand_map_so = so_hist.get(s, {})
        cand_map_ro = ro_hist.get(r, {})

        for j, cand_o in enumerate(cand_np[i]):
            cand_o = int(cand_o)

            times_sr = cand_map_sr.get(cand_o, [])
            if times_sr:
                lt = last_time_before(times_sr, t)
                if lt is not None:
                    seen_sr[i, j] = 1.0
                    dt_sr[i, j] = float(t - lt)
                    freq_sr[i, j] = float(freq_before(times_sr, t))

            times_so = cand_map_so.get(cand_o, [])
            if times_so:
                lt = last_time_before(times_so, t)
                if lt is not None:
                    seen_so[i, j] = 1.0
                    dt_so[i, j] = float(t - lt)
                    freq_so[i, j] = float(freq_before(times_so, t))

            times_ro = cand_map_ro.get(cand_o, [])
            if times_ro:
                lt = last_time_before(times_ro, t)
                if lt is not None:
                    seen_ro[i, j] = 1.0
                    dt_ro[i, j] = float(t - lt)
                    freq_ro[i, j] = float(freq_before(times_ro, t))

    return seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro


def novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist):
    times_sr = sr_hist.get((s, r), {}).get(o, [])
    lt_sr = last_time_before(times_sr, t)
    if lt_sr is not None:
        return "repeat"

    times_so = so_hist.get(s, {}).get(o, [])
    lt_so = last_time_before(times_so, t)

    times_ro = ro_hist.get(r, {}).get(o, [])
    lt_ro = last_time_before(times_ro, t)

    if lt_so is not None or lt_ro is not None:
        return "near_repeat"

    return "novel"


def stale_exact_bucket(s, r, o, t, sr_hist):
    times = sr_hist.get((s, r), {}).get(o, [])
    lt = last_time_before(times, t)
    if lt is None:
        return "novel"
    gap = t - lt
    if gap <= 1:
        return "recent"
    if gap <= 10:
        return "mid"
    return "stale"


class RelationHistoryValidityCalibrator(nn.Module):
    def __init__(self, num_relations: int, mode: str = "full"):
        super().__init__()
        assert mode in {"full", "recency_only", "frequency_only"}

        self.mode = mode

        # exact branch
        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        # near branches
        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        # softer exact, stronger near
        self.gamma_exact = nn.Parameter(torch.tensor(0.01))
        self.gamma_near = nn.Parameter(torch.tensor(0.08))

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        # softer stale penalty than before
        nn.init.constant_(self.rel_w_stale_sr.weight, 0.3)

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        freq_feat = freq_feat / (freq_feat.max(dim=1, keepdim=True).values + 1e-8)
        freq_feat = freq_feat * seen
        return freq_feat

    def _branch_exact(self, rel_ids, seen, dt, freq):
        lam = F.softplus(self.rel_lambda_sr(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = self.rel_w_rec_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = self.rel_w_freq_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wstale = self.rel_w_stale_sr(rel_ids).squeeze(-1).unsqueeze(1)                                
        b = self.rel_bias_sr(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        stale = (1.0 - rec) * seen
        freq_feat = self._normalize_freq(freq, seen)

        if self.mode == "recency_only":
            score = wrec * rec - wstale * stale + b
        elif self.mode == "frequency_only":
            score = wfreq * freq_feat - wstale * stale + b
        else:
            score = wrec * rec + wfreq * freq_feat - wstale * stale + b

        return torch.tanh(score) * seen

    def _branch_near(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)

        if self.mode == "recency_only":
            score = wrec * rec + b
        elif self.mode == "frequency_only":
            score = wfreq * freq_feat + b
        else:
            score = wrec * rec + wfreq * freq_feat + b

        return torch.tanh(score) * seen

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr = self._branch_exact(rel_ids, seen_sr, dt_sr, freq_sr)

        g_so = self._branch_near(
            rel_ids, seen_so, dt_so, freq_so,
            self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_bias_so
        )

        g_ro = self._branch_near(
            rel_ids, seen_ro, dt_ro, freq_ro,
            self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_bias_ro
        )

        hist_bias = self.gamma_exact * g_sr + self.gamma_near * 0.5 * (g_so + g_ro)
        logits = base_scores + hist_bias
        return logits, hist_bias
