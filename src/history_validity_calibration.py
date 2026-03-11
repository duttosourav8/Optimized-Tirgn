import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
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
    """
    TiRGN predict() uses inverse triples:
      (s, r, o, t) -> (o, r + num_rels, s, t)

    RHVC must use the same augmented triple space.
    """
    aug = []
    for s, r, o, t in triples:
        aug.append((s, r, o, t))
        aug.append((o, r + num_rels, s, t))
    return aug


def build_sr_history(triples: List[Triple]) -> Dict[Tuple[int, int], Dict[int, List[int]]]:
    """
    Build exact history:
      (s, r) -> {o -> sorted list of timestamps}
    """
    sr_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        sr_hist[(s, r)][o].append(t)

    for sr_key in sr_hist:
        for o in sr_hist[sr_key]:
            sr_hist[sr_key][o].sort()

    return sr_hist


def last_time_before(times: List[int], t: int):
    idx = bisect.bisect_left(times, t) - 1
    if idx < 0:
        return None
    return times[idx]


def freq_before(times: List[int], t: int) -> int:
    return bisect.bisect_left(times, t)


def build_dense_history_features(
    query_triples: np.ndarray,
    sr_hist: Dict[Tuple[int, int], Dict[int, List[int]]],
    num_entities: int,
    device: torch.device,
):
    """
    query_triples: [B, 4] with columns [s, r, o, t]

    Returns:
        seen: [B, num_entities]
        dt:   [B, num_entities]
        freq: [B, num_entities]
    """
    batch_size = query_triples.shape[0]

    seen = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_triples[i])
        cand_map = sr_hist.get((s, r), {})
        if not cand_map:
            continue

        for cand_o, times in cand_map.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen[i, cand_o] = 1.0
            dt[i, cand_o] = float(t - lt)
            freq[i, cand_o] = float(freq_before(times, t))

    return seen, dt, freq


def true_bucket_from_history(s: int, r: int, o: int, t: int, sr_hist) -> str:
    cand_map = sr_hist.get((s, r), {})
    times = cand_map.get(o, [])
    lt = last_time_before(times, t)
    if lt is None:
        return "novel"
    gap = t - lt
    if gap <= 1:
        return "recent"
    if gap <= 10:
        return "near"
    return "stale"


class RelationHistoryValidityCalibrator(nn.Module):
    """
    RHVC:
      calibrated_score = base_score + gamma * sigmoid(w_rec * rec + w_freq * log1p(freq) + b) * seen
    """
    def __init__(self, num_relations: int, mode: str = "full"):
        super().__init__()
        assert mode in {"full", "recency_only", "frequency_only"}

        self.mode = mode

        self.rel_lambda = nn.Embedding(num_relations, 1)
        self.rel_w_rec = nn.Embedding(num_relations, 1)
        self.rel_w_freq = nn.Embedding(num_relations, 1)
        self.rel_bias = nn.Embedding(num_relations, 1)
        self.gamma = nn.Parameter(torch.tensor(0.10))

        nn.init.constant_(self.rel_lambda.weight, 0.10)
        nn.init.constant_(self.rel_w_rec.weight, 1.00)
        nn.init.constant_(self.rel_w_freq.weight, 0.50)
        nn.init.constant_(self.rel_bias.weight, 0.00)

    def forward(
        self,
        base_scores: torch.Tensor,
        rel_ids: torch.Tensor,
        seen: torch.Tensor,
        dt: torch.Tensor,
        freq: torch.Tensor,
    ):
        """
        base_scores: [B, E]
        rel_ids: [B]
        seen, dt, freq: [B, E]
        """
        lam = F.softplus(self.rel_lambda(rel_ids)).squeeze(-1).unsqueeze(1) + 1e-8
        w_rec = self.rel_w_rec(rel_ids).squeeze(-1).unsqueeze(1)
        w_freq = self.rel_w_freq(rel_ids).squeeze(-1).unsqueeze(1)
        bias_r = self.rel_bias(rel_ids).squeeze(-1).unsqueeze(1)

        rec = torch.exp(-lam * torch.log1p(torch.clamp(dt, min=0.0))) * seen
        freq_term = torch.log1p(torch.clamp(freq, min=0.0)) * seen

        if self.mode == "recency_only":
            raw = w_rec * rec + bias_r
        elif self.mode == "frequency_only":
            raw = w_freq * freq_term + bias_r
        else:
            raw = w_rec * rec + w_freq * freq_term + bias_r

        validity = torch.sigmoid(raw)
        hist_bias = self.gamma * validity * seen
        calibrated_scores = base_scores + hist_bias

        return calibrated_scores, hist_bias