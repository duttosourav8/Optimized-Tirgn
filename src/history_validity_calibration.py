import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torchimport bisect
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
    Exact history:
      (s, r) -> {o -> sorted list of timestamps}
    """
    sr_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        sr_hist[(s, r)][o].append(t)

    for sr_key in sr_hist:
        for o in sr_hist[sr_key]:
            sr_hist[sr_key][o].sort()

    return sr_hist


def build_so_history(triples: List[Triple]):
    """
    Subject-object context history:
      s -> {o -> sorted list of timestamps}
    """
    so_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        so_hist[s][o].append(t)
    for s in so_hist:
        for o in so_hist[s]:
            so_hist[s][o].sort()
    return so_hist


def build_ro_history(triples: List[Triple]):
    """
    Relation-object context history:
      r -> {o -> sorted list of timestamps}
    """
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


def build_dense_history_features_dual(
    query_triples,
    sr_hist,
    so_hist,
    ro_hist,
    num_entities,
    device,
):
    batch_size = query_triples.shape[0]

    seen_sr = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_sr   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_sr = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    seen_so = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_so   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_so = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    seen_ro = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_ro   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_ro = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_triples[i])

        cand_map_sr = sr_hist.get((s, r), {})
        for cand_o, times in cand_map_sr.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_sr[i, cand_o] = 1.0
            dt_sr[i, cand_o] = float(t - lt)
            freq_sr[i, cand_o] = float(freq_before(times, t))

        cand_map_so = so_hist.get(s, {})
        for cand_o, times in cand_map_so.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_so[i, cand_o] = 1.0
            dt_so[i, cand_o] = float(t - lt)
            freq_so[i, cand_o] = float(freq_before(times, t))

        cand_map_ro = ro_hist.get(r, {})
        for cand_o, times in cand_map_ro.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_ro[i, cand_o] = 1.0
            dt_ro[i, cand_o] = float(t - lt)
            freq_ro[i, cand_o] = float(freq_before(times, t))

    return seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro


def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
):
    """
    Build RHVC features only for candidate_ids of shape [B, K].
    """
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

        # exact (s, r, o)
        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        # near branch (s, o)
        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        # near branch (r, o)
        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        # reduce exact dominance, boost near-repeat help
        self.gamma_exact = nn.Parameter(torch.tensor(0.02))
        self.gamma_near = nn.Parameter(torch.tensor(0.08))

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_stale_sr.weight, 1.0)

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        freq_feat = freq_feat / (freq_feat.max(dim=1, keepdim=True).values + 1e-8)
        freq_feat = freq_feat * seen
        return freq_feat

    def _branch_exact(self, rel_ids, seen, dt, freq):
        """
        Exact branch must be able to PENALIZE stale repeats.
        """
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
        """
        Near branches should help contextual near-repeat candidates,
        but do not use stale penalty here.
        """
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

def build_so_history(triples):
    so_hist = defaultdict(lambda: defaultdict(list))
    for s, r, o, t in triples:
        so_hist[s][o].append(t)
    for s in so_hist:
        for o in so_hist[s]:
            so_hist[s][o].sort()
    return so_hist


def build_ro_history(triples):
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


def build_dense_history_features_dual(
    query_triples,
    sr_hist,
    so_hist,
    ro_hist,
    num_entities,
    device,
):
    batch_size = query_triples.shape[0]

    seen_sr = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_sr   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_sr = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    seen_so = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_so   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_so = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    seen_ro = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    dt_ro   = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)
    freq_ro = torch.zeros((batch_size, num_entities), dtype=torch.float32, device=device)

    for i in range(batch_size):
        s, r, _, t = map(int, query_triples[i])

        cand_map_sr = sr_hist.get((s, r), {})
        for cand_o, times in cand_map_sr.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_sr[i, cand_o] = 1.0
            dt_sr[i, cand_o] = float(t - lt)
            freq_sr[i, cand_o] = float(freq_before(times, t))

        cand_map_so = so_hist.get(s, {})
        for cand_o, times in cand_map_so.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_so[i, cand_o] = 1.0
            dt_so[i, cand_o] = float(t - lt)
            freq_so[i, cand_o] = float(freq_before(times, t))

        cand_map_ro = ro_hist.get(r, {})
        for cand_o, times in cand_map_ro.items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            seen_ro[i, cand_o] = 1.0
            dt_ro[i, cand_o] = float(t - lt)
            freq_ro[i, cand_o] = float(freq_before(times, t))

    return seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro
def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
):
    """
    Build RHVC features only for candidate_ids of shape [B, K].

    query_triples:
        numpy array or tensor of shape [B, 4]
    candidate_ids:
        tensor or numpy array of shape [B, K]
    """
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

        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr  = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)   # NEW
        self.rel_bias_sr   = nn.Embedding(num_relations, 1)

        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so  = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so   = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro  = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro   = nn.Embedding(num_relations, 1)

        # Slightly reduce exact branch, slightly increase near branch
        self.gamma_exact = nn.Parameter(torch.tensor(0.02))
        self.gamma_near  = nn.Parameter(torch.tensor(0.08))

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_stale_sr.weight, 1.0)   # NEW

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

    def _branch(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        freq_feat = freq_feat / (freq_feat.max(dim=1, keepdim=True).values + 1e-8)
        freq_feat = freq_feat * seen

        if self.mode == "recency_only":
            score = wrec * rec + b
        elif self.mode == "frequency_only":
            score = wfreq * freq_feat + b
        else:
            score = wrec * rec + wfreq * freq_feat + b

        return torch.sigmoid(score) * seen

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr = self._branch(rel_ids, seen_sr, dt_sr, freq_sr,
                            self.rel_lambda_sr, self.rel_w_rec_sr, self.rel_w_freq_sr, self.rel_bias_sr)

        g_so = self._branch(rel_ids, seen_so, dt_so, freq_so,
                            self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_bias_so)

        g_ro = self._branch(rel_ids, seen_ro, dt_ro, freq_ro,
                            self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_bias_ro)

        hist_bias = self.gamma_exact * g_sr + self.gamma_near * 0.5 * (g_so + g_ro)
        logits = base_scores + hist_bias
        return logits, hist_bias
