import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rgcn import utils

from history_validity_calibration import (
    read_triples,
    augment_with_inverse,
    build_sr_history,
    build_so_history,
    build_ro_history,
    build_topk_history_features_dual,
    novelty_bucket_from_history,
    stale_exact_bucket,
    RelationHistoryValidityCalibrator,
)


def load_dump(path: str):
    obj = np.load(path)
    scores = obj["scores"].astype(np.float32)
    triples = obj["triples"].astype(np.int64)
    return scores, triples


def get_augmented_snapshot_sizes(snapshot_list):
    return [2 * len(snap) for snap in snapshot_list]


def safe_div(x, y):
    return 0.0 if y == 0 else x / y


def build_topk_candidate_ids(base_scores, gold_ids, topk_cands):
    """
    base_scores: [B, N]
    gold_ids:    [B]
    returns candidate_ids: [B, K], where gold is always included
    """
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


def scatter_topk_back(full_scores, candidate_ids, adjusted_topk_scores):
    """
    full_scores:         [B, N]
    candidate_ids:       [B, K]
    adjusted_topk_scores:[B, K]
    returns full score matrix where only top-K slots are replaced
    """
    out = full_scores.clone()
    out.scatter_(1, candidate_ids, adjusted_topk_scores)
    return out


@torch.no_grad()
def evaluate_model_filtered(
    model,
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    snapshot_list,
    all_ans_list,
    device,
    topk_cands=256,
):
    model.eval()

    overall = {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0}
    bucket_stats = defaultdict(lambda: {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0})
    relation_error = defaultdict(lambda: {"errors": 0, "stale_repeat_errors": 0})

    snapshot_sizes = get_augmented_snapshot_sizes(snapshot_list)
    assert sum(snapshot_sizes) == len(triples_np), (
        f"Dumped triples count {len(triples_np)} does not match expected augmented snapshot total {sum(snapshot_sizes)}"
    )

    offset = 0
    for time_idx, snap_size in enumerate(snapshot_sizes):
        start = offset
        end = offset + snap_size
        offset = end

        snap_scores_np = scores_np[start:end]
        snap_triples_np = triples_np[start:end]

        full_scores = torch.tensor(snap_scores_np, dtype=torch.float32, device=device)
        snap_triples = torch.tensor(snap_triples_np, dtype=torch.long, device=device)
        rel_ids = snap_triples[:, 1]
        gold_ids = snap_triples[:, 2]

        candidate_ids = build_topk_candidate_ids(full_scores, gold_ids, topk_cands)
        base_scores_topk = torch.gather(full_scores, 1, candidate_ids)

        seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
            query_triples=snap_triples_np,
            candidate_ids=candidate_ids,
            sr_hist=sr_hist,
            so_hist=so_hist,
            ro_hist=ro_hist,
            device=device,
        )

        adjusted_topk_scores, _ = model(
            base_scores_topk,
            rel_ids,
            seen_sr, dt_sr, freq_sr,
            seen_so, dt_so, freq_so,
            seen_ro, dt_ro, freq_ro,
        )

        logits = scatter_topk_back(full_scores, candidate_ids, adjusted_topk_scores)

        _, _, rank_raw, rank_filter = utils.get_total_rank(
            snap_triples,
            logits,
            all_ans_list[time_idx],
            eval_bz=1000,
            rel_predict=0,
        )

        top1 = logits.argmax(dim=1)

        for i in range(logits.size(0)):
            rank = int(rank_filter[i].item())
            s, r, o, t = map(int, snap_triples_np[i])
            pred_o = int(top1[i].item())

            mrr = 1.0 / rank
            h1 = 1.0 if rank <= 1 else 0.0
            h3 = 1.0 if rank <= 3 else 0.0
            h10 = 1.0 if rank <= 10 else 0.0

            overall["MRR"] += mrr
            overall["Hits@1"] += h1
            overall["Hits@3"] += h3
            overall["Hits@10"] += h10
            overall["count"] += 1

            bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)
            bucket_stats[bucket]["MRR"] += mrr
            bucket_stats[bucket]["Hits@1"] += h1
            bucket_stats[bucket]["Hits@3"] += h3
            bucket_stats[bucket]["Hits@10"] += h10
            bucket_stats[bucket]["count"] += 1

            if pred_o != o:
                relation_error[r]["errors"] += 1
                pred_exact_bucket = stale_exact_bucket(s, r, pred_o, t, sr_hist)
                if pred_exact_bucket == "stale":
                    relation_error[r]["stale_repeat_errors"] += 1

    def finalize(stats):
        out = {}
        for k, v in stats.items():
            c = max(v["count"], 1)
            out[k] = {
                "count": int(v["count"]),
                "MRR": safe_div(v["MRR"], c),
                "Hits@1": safe_div(v["Hits@1"], c),
                "Hits@3": safe_div(v["Hits@3"], c),
                "Hits@10": safe_div(v["Hits@10"], c),
            }
        return out

    overall_out = {
        "count": int(overall["count"]),
        "MRR": safe_div(overall["MRR"], overall["count"]),
        "Hits@1": safe_div(overall["Hits@1"], overall["count"]),
        "Hits@3": safe_div(overall["Hits@3"], overall["count"]),
        "Hits@10": safe_div(overall["Hits@10"], overall["count"]),
    }

    bucket_out = finalize(bucket_stats)

    rel_rows = []
    for r, v in relation_error.items():
        rel_rows.append({
            "relation": int(r),
            "errors": int(v["errors"]),
            "stale_repeat_errors": int(v["stale_repeat_errors"]),
            "stale_repeat_error_rate": safe_div(v["stale_repeat_errors"], v["errors"]),
        })
    rel_rows = sorted(rel_rows, key=lambda x: x["stale_repeat_error_rate"], reverse=True)

    return overall_out, bucket_out, rel_rows


@torch.no_grad()
def stale_top1_interference(
    model,
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    batch_size=64,
    topk_cands=256,
):
    model.eval()

    total = 0
    stale_top1 = 0
    n = len(triples_np)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_scores = torch.tensor(scores_np[start:end], dtype=torch.float32, device=device)
        batch_triples = triples_np[start:end]
        rel_ids = torch.tensor(batch_triples[:, 1], dtype=torch.long, device=device)
        gold_ids = torch.tensor(batch_triples[:, 2], dtype=torch.long, device=device)

        candidate_ids = build_topk_candidate_ids(batch_scores, gold_ids, topk_cands)
        base_scores_topk = torch.gather(batch_scores, 1, candidate_ids)

        seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
            query_triples=batch_triples,
            candidate_ids=candidate_ids,
            sr_hist=sr_hist,
            so_hist=so_hist,
            ro_hist=ro_hist,
            device=device,
        )

        adjusted_topk_scores, _ = model(
            base_scores_topk,
            rel_ids,
            seen_sr, dt_sr, freq_sr,
            seen_so, dt_so, freq_so,
            seen_ro, dt_ro, freq_ro,
        )

        logits = scatter_topk_back(batch_scores, candidate_ids, adjusted_topk_scores)
        top1 = logits.argmax(dim=1)

        for i in range(logits.size(0)):
            s, r, o, t = map(int, batch_triples[i])
            true_bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)

            if true_bucket in {"near_repeat", "novel"}:
                total += 1
                pred_o = int(top1[i].item())
                pred_bucket = stale_exact_bucket(s, r, pred_o, t, sr_hist)
                if pred_bucket == "stale":
                    stale_top1 += 1

    return {
        "count": int(total),
        "stale_top1_count": int(stale_top1),
        "stale_top1_rate": safe_div(stale_top1, total),
    }


def train_calibrator(
    model,
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    topk_cands=256,
):
    model.train()

    dataset = TensorDataset(
        torch.tensor(scores_np, dtype=torch.float32),
        torch.tensor(triples_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0

        for batch_scores_cpu, batch_triples_cpu in loader:
            full_scores = batch_scores_cpu.to(device)
            batch_triples = batch_triples_cpu.cpu().numpy()
            rel_ids = torch.tensor(batch_triples[:, 1], dtype=torch.long, device=device)
            gold_ids = torch.tensor(batch_triples[:, 2], dtype=torch.long, device=device)

            candidate_ids = build_topk_candidate_ids(full_scores, gold_ids, topk_cands)
            base_scores_topk = torch.gather(full_scores, 1, candidate_ids)

            seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
                query_triples=batch_triples,
                candidate_ids=candidate_ids,
                sr_hist=sr_hist,
                so_hist=so_hist,
                ro_hist=ro_hist,
                device=device,
            )

            adjusted_topk_scores, _ = model(
                base_scores_topk,
                rel_ids,
                seen_sr, dt_sr, freq_sr,
                seen_so, dt_so, freq_so,
                seen_ro, dt_ro, freq_ro,
            )

            local_gold = (candidate_ids == gold_ids.unsqueeze(1)).long().argmax(dim=1)
            loss = F.cross_entropy(adjusted_topk_scores, local_gold)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * full_scores.size(0)
            total_count += full_scores.size(0)

        print(f"epoch {epoch + 1}: calibration_loss = {safe_div(total_loss, total_count):.6f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--valid-dump", type=str, required=True)
    parser.add_argument("--test-dump", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--num-rels", type=int, required=True,
                        help="base number of relations before inverse augmentation")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "recency_only", "frequency_only"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--topk-cands", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_triples = read_triples(os.path.join(args.data_dir, "train.txt"))
    valid_triples = read_triples(os.path.join(args.data_dir, "valid.txt"))

    train_aug = augment_with_inverse(train_triples, args.num_rels)
    train_valid_aug = augment_with_inverse(train_triples + valid_triples, args.num_rels)

    hist_train_sr = build_sr_history(train_aug)
    hist_train_so = build_so_history(train_aug)
    hist_train_ro = build_ro_history(train_aug)

    hist_train_valid_sr = build_sr_history(train_valid_aug)
    hist_train_valid_so = build_so_history(train_valid_aug)
    hist_train_valid_ro = build_ro_history(train_valid_aug)

    valid_scores, valid_queries = load_dump(args.valid_dump)
    test_scores, test_queries = load_dump(args.test_dump)

    num_relations = 2 * args.num_rels

    data = utils.load_data(args.dataset)
    valid_list, _ = utils.split_by_time(data.valid)
    test_list, _ = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    base_num_rels = data.num_rels

    if base_num_rels != args.num_rels:
        print(f"Warning: args.num_rels={args.num_rels}, but dataset loader says data.num_rels={base_num_rels}")

    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, base_num_rels, num_nodes, False)
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, base_num_rels, num_nodes, False)

    model = RelationHistoryValidityCalibrator(
        num_relations=num_relations,
        mode=args.mode,
    ).to(device)

    print("training RHVC on validation logits using train history only")
    model = train_calibrator(
        model=model,
        scores_np=valid_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        topk_cands=args.topk_cands,
    )

    ckpt_path = os.path.join(args.out_dir, f"rhvc_{args.mode}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("saved calibrator to:", ckpt_path)

    print("evaluating RHVC on validation logits using train history only")
    valid_overall, valid_by_bucket, _ = evaluate_model_filtered(
        model=model,
        scores_np=valid_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
        snapshot_list=valid_list,
        all_ans_list=all_ans_list_valid,
        device=device,
        topk_cands=args.topk_cands,
    )

    print("evaluating RHVC on test logits using train + valid history")
    overall, by_bucket, rel_rows = evaluate_model_filtered(
        model=model,
        scores_np=test_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
        snapshot_list=test_list,
        all_ans_list=all_ans_list_test,
        device=device,
        topk_cands=args.topk_cands,
    )

    interference = stale_top1_interference(
        model=model,
        scores_np=test_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
        device=device,
        batch_size=args.batch_size,
        topk_cands=args.topk_cands,
    )

    out = {
        "mode": args.mode,
        "topk_cands": args.topk_cands,
        "valid_overall_filtered": valid_overall,
        "valid_bucket_metrics_filtered": valid_by_bucket,
        "overall_filtered": overall,
        "bucket_metrics_filtered": by_bucket,
        "stale_top1_interference": interference,
        "relationwise_stale_error_top20": rel_rows[:20],
    }

    out_path = os.path.join(args.out_dir, f"results_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("saved results to:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
