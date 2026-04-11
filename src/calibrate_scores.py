import os
import json
import copy
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rgcn import utils

try:
    from src.history_validity_calibration import (
        read_triples,
        augment_with_inverse,
        build_sr_history,
        build_so_history,
        build_ro_history,
        build_topk_history_features_dual,
        build_topk_candidate_ids,
        scatter_topk_back,
        novelty_bucket_from_history,
        stale_exact_bucket,
        RelationHistoryValidityCalibrator,
    )
except ImportError:
    from history_validity_calibration import (
        read_triples,
        augment_with_inverse,
        build_sr_history,
        build_so_history,
        build_ro_history,
        build_topk_history_features_dual,
        build_topk_candidate_ids,
        scatter_topk_back,
        novelty_bucket_from_history,
        stale_exact_bucket,
        RelationHistoryValidityCalibrator,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dump(path: str):
    obj = np.load(path)
    scores = obj["scores"].astype(np.float32)
    triples = obj["triples"].astype(np.int64)
    return scores, triples


def get_augmented_snapshot_sizes(snapshot_list):
    return [2 * len(snap) for snap in snapshot_list]


def safe_div(x, y):
    return 0.0 if y == 0 else x / y


def split_valid_dump_by_snapshots(valid_scores, valid_queries, valid_list, valid_all_ans_list, dev_frac=0.2):
    if dev_frac <= 0.0 or len(valid_list) < 2:
        return {
            "train_scores": valid_scores,
            "train_queries": valid_queries,
            "train_list": valid_list,
            "train_all_ans": valid_all_ans_list,
            "dev_scores": None,
            "dev_queries": None,
            "dev_list": None,
            "dev_all_ans": None,
            "num_train_snaps": len(valid_list),
            "num_dev_snaps": 0,
        }

    num_dev = max(1, int(round(len(valid_list) * dev_frac)))
    num_dev = min(num_dev, len(valid_list) - 1)
    num_train = len(valid_list) - num_dev

    snap_sizes = get_augmented_snapshot_sizes(valid_list)
    cut_row = sum(snap_sizes[:num_train])

    return {
        "train_scores": valid_scores[:cut_row],
        "train_queries": valid_queries[:cut_row],
        "train_list": valid_list[:num_train],
        "train_all_ans": valid_all_ans_list[:num_train],
        "dev_scores": valid_scores[cut_row:],
        "dev_queries": valid_queries[cut_row:],
        "dev_list": valid_list[num_train:],
        "dev_all_ans": valid_all_ans_list[num_train:],
        "num_train_snaps": num_train,
        "num_dev_snaps": num_dev,
    }


def finalize_bucket_stats(bucket_stats):
    out = {}
    for k, v in bucket_stats.items():
        c = max(v["count"], 1)
        out[k] = {
            "count": int(v["count"]),
            "MRR": safe_div(v["MRR"], c),
            "Hits@1": safe_div(v["Hits@1"], c),
            "Hits@3": safe_div(v["Hits@3"], c),
            "Hits@10": safe_div(v["Hits@10"], c),
        }
    return out


@torch.no_grad()
def evaluate_scores_filtered(
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    snapshot_list,
    all_ans_list,
    device,
):
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

        logits = torch.tensor(snap_scores_np, dtype=torch.float32, device=device)
        snap_triples = torch.tensor(snap_triples_np, dtype=torch.long, device=device)

        _, _, _, rank_filter = utils.get_total_rank(
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

    overall_out = {
        "count": int(overall["count"]),
        "MRR": safe_div(overall["MRR"], overall["count"]),
        "Hits@1": safe_div(overall["Hits@1"], overall["count"]),
        "Hits@3": safe_div(overall["Hits@3"], overall["count"]),
        "Hits@10": safe_div(overall["Hits@10"], overall["count"]),
    }

    rel_rows = []
    for r, v in relation_error.items():
        rel_rows.append({
            "relation": int(r),
            "errors": int(v["errors"]),
            "stale_repeat_errors": int(v["stale_repeat_errors"]),
            "stale_repeat_error_rate": safe_div(v["stale_repeat_errors"], v["errors"]),
        })
    rel_rows = sorted(rel_rows, key=lambda x: (x["stale_repeat_error_rate"], x["stale_repeat_errors"]), reverse=True)

    return overall_out, finalize_bucket_stats(bucket_stats), rel_rows


@torch.no_grad()
def stale_top1_interference_from_scores(
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
):
    top1 = np.argmax(scores_np, axis=1)
    total = 0
    stale_top1 = 0

    for i in range(len(triples_np)):
        s, r, o, t = map(int, triples_np[i])
        true_bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)

        if true_bucket in {"near_repeat", "novel"}:
            total += 1
            pred_o = int(top1[i])
            pred_bucket = stale_exact_bucket(s, r, pred_o, t, sr_hist)
            if pred_bucket == "stale":
                stale_top1 += 1

    return {
        "count": int(total),
        "stale_top1_count": int(stale_top1),
        "stale_top1_rate": safe_div(stale_top1, total),
    }


@torch.no_grad()
def apply_calibrator_to_scores(
    model,
    scores_np,
    triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    batch_size=128,
    topk_cands=256,
):
    model.eval()
    out_batches = []

    n = len(triples_np)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_scores = torch.tensor(scores_np[start:end], dtype=torch.float32, device=device)
        batch_triples_np = triples_np[start:end]
        batch_triples = torch.tensor(batch_triples_np, dtype=torch.long, device=device)

        rel_ids = batch_triples[:, 1]
        gold_ids = batch_triples[:, 2]

        candidate_ids = build_topk_candidate_ids(batch_scores, gold_ids, topk_cands)
        base_scores_topk = torch.gather(batch_scores, 1, candidate_ids)

        seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
            query_triples=batch_triples_np,
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

        adjusted_scores = scatter_topk_back(batch_scores, candidate_ids, adjusted_topk_scores)
        out_batches.append(adjusted_scores.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out_batches, axis=0)


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
    batch_size=128,
    topk_cands=256,
):
    adjusted_scores_np = apply_calibrator_to_scores(
        model=model,
        scores_np=scores_np,
        triples_np=triples_np,
        sr_hist=sr_hist,
        so_hist=so_hist,
        ro_hist=ro_hist,
        device=device,
        batch_size=batch_size,
        topk_cands=topk_cands,
    )

    overall, bucket, rel_rows = evaluate_scores_filtered(
        scores_np=adjusted_scores_np,
        triples_np=triples_np,
        sr_hist=sr_hist,
        so_hist=so_hist,
        ro_hist=ro_hist,
        snapshot_list=snapshot_list,
        all_ans_list=all_ans_list,
        device=device,
    )
    return adjusted_scores_np, overall, bucket, rel_rows


def compute_delta(base_obj, cal_obj):
    out = {}
    for k in base_obj.keys():
        if isinstance(base_obj[k], (int, float)) and isinstance(cal_obj.get(k, None), (int, float)):
            out[k] = cal_obj[k] - base_obj[k]
    return out


def train_calibrator(
    model,
    train_scores_np,
    train_triples_np,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    dev_scores_np=None,
    dev_triples_np=None,
    dev_snapshot_list=None,
    dev_all_ans_list=None,
    epochs=12,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    topk_cands=256,
    eval_topk_cands=256,
    patience=3,
    min_epochs=4,
    pairwise_weight=0.25,
    margin=0.20,
    bias_reg=1e-4,
    label_smoothing=0.0,
    grad_norm=1.0,
    lr_factor=0.5,
    lr_patience=1,
    min_lr=1e-5,
):
    model.train()

    dataset = TensorDataset(
        torch.tensor(train_scores_np, dtype=torch.float32),
        torch.tensor(train_triples_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if dev_scores_np is not None and dev_snapshot_list is not None and dev_all_ans_list is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )

    best_state = None
    best_epoch = -1
    best_dev_mrr = -1.0
    wait = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_pair = 0.0
        total_bias = 0.0
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

            adjusted_topk_scores, hist_bias = model(
                base_scores_topk,
                rel_ids,
                seen_sr, dt_sr, freq_sr,
                seen_so, dt_so, freq_so,
                seen_ro, dt_ro, freq_ro,
            )

            local_gold = (candidate_ids == gold_ids.unsqueeze(1)).long().argmax(dim=1)

            ce_loss = F.cross_entropy(
                adjusted_topk_scores,
                local_gold,
                label_smoothing=label_smoothing,
            )

            gold_score = adjusted_topk_scores.gather(1, local_gold.unsqueeze(1)).squeeze(1)
            neg_mask = candidate_ids == gold_ids.unsqueeze(1)
            hardest_neg = adjusted_topk_scores.masked_fill(neg_mask, float("-inf")).max(dim=1).values
            pairwise_loss = F.relu(margin - (gold_score - hardest_neg)).mean()

            bias_penalty = hist_bias.pow(2).mean()
            loss = ce_loss + pairwise_weight * pairwise_loss + bias_reg * bias_penalty

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            opt.step()

            batch_n = full_scores.size(0)
            total_loss += float(loss.item()) * batch_n
            total_ce += float(ce_loss.item()) * batch_n
            total_pair += float(pairwise_loss.item()) * batch_n
            total_bias += float(bias_penalty.item()) * batch_n
            total_count += batch_n

        train_loss = safe_div(total_loss, total_count)
        train_ce = safe_div(total_ce, total_count)
        train_pair = safe_div(total_pair, total_count)
        train_bias = safe_div(total_bias, total_count)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_pairwise": train_pair,
            "train_bias_penalty": train_bias,
            "lr": opt.param_groups[0]["lr"],
        }

        if dev_scores_np is not None and dev_snapshot_list is not None and dev_all_ans_list is not None:
            _, dev_overall, _, _ = evaluate_model_filtered(
                model=model,
                scores_np=dev_scores_np,
                triples_np=dev_triples_np,
                sr_hist=sr_hist,
                so_hist=so_hist,
                ro_hist=ro_hist,
                snapshot_list=dev_snapshot_list,
                all_ans_list=dev_all_ans_list,
                device=device,
                batch_size=batch_size,
                topk_cands=eval_topk_cands,
            )
            dev_mrr = dev_overall["MRR"]
            epoch_info["dev_mrr"] = dev_mrr

            if scheduler is not None:
                scheduler.step(dev_mrr)

            if dev_mrr > best_dev_mrr:
                best_dev_mrr = dev_mrr
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            print(
                f"epoch {epoch + 1}: loss={train_loss:.6f} | ce={train_ce:.6f} | pair={train_pair:.6f} "
                f"| bias={train_bias:.6f} | dev_mrr={dev_mrr:.6f} | best_dev_mrr={best_dev_mrr:.6f} | wait={wait}"
            )

            if (epoch + 1) >= min_epochs and wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch}.")
                history.append(epoch_info)
                break
        else:
            print(
                f"epoch {epoch + 1}: loss={train_loss:.6f} | ce={train_ce:.6f} "
                f"| pair={train_pair:.6f} | bias={train_bias:.6f}"
            )

        history.append(epoch_info)

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_dev_mrr": best_dev_mrr,
        "history": history,
    }
    return model, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--valid-dump", type=str, required=True)
    parser.add_argument("--test-dump", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--num-rels", type=int, required=True, help="base relation count before inverse augmentation")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "recency_only", "frequency_only", "exact_only"])

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--topk-cands", type=int, default=256)
    parser.add_argument("--eval-topk-cands", type=int, default=256)

    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=4)

    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--bias-reg", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    parser.add_argument("--rel-emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--init-gamma-exact", type=float, default=0.02)
    parser.add_argument("--init-gamma-near", type=float, default=0.10)
    parser.add_argument("--stale-init", type=float, default=0.40)
    parser.add_argument("--init-base-scale", type=float, default=1.0)
    parser.add_argument("--max-bias", type=float, default=2.5)

    parser.add_argument("--disable-score-mlp", action="store_true", default=False)
    parser.add_argument("--disable-uncertainty-gate", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    set_seed(args.seed)
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

    valid_split = split_valid_dump_by_snapshots(
        valid_scores=valid_scores,
        valid_queries=valid_queries,
        valid_list=valid_list,
        valid_all_ans_list=all_ans_list_valid,
        dev_frac=args.dev_frac,
    )

    model = RelationHistoryValidityCalibrator(
        num_relations=num_relations,
        mode=args.mode,
        rel_emb_dim=args.rel_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        init_gamma_exact=args.init_gamma_exact,
        init_gamma_near=args.init_gamma_near,
        stale_init=args.stale_init,
        init_base_scale=args.init_base_scale,
        max_bias=args.max_bias,
        use_score_mlp=not args.disable_score_mlp,
        use_uncertainty_gate=not args.disable_uncertainty_gate,
    ).to(device)

    print("training RHVC on validation-train logits using train history only")
    model, train_summary = train_calibrator(
        model=model,
        train_scores_np=valid_split["train_scores"],
        train_triples_np=valid_split["train_queries"],
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
        device=device,
        dev_scores_np=valid_split["dev_scores"],
        dev_triples_np=valid_split["dev_queries"],
        dev_snapshot_list=valid_split["dev_list"],
        dev_all_ans_list=valid_split["dev_all_ans"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        topk_cands=args.topk_cands,
        eval_topk_cands=args.eval_topk_cands,
        patience=args.patience,
        min_epochs=args.min_epochs,
        pairwise_weight=args.pairwise_weight,
        margin=args.margin,
        bias_reg=args.bias_reg,
        label_smoothing=args.label_smoothing,
        grad_norm=args.grad_norm,
    )

    ckpt_path = os.path.join(args.out_dir, f"rhvc_{args.mode}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("saved calibrator to:", ckpt_path)

    # Base metrics
    valid_base_overall, valid_base_bucket, valid_base_rel = evaluate_scores_filtered(
        scores_np=valid_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
        snapshot_list=valid_list,
        all_ans_list=all_ans_list_valid,
        device=device,
    )
    valid_base_interference = stale_top1_interference_from_scores(
        scores_np=valid_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
    )

    test_base_overall, test_base_bucket, test_base_rel = evaluate_scores_filtered(
        scores_np=test_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
        snapshot_list=test_list,
        all_ans_list=all_ans_list_test,
        device=device,
    )
    test_base_interference = stale_top1_interference_from_scores(
        scores_np=test_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
    )

    # Calibrated metrics
    print("evaluating calibrated RHVC on full validation logits using train history only")
    valid_adjusted_scores, valid_overall, valid_by_bucket, valid_rel_rows = evaluate_model_filtered(
        model=model,
        scores_np=valid_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
        snapshot_list=valid_list,
        all_ans_list=all_ans_list_valid,
        device=device,
        batch_size=args.eval_batch_size,
        topk_cands=args.eval_topk_cands,
    )
    valid_interference = stale_top1_interference_from_scores(
        scores_np=valid_adjusted_scores,
        triples_np=valid_queries,
        sr_hist=hist_train_sr,
        so_hist=hist_train_so,
        ro_hist=hist_train_ro,
    )

    if valid_split["dev_scores"] is not None:
        print("evaluating calibrated RHVC on held-out validation-dev logits using train history only")
        _, dev_overall, dev_by_bucket, _ = evaluate_model_filtered(
            model=model,
            scores_np=valid_split["dev_scores"],
            triples_np=valid_split["dev_queries"],
            sr_hist=hist_train_sr,
            so_hist=hist_train_so,
            ro_hist=hist_train_ro,
            snapshot_list=valid_split["dev_list"],
            all_ans_list=valid_split["dev_all_ans"],
            device=device,
            batch_size=args.eval_batch_size,
            topk_cands=args.eval_topk_cands,
        )
    else:
        dev_overall, dev_by_bucket = None, None

    print("evaluating calibrated RHVC on test logits using train + valid history")
    test_adjusted_scores, overall, by_bucket, rel_rows = evaluate_model_filtered(
        model=model,
        scores_np=test_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
        snapshot_list=test_list,
        all_ans_list=all_ans_list_test,
        device=device,
        batch_size=args.eval_batch_size,
        topk_cands=args.eval_topk_cands,
    )
    interference = stale_top1_interference_from_scores(
        scores_np=test_adjusted_scores,
        triples_np=test_queries,
        sr_hist=hist_train_valid_sr,
        so_hist=hist_train_valid_so,
        ro_hist=hist_train_valid_ro,
    )

    out = {
        "config": vars(args),
        "num_valid_train_snapshots": valid_split["num_train_snaps"],
        "num_valid_dev_snapshots": valid_split["num_dev_snaps"],
        "training_summary": train_summary,
        "valid_base_overall_filtered": valid_base_overall,
        "valid_base_bucket_metrics_filtered": valid_base_bucket,
        "valid_base_stale_top1_interference": valid_base_interference,
        "valid_overall_filtered": valid_overall,
        "valid_bucket_metrics_filtered": valid_by_bucket,
        "valid_stale_top1_interference": valid_interference,
        "valid_improvement_over_base": compute_delta(valid_base_overall, valid_overall),
        "dev_overall_filtered": dev_overall,
        "dev_bucket_metrics_filtered": dev_by_bucket,
        "test_base_overall_filtered": test_base_overall,
        "test_base_bucket_metrics_filtered": test_base_bucket,
        "test_base_stale_top1_interference": test_base_interference,
        "overall_filtered": overall,
        "bucket_metrics_filtered": by_bucket,
        "stale_top1_interference": interference,
        "test_improvement_over_base": compute_delta(test_base_overall, overall),
        "valid_relationwise_stale_error_top20": valid_rel_rows[:20],
        "test_relationwise_stale_error_top20": rel_rows[:20],
    }

    out_path = os.path.join(args.out_dir, f"results_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("saved results to:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
