import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from src.rrgcn import RecurrentRGCN
from src.history_validity_gate import (
    triples_array_to_list,
    augment_with_inverse,
    build_sr_history,
    build_so_history,
    build_ro_history,
)


def save_json(obj, path):
    if path == "":
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_hva_histories(data, num_rels):
    train_triples = triples_array_to_list(data.train)
    valid_triples = triples_array_to_list(data.valid)

    train_aug = augment_with_inverse(train_triples, num_rels)
    train_valid_aug = augment_with_inverse(train_triples + valid_triples, num_rels)

    train_hist = {
        "sr": build_sr_history(train_aug),
        "so": build_so_history(train_aug),
        "ro": build_ro_history(train_aug),
    }
    train_valid_hist = {
        "sr": build_sr_history(train_valid_aug),
        "so": build_so_history(train_valid_aug),
        "ro": build_ro_history(train_valid_aug),
    }
    return train_hist, train_valid_hist


class SparseHistoryMatrixCache:
    """
    LRU-bounded sparse history cache.
    This keeps the original method semantics but prevents RAM growth from caching
    every timestamp forever.
    """

    def __init__(self, dataset: str, max_size: int = 24):
        self.history_dir = os.path.join("..", "data", dataset, "history")
        self.tail_cache: "OrderedDict[int, sp.csr_matrix]" = OrderedDict()
        self.rel_cache: "OrderedDict[int, sp.csr_matrix]" = OrderedDict()
        self.max_size = int(max_size)

    def _evict_if_needed(self, cache: OrderedDict):
        while len(cache) > self.max_size:
            cache.popitem(last=False)

    def _load_sparse(self, kind: str, timestamp: int):
        timestamp = int(timestamp)
        if kind == "tail":
            cache = self.tail_cache
            filename = f"tail_history_{timestamp}.npz"
        elif kind == "rel":
            cache = self.rel_cache
            filename = f"rel_history_{timestamp}.npz"
        else:
            raise ValueError(f"Unknown sparse history kind: {kind}")

        if timestamp in cache:
            cache.move_to_end(timestamp)
            return cache[timestamp]

        path = os.path.join(self.history_dir, filename)
        cache[timestamp] = sp.load_npz(path).tocsr()
        cache.move_to_end(timestamp)
        self._evict_if_needed(cache)
        return cache[timestamp]

    def get_one_hot_sequences(self, histroy_data_np, timestamp: int, num_nodes: int, num_rels: int, use_cuda: bool, gpu: int):
        timestamp = int(timestamp)

        all_tail_seq = self._load_sparse("tail", timestamp)
        seq_idx = histroy_data_np[:, 0] * num_rels * 2 + histroy_data_np[:, 1]
        tail_seq = torch.from_numpy(np.asarray(all_tail_seq[seq_idx].todense(), dtype=np.float32))
        one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

        all_rel_seq = self._load_sparse("rel", timestamp)
        rel_seq_idx = histroy_data_np[:, 0] * num_nodes + histroy_data_np[:, 2]
        rel_seq = torch.from_numpy(np.asarray(all_rel_seq[rel_seq_idx].todense(), dtype=np.float32))
        one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda(gpu, non_blocking=True)
            one_hot_rel_seq = one_hot_rel_seq.cuda(gpu, non_blocking=True)

        return one_hot_tail_seq, one_hot_rel_seq


class BoundedGraphCache:
    """
    LRU-bounded graph cache.
    Only used for HVA runs. Baseline stays on original on-demand graph path.
    """

    def __init__(self, num_nodes: int, num_rels: int, gpu: int, max_size: int = 24):
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.gpu = gpu
        self.max_size = int(max_size)
        self.cache: "OrderedDict[int, object]" = OrderedDict()

    def get(self, ts: int, snap):
        ts = int(ts)
        if ts in self.cache:
            self.cache.move_to_end(ts)
            return self.cache[ts]

        g = build_sub_graph(self.num_nodes, self.num_rels, snap, False, self.gpu)
        self.cache[ts] = g
        self.cache.move_to_end(ts)

        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return g


def build_history_glist_from_times(time_ids, snap_map, num_nodes, num_rels, use_cuda, gpu, graph_cache=None):
    if graph_cache is None:
        return [
            build_sub_graph(num_nodes, num_rels, snap_map[int(ts)], use_cuda, gpu)
            for ts in time_ids
        ]
    return [
        graph_cache.get(int(ts), snap_map[int(ts)])
        for ts in time_ids
    ]


def test(
    model,
    history_list,
    history_times,
    test_list,
    test_times,
    num_rels,
    num_nodes,
    use_cuda,
    all_ans_list,
    all_ans_r_list,
    ckpt_path,
    static_graph,
    history_time_nogt,
    mode,
    args,
    hva_histories=None,
    graph_cache=None,
    sequence_cache=None,
    snap_map=None,
):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    dump_scores = []
    dump_triples = []

    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(ckpt_path, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        print("Load checkpoint:", ckpt_path, "epoch:", checkpoint["epoch"])
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    if args.multi_step:
        input_list = [snap for snap in history_list[-args.test_history_len:]]
        all_tail_seq = sequence_cache._load_sparse("tail", int(history_time_nogt))
        all_rel_seq = sequence_cache._load_sparse("rel", int(history_time_nogt))
    else:
        input_time_list = [int(ts) for ts in history_times[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        current_ts = int(test_times[time_idx])

        if args.multi_step:
            history_glist = [
                build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                for g in input_list
            ]
        else:
            history_glist = build_history_glist_from_times(
                time_ids=input_time_list,
                snap_map=snap_map,
                num_nodes=num_nodes,
                num_rels=num_rels,
                use_cuda=use_cuda,
                gpu=args.gpu,
                graph_cache=graph_cache,
            )

        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.cuda(args.gpu)

        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data_np = torch.cat([histroy_data, inverse_histroy_data]).cpu().numpy()

        if args.multi_step:
            seq_idx = histroy_data_np[:, 0] * num_rels * 2 + histroy_data_np[:, 1]
            tail_seq = torch.from_numpy(np.asarray(all_tail_seq[seq_idx].todense(), dtype=np.float32))
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

            rel_seq_idx = histroy_data_np[:, 0] * num_nodes + histroy_data_np[:, 2]
            rel_seq = torch.from_numpy(np.asarray(all_rel_seq[rel_seq_idx].todense(), dtype=np.float32))
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

            if use_cuda:
                one_hot_tail_seq = one_hot_tail_seq.cuda(args.gpu, non_blocking=True)
                one_hot_rel_seq = one_hot_rel_seq.cuda(args.gpu, non_blocking=True)
        else:
            one_hot_tail_seq, one_hot_rel_seq = sequence_cache.get_one_hot_sequences(
                histroy_data_np,
                current_ts,
                num_nodes,
                num_rels,
                use_cuda,
                args.gpu,
            )

        test_triples, final_score, final_r_score = model.predict(
            history_glist,
            num_rels,
            static_graph,
            test_triples_input,
            one_hot_tail_seq,
            one_hot_rel_seq,
            use_cuda,
            hva_histories=hva_histories,
        )

        if args.dump_full_scores:
            dump_scores.append(final_score.detach().cpu().numpy().astype(np.float32))
            dump_triples.append(test_triples.detach().cpu().numpy().astype(np.int64))

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_time_list.pop(0)
            input_time_list.append(current_ts)

    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    if args.dump_full_scores:
        if args.full_score_path == "":
            raise ValueError("--dump-full-scores requires --full-score-path")
        os.makedirs(os.path.dirname(args.full_score_path), exist_ok=True)
        all_scores = np.concatenate(dump_scores, axis=0)
        all_triples = np.concatenate(dump_triples, axis=0)
        np.savez_compressed(args.full_score_path, scores=all_scores, triples=all_triples)
        print("Saved full scores to:", args.full_score_path)
        print("Scores shape:", all_scores.shape, "Triples shape:", all_triples.shape)

    return (
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r,
        hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r
    )


def run_experiment(args):
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time(data.train)
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)

    time_interval = train_times[1] - train_times[0]
    print("num_times", num_times, "time_interval", time_interval)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    history_val_time_nogt = int(valid_times[0])
    history_test_time_nogt = int(test_times[0])

    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}".format(
        args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers,
        args.dilate_len, args.train_history_len, args.weight, args.discount,
        args.angle, args.dropout, args.input_dropout, args.hidden_dropout,
        args.feat_dropout, args.gpu, args.save
    )

    os.makedirs("../models", exist_ok=True)
    model_state_file = os.path.join("../models", model_name)

    load_ckpt_path = model_state_file
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        load_ckpt_path = args.resume_ckpt

    print("Checkpoint used:", load_ckpt_path)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    train_snap_map = {int(ts): snap for snap, ts in zip(train_list, train_times)}
    valid_snap_map = {int(ts): snap for snap, ts in zip(valid_list, valid_times)}
    test_snap_map = {int(ts): snap for snap, ts in zip(test_list, test_times)}
    all_snap_map = {}
    all_snap_map.update(train_snap_map)
    all_snap_map.update(valid_snap_map)
    all_snap_map.update(test_snap_map)

    # Baseline path stays clean/native: no graph cache.
    # HVA path gets bounded graph caching.
    graph_cache = BoundedGraphCache(
        num_nodes=num_nodes,
        num_rels=num_rels,
        gpu=args.gpu,
        max_size=args.graph_cache_size,
    ) if args.use_history_gate else None

    # Sparse sequence cache is safe for both baseline and HVA, but bounded.
    sequence_cache = SparseHistoryMatrixCache(args.dataset, max_size=args.sparse_cache_size)

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list(f"../data/{args.dataset}/e-w-graph.txt", {}, {}, load_time=False)
        )
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = (
            torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu)
            if use_cuda else
            torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        )
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    model = RecurrentRGCN(
        args.decoder,
        args.encoder,
        num_nodes,
        num_rels,
        num_static_rels,
        num_words,
        num_times,
        time_interval,
        args.n_hidden,
        args.opn,
        args.history_rate,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_basis=args.n_basis,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        aggregation=args.aggregation,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis,
        use_history_gate=args.use_history_gate,
        hva_topk=args.hva_topk,
        hva_mode=args.hva_mode,
        hva_gamma_exact=args.hva_gamma_exact,
        hva_gamma_near=args.hva_gamma_near,
        hva_stale_init=args.hva_stale_init,
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    hva_hist_train = None
    hva_hist_train_valid = None
    if args.use_history_gate:
        hva_hist_train, hva_hist_train_valid = build_hva_histories(data, num_rels)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_log = {
        "config": vars(args),
        "epochs": [],
        "best_mrr": 0.0,
        "best_epoch": None,
    }

    best_mrr = 0.0
    start_epoch = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mrr = ckpt.get("best_mrr", 0.0)
        train_log["best_mrr"] = best_mrr
        print(f"Resumed from epoch {start_epoch}, best_mrr={best_mrr:.6f}")

    def resolve_train_log_path():
        if args.train_log_path:
            return args.train_log_path
        if args.ckpt_dir:
            return os.path.join(args.ckpt_dir, "training_log.json")
        return ""

    train_log_path = resolve_train_log_path()

    if args.eval_mode == "dump_valid":
        return test(
            model,
            train_list,
            train_times,
            valid_list,
            valid_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_valid,
            all_ans_list_r_valid,
            load_ckpt_path,
            static_graph,
            history_val_time_nogt,
            mode="test",
            args=args,
            hva_histories=hva_hist_train,
            graph_cache=graph_cache,
            sequence_cache=sequence_cache,
            snap_map=all_snap_map,
        )

    if args.eval_mode == "dump_test":
        return test(
            model,
            train_list + valid_list,
            list(train_times) + list(valid_times),
            test_list,
            test_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            load_ckpt_path,
            static_graph,
            history_test_time_nogt,
            mode="test",
            args=args,
            hva_histories=hva_hist_train_valid,
            graph_cache=graph_cache,
            sequence_cache=sequence_cache,
            snap_map=all_snap_map,
        )

    if args.test:
        return test(
            model,
            train_list + valid_list,
            list(train_times) + list(valid_times),
            test_list,
            test_times,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            load_ckpt_path,
            static_graph,
            history_test_time_nogt,
            mode="test",
            args=args,
            hva_histories=hva_hist_train_valid,
            graph_cache=graph_cache,
            sequence_cache=sequence_cache,
            snap_map=all_snap_map,
        )

    print("---------------------------------------- start training ----------------------------------------")
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        losses, losses_e, losses_r, losses_static = [], [], [], []

        idx = list(range(len(train_list)))
        random.shuffle(idx)

        for train_sample_num in tqdm(idx):
            if train_sample_num == 0:
                continue

            if train_sample_num - args.train_history_len < 0:
                input_time_list = [int(ts) for ts in train_times[0:train_sample_num]]
            else:
                input_time_list = [int(ts) for ts in train_times[train_sample_num - args.train_history_len: train_sample_num]]

            history_glist = build_history_glist_from_times(
                time_ids=input_time_list,
                snap_map=all_snap_map,
                num_nodes=num_nodes,
                num_rels=num_rels,
                use_cuda=use_cuda,
                gpu=args.gpu,
                graph_cache=graph_cache,
            )

            output_np = train_list[train_sample_num]
            if use_cuda:
                output = torch.from_numpy(output_np).long().cuda(args.gpu)
            else:
                output = torch.from_numpy(output_np).long()

            histroy_data = output
            inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
            inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
            histroy_data_np = torch.cat([histroy_data, inverse_histroy_data]).cpu().numpy()

            one_hot_tail_seq, one_hot_rel_seq = sequence_cache.get_one_hot_sequences(
                histroy_data_np,
                int(train_times[train_sample_num]),
                num_nodes,
                num_rels,
                use_cuda,
                args.gpu,
            )

            loss_e, loss_r, loss_static = model.get_loss(
                history_glist,
                output,
                static_graph,
                one_hot_tail_seq,
                one_hot_rel_seq,
                use_cuda,
                hva_histories=hva_hist_train,
            )

            loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static

            losses.append(loss.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_loss_entity": float(np.mean(losses_e)),
            "train_loss_relation": float(np.mean(losses_r)),
            "train_loss_static": float(np.mean(losses_static)),
            "val_mrr_filter": None,
            "is_best": False,
        }

        if args.ckpt_dir:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            latest_path = os.path.join(args.ckpt_dir, "latest.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mrr": best_mrr,
                },
                latest_path,
            )

        if epoch and ((epoch + 1) % args.evaluate_every == 0):
            _, mrr_filter, _, mrr_filter_r, _, _, _, _ = test(
                model,
                train_list,
                train_times,
                valid_list,
                valid_times,
                num_rels,
                num_nodes,
                use_cuda,
                all_ans_list_valid,
                all_ans_list_r_valid,
                model_state_file,
                static_graph,
                history_val_time_nogt,
                mode="train",
                args=args,
                hva_histories=hva_hist_train,
                graph_cache=graph_cache,
                sequence_cache=sequence_cache,
                snap_map=all_snap_map,
            )

            current_mrr = mrr_filter if not args.relation_evaluation else mrr_filter_r
            epoch_row["val_mrr_filter"] = float(current_mrr)

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                train_log["best_mrr"] = float(best_mrr)
                train_log["best_epoch"] = int(epoch)
                epoch_row["is_best"] = True

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_mrr": best_mrr,
                    },
                    model_state_file,
                )

                if args.ckpt_dir:
                    best_path = os.path.join(args.ckpt_dir, "best.pt")
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_mrr": best_mrr,
                        },
                        best_path,
                    )
                    print(f"Saved best checkpoint: {best_path}")

        train_log["epochs"].append(epoch_row)
        save_json(train_log, train_log_path)

        print(
            "Epoch {:04d} | Ave Loss {:.4f} | ent-rel-static {:.4f}-{:.4f}-{:.4f} | Best MRR {:.6f}".format(
                epoch,
                np.mean(losses),
                np.mean(losses_e),
                np.mean(losses_r),
                np.mean(losses_static),
                best_mrr,
            )
        )

    final_eval_ckpt = model_state_file
    if args.ckpt_dir:
        best_path = os.path.join(args.ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            final_eval_ckpt = best_path

    return test(
        model,
        train_list + valid_list,
        list(train_times) + list(valid_times),
        test_list,
        test_times,
        num_rels,
        num_nodes,
        use_cuda,
        all_ans_list_test,
        all_ans_list_r_test,
        final_eval_ckpt,
        static_graph,
        history_test_time_nogt,
        mode="test",
        args=args,
        hva_histories=hva_hist_train_valid,
        graph_cache=graph_cache,
        sequence_cache=sequence_cache,
        snap_map=all_snap_map,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TiRGN with optional end-to-end HVA")

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--run-analysis", action="store_true", default=False)
    parser.add_argument("--run-statistic", action="store_true", default=False)

    parser.add_argument("--dump-full-scores", action="store_true", default=False)
    parser.add_argument("--full-score-path", type=str, default="")
    parser.add_argument("--eval-mode", type=str, default="normal", choices=["normal", "dump_valid", "dump_test"])

    parser.add_argument("--multi-step", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--add-static-graph", action="store_true", default=False)
    parser.add_argument("--add-rel-word", action="store_true", default=False)
    parser.add_argument("--relation-evaluation", action="store_true", default=False)

    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--angle", type=int, default=10)

    parser.add_argument("--encoder", type=str, default="uvrgcn")
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action="store_true", default=False)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--opn", type=str, default="sub")

    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--self-loop", action="store_true", default=True)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--relation-prediction", action="store_true", default=False)
    parser.add_argument("--entity-prediction", action="store_true", default=False)
    parser.add_argument("--split_by_relation", action="store_true", default=False)

    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=20)

    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--feat-dropout", type=float, default=0.2)

    parser.add_argument("--train-history-len", type=int, default=10)
    parser.add_argument("--test-history-len", type=int, default=20)
    parser.add_argument("--dilate-len", type=int, default=1)

    parser.add_argument("--history-rate", type=float, default=0.3)
    parser.add_argument("--save", type=str, default="one")

    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--train-log-path", type=str, default="")

    parser.add_argument("--use-history-gate", action="store_true", default=False)
    parser.add_argument("--hva-topk", type=int, default=256)
    parser.add_argument("--hva-mode", type=str, default="dual_branch", choices=["exact_only", "dual_branch"])
    parser.add_argument("--hva-gamma-exact", type=float, default=0.005)
    parser.add_argument("--hva-gamma-near", type=float, default=0.08)
    parser.add_argument("--hva-stale-init", type=float, default=0.2)

    parser.add_argument("--graph-cache-size", type=int, default=24)
    parser.add_argument("--sparse-cache-size", type=int, default=24)

    args = parser.parse_args()
    print(args)
    run_experiment(args)
