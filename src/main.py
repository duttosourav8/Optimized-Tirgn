import argparse
import itertools
import os
import sys
import time
import pickle
import random
from collections import defaultdict

import dgl
import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn.modules.rnn

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import *
from rgcn.knowledge_graph import _read_triplets_as_list


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list, model_name, static_graph,
         time_list, history_time_nogt, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    dump_scores = []
    dump_triples = []

    idx = 0
    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device("cpu"))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint["epoch"]))
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    input_list = [snap for snap in history_list[-args.test_history_len:]]

    if args.multi_step:
        all_tail_seq = sp.load_npz(
            "../data/{}/history/tail_history_{}.npz".format(args.dataset, history_time_nogt)
        )
        all_rel_seq = sp.load_npz(
            "../data/{}/history/rel_history_{}.npz".format(args.dataset, history_time_nogt)
        )

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [
            build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
            for g in input_list
        ]

        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.cuda(args.gpu)

        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()

        if args.multi_step:
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)
        else:
            all_tail_seq = sp.load_npz(
                "../data/{}/history/tail_history_{}.npz".format(args.dataset, time_list[time_idx])
            )
            seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

            all_rel_seq = sp.load_npz(
                "../data/{}/history/rel_history_{}.npz".format(args.dataset, time_list[time_idx])
            )
            rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
            rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
            one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

        if use_cuda:
            one_hot_tail_seq = one_hot_tail_seq.cuda(args.gpu)
            one_hot_rel_seq = one_hot_rel_seq.cuda(args.gpu)

        test_triples, final_score, final_r_score = model.predict(
            history_glist, num_rels, static_graph, test_triples_input,
            one_hot_tail_seq, one_hot_rel_seq, use_cuda
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
            input_list.pop(0)
            input_list.append(test_snap)

        idx += 1

    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    if args.dump_full_scores:
        if args.full_score_path == "":
            raise ValueError("args.dump_full_scores=True but --full-score-path is empty")

        save_dir = os.path.dirname(args.full_score_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        all_scores = np.concatenate(dump_scores, axis=0)
        all_triples = np.concatenate(dump_triples, axis=0)
        np.savez_compressed(args.full_score_path, scores=all_scores, triples=all_triples)
        print("Saved full scores to:", args.full_score_path)
        print("Scores shape:", all_scores.shape)
        print("Triples shape:", all_triples.shape)

    return (
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r,
        hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r
    )


def run_experiment(args, history_len=None, n_layers=None, dropout=None,
                   n_bases=None, angle=None, history_rate=None):
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate

    mrr_raw = None
    mrr_filter = None
    mrr_raw_r = None
    mrr_filter_r = None
    hit_result_raw = None
    hit_result_filter = None
    hit_result_raw_r = None
    hit_result_filter_r = None

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
    print("num_times", num_times, "--------------", time_interval)

    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "gl_rate_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}".format(
        args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers,
        args.dilate_len, args.train_history_len, args.weight, args.discount,
        args.angle, args.dropout, args.input_dropout, args.hidden_dropout,
        args.feat_dropout, args.gpu, args.save
    )

    os.makedirs("../models", exist_ok=True)
    model_state_file = os.path.join("../models/", model_name)

    load_ckpt_path = model_state_file
    if hasattr(args, "resume_ckpt") and args.resume_ckpt and os.path.exists(args.resume_ckpt):
        load_ckpt_path = args.resume_ckpt

    print("Checkpoint used for eval/dump:", load_ckpt_path)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False)
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
        analysis=args.run_analysis
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    if args.add_static_graph:
        static_graph = build_sub_graph(
            len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    _start_epoch = 0
    best_mrr = 0.0

    if hasattr(args, "resume_ckpt") and args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        _start_epoch = ckpt.get("epoch", 0) + 1
        best_mrr = ckpt.get("best_mrr", 0.0)
        print(f"Resumed from epoch {_start_epoch}, best_mrr={best_mrr:.4f}")

    if args.eval_mode == "dump_valid":
        if not os.path.exists(load_ckpt_path):
            raise FileNotFoundError("Checkpoint not found for validation dump: {}".format(load_ckpt_path))
        if not args.dump_full_scores:
            raise ValueError("dump_valid requires --dump-full-scores")
        if args.full_score_path == "":
            raise ValueError("dump_valid requires --full-score-path")

        print("-------------- dumping validation full scores from best checkpoint ----------------")
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
            model,
            train_list,
            valid_list,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_valid,
            all_ans_list_r_valid,
            load_ckpt_path,
            static_graph,
            valid_times,
            history_val_time_nogt,
            mode="test"
        )

    elif args.eval_mode == "dump_test":
        if not os.path.exists(load_ckpt_path):
            raise FileNotFoundError("Checkpoint not found for test dump: {}".format(load_ckpt_path))
        if not args.dump_full_scores:
            raise ValueError("dump_test requires --dump-full-scores")
        if args.full_score_path == "":
            raise ValueError("dump_test requires --full-score-path")

        print("-------------- dumping test full scores from best checkpoint ----------------")
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
            model,
            train_list + valid_list,
            test_list,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            load_ckpt_path,
            static_graph,
            test_times,
            history_test_time_nogt,
            mode="test"
        )

    elif args.test:
        if not os.path.exists(load_ckpt_path):
            print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(load_ckpt_path))
        else:
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
                model,
                train_list + valid_list,
                test_list,
                num_rels,
                num_nodes,
                use_cuda,
                all_ans_list_test,
                all_ans_list_r_test,
                load_ckpt_path,
                static_graph,
                test_times,
                history_test_time_nogt,
                "test"
            )

    else:
        print("----------------------------------------start training----------------------------------------\n")

        for epoch in range(_start_epoch, args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0:
                    continue

                output = train_list[train_sample_num:train_sample_num + 1]

                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0:train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:train_sample_num]

                history_glist = [
                    build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                    for snap in input_list
                ]

                if use_cuda:
                    output = [torch.from_numpy(_).long().cuda(args.gpu) for _ in output]
                else:
                    output = [torch.from_numpy(_).long() for _ in output]

                histroy_data = output[0]
                inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
                histroy_data = torch.cat([histroy_data, inverse_histroy_data])
                histroy_data = histroy_data.cpu().numpy()

                all_tail_seq = sp.load_npz(
                    "../data/{}/history/tail_history_{}.npz".format(args.dataset, train_times[train_sample_num])
                )
                seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
                tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

                all_rel_seq = sp.load_npz(
                    "../data/{}/history/rel_history_{}.npz".format(args.dataset, train_times[train_sample_num])
                )
                rel_seq_idx = histroy_data[:, 0] * num_nodes + histroy_data[:, 2]
                rel_seq = torch.Tensor(all_rel_seq[rel_seq_idx].todense())
                one_hot_rel_seq = rel_seq.masked_fill(rel_seq != 0, 1)

                if use_cuda:
                    one_hot_tail_seq = one_hot_tail_seq.cuda(args.gpu)
                    one_hot_rel_seq = one_hot_rel_seq.cuda(args.gpu)

                loss_e, loss_r, loss_static = model.get_loss(
                    history_glist, output[0], static_graph,
                    one_hot_tail_seq, one_hot_rel_seq, use_cuda
                )
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            print(
                "Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} ".format(
                    epoch,
                    np.mean(losses),
                    np.mean(losses_e),
                    np.mean(losses_r),
                    np.mean(losses_static),
                    best_mrr,
                    model_name
                )
            )

            if args.ckpt_dir:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                latest_path = os.path.join(args.ckpt_dir, "latest.pt")
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_mrr": best_mrr
                    },
                    latest_path
                )

            if epoch and ((epoch + 1) % args.evaluate_every == 0):
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
                    model,
                    train_list,
                    valid_list,
                    num_rels,
                    num_nodes,
                    use_cuda,
                    all_ans_list_valid,
                    all_ans_list_r_valid,
                    model_state_file,
                    static_graph,
                    valid_times,
                    history_val_time_nogt,
                    mode="train"
                )

                if not args.relation_evaluation:
                    current_mrr = mrr_filter
                else:
                    current_mrr = mrr_filter_r

                if current_mrr > best_mrr:
                    best_mrr = current_mrr
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_mrr": best_mrr
                        },
                        model_state_file
                    )

                    if args.ckpt_dir:
                        best_path = os.path.join(args.ckpt_dir, "best.pt")
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "best_mrr": best_mrr
                            },
                            best_path
                        )
                        print(f"Saved best checkpoint: {best_path}")

        final_eval_ckpt = model_state_file
        if args.ckpt_dir:
            best_path = os.path.join(args.ckpt_dir, "best.pt")
            if os.path.exists(best_path):
                final_eval_ckpt = best_path

        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = test(
            model,
            train_list + valid_list,
            test_list,
            num_rels,
            num_nodes,
            use_cuda,
            all_ans_list_test,
            all_ans_list_r_test,
            final_eval_ckpt,
            static_graph,
            test_times,
            history_test_time_nogt,
            mode="test"
        )

    return (
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r,
        hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="TIRGN")

    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--batch-size", type=int, default=1, help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--test", action="store_true", default=False, help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action="store_true", default=False, help="print log info")
    parser.add_argument("--run-statistic", action="store_true", default=False, help="statistic the result")

    parser.add_argument("--dump-full-scores", action="store_true", default=False,
                        help="dump full entity score matrix and aligned triples")
    parser.add_argument("--full-score-path", type=str, default="",
                        help="path to save dumped scores, e.g. ../results/test_scores.npz")
    parser.add_argument("--eval-mode", type=str, default="normal",
                        choices=["normal", "dump_valid", "dump_test"],
                        help="normal training/testing, or dump validation/test scores from best checkpoint")

    parser.add_argument("--multi-step", action="store_true", default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action="store_true", default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action="store_true", default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action="store_true", default=False,
                        help="save model according to the relation evaluation")

    parser.add_argument("--weight", type=float, default=1, help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7, help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1, help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10, help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn", help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none", help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--skip-connect", action="store_true", default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub", help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--self-loop", action="store_true", default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action="store_true", default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action="store_true", default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action="store_true", default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action="store_true", default=False,
                        help="do relation prediction")

    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")

    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    parser.add_argument("--decoder", type=str, default="convtranse", help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

    parser.add_argument("--train-history-len", type=int, default=10, help="history length")
    parser.add_argument("--test-history-len", type=int, default=20, help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1, help="dilate history graph")

    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str,
                        default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500, help="number of triples generated")

    parser.add_argument("--history-rate", type=float, default=0.3, help="history rate")

    parser.add_argument("--save", type=str, default="one", help="number of save")

    parser.add_argument("--ckpt-dir", type=str, default="", help="directory to save latest.pt and best.pt")
    parser.add_argument("--resume-ckpt", type=str, default="",
                        help="explicit checkpoint path for resume or evaluation")

    args = parser.parse_args()
    print(args)

    if args.grid_search:
        out_log = "../results/{}.{}.gs".format(args.dataset, args.encoder + "-" + args.decoder + "-" + args.save)
        o_f = open(out_log, "w")
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(",")

        if args.tune == "" or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)

        if args.dataset == "ICEWS14s":
            hp_range_ = hp_range
        if args.dataset == "WIKI":
            hp_range_ = hp_range_WIKI
        if args.dataset == "YAGO":
            hp_range_ = hp_range_YAGO
        if args.dataset == "ICEWS18":
            hp_range_ = hp_range_ICEWS18
        if args.dataset == "ICEWS05-15":
            hp_range_ = hp_range_ICEWS05_15
        if args.dataset == "GDELT":
            hp_range_ = hp_range_GDELT

        grid = hp_range_[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range_[hp])
        grid = list(grid)

        print("* {} hyperparameter combinations to try".format(len(grid)))
        o_f.write("* {} hyperparameter combinations to try\n".format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, "a")

            if not (type(grid_entry) is list or type(grid_entry) is tuple):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)

            print("\n\n* Hyperparameter Set {}:".format(i))
            o_f.write("* Hyperparameter Set {}:\n".format(i))
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")

            args.test = False
            args.multi_step = False
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(
                args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5]
            )

            hits = [1, 3, 10]
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (filter): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filter) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw_rel): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw_rel) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (filter_rel): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filter_rel) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))

            args.test = True
            args.topk = 0
            args.multi_step = True
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(
                args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5]
            )

            o_f.write("No ground truth result:\n")
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (filter): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filter) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw_rel): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw_rel) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (filter_rel): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (filter_rel) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))

            o_f.close()

    else:
        run_experiment(args)

    sys.exit()