import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import TimeConvTransE, TimeConvTransR
from src.history_validity_gate import (
    HistoryValidityAdapter,
    build_topk_candidate_ids,
    build_topk_history_features_dual,
    scatter_topk_back,
)


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(
                self.h_dim,
                self.h_dim,
                self.num_rels,
                self.num_bases,
                activation=act,
                dropout=self.dropout,
                self_loop=self.self_loop,
                skip_connect=sc,
                rel_emb=self.rel_emb,
            )
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata["id"].squeeze()
            g.ndata["h"] = init_ent_emb[node_id]
            _, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop("h")
        else:
            if self.features is not None:
                g.ndata["id"] = self.features
            node_id = g.ndata["id"].squeeze()
            g.ndata["h"] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop("h")


class RecurrentRGCN(nn.Module):
    def __init__(
        self,
        decoder_name,
        encoder_name,
        num_ents,
        num_rels,
        num_static_rels,
        num_words,
        num_times,
        time_interval,
        h_dim,
        opn,
        history_rate,
        sequence_len,
        num_bases=-1,
        num_basis=-1,
        num_hidden_layers=1,
        dropout=0,
        self_loop=False,
        skip_connect=False,
        layer_norm=False,
        input_dropout=0,
        hidden_dropout=0,
        feat_dropout=0,
        aggregation="cat",
        weight=1,
        discount=0,
        angle=0,
        use_static=False,
        entity_prediction=False,
        relation_prediction=False,
        use_cuda=False,
        gpu=0,
        analysis=False,
        use_history_gate=False,
        hva_topk=256,
        hva_mode="dual_branch",
        hva_gamma_exact=0.005,
        hva_gamma_near=0.08,
        hva_stale_init=0.2,
    ):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.use_cuda = None

        self.use_history_gate = use_history_gate
        self.hva_topk = hva_topk
        self.hva_mode = hva_mode
        self.hva_gamma_exact = hva_gamma_exact
        self.hva_gamma_near = hva_gamma_near
        self.hva_stale_init = hva_stale_init

        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                self.h_dim,
                self.h_dim,
                self.num_static_rels * 2,
                num_bases,
                activation=F.rrelu,
                dropout=dropout,
                self_loop=False,
                skip_connect=False,
            )
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(
            num_ents,
            h_dim,
            h_dim,
            num_rels * 2,
            num_bases,
            num_basis,
            num_hidden_layers,
            dropout,
            self_loop,
            skip_connect,
            encoder_name,
            self.opn,
            self.emb_rel,
            use_cuda,
            analysis,
        )

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain("relu"))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain("relu"))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

        if self.use_history_gate:
            self.history_validity_adapter = HistoryValidityAdapter(
                num_relations=self.num_rels * 2,
                mode=self.hva_mode,
                gamma_exact=self.hva_gamma_exact,
                gamma_near=self.hva_gamma_near,
                stale_init=self.hva_stale_init,
            )
            print(
                f"[HVA] enabled | mode={self.hva_mode} | topk={self.hva_topk} "
                f"| gamma_exact={self.hva_gamma_exact} | gamma_near={self.hva_gamma_near} "
                f"| stale_init={self.hva_stale_init}"
            )
        else:
            self.history_validity_adapter = None

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata["h"] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop("h")[: self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []
        device = self.emb_rel.device

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim, device=device, dtype=torch.float32)

            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]: span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)

        return history_embs, static_emb, self.h_0, gate_list, degree_list

    def _apply_history_validity_adapter(self, entity_log_scores, all_triples, hva_histories):
        if (not self.use_history_gate) or (self.history_validity_adapter is None) or (hva_histories is None):
            return entity_log_scores

        gold_ids = all_triples[:, 2]
        rel_ids = all_triples[:, 1]

        with torch.no_grad():
            candidate_ids = build_topk_candidate_ids(entity_log_scores, gold_ids, self.hva_topk)
            seen_sr, dt_sr, freq_sr, seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro = build_topk_history_features_dual(
                query_triples=all_triples,
                candidate_ids=candidate_ids,
                sr_hist=hva_histories["sr"],
                so_hist=hva_histories["so"],
                ro_hist=hva_histories["ro"],
                device=entity_log_scores.device,
                mode=self.hva_mode,
            )

        base_scores_topk = torch.gather(entity_log_scores, 1, candidate_ids)

        adjusted_topk_scores, _ = self.history_validity_adapter(
            base_scores_topk,
            rel_ids,
            seen_sr, dt_sr, freq_sr,
            seen_so, dt_so, freq_so,
            seen_ro, dt_ro, freq_ro,
        )

        adjusted_scores = scatter_topk_back(entity_log_scores, candidate_ids, adjusted_topk_scores)
        return adjusted_scores

    def predict(
        self,
        test_graph,
        num_rels,
        static_graph,
        test_triplets,
        entity_history_vocabulary,
        rel_history_vocabulary,
        use_cuda,
        hva_histories=None,
    ):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)

            score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score = torch.log(score)
            score = self._apply_history_validity_adapter(score, all_triples, hva_histories)

            return all_triples, score, score_rel

    def get_loss(
        self,
        glist,
        triples,
        static_graph,
        entity_history_vocabulary,
        rel_history_vocabulary,
        use_cuda,
        hva_histories=None,
    ):
        self.use_cuda = use_cuda
        device = self.emb_rel.device
        loss_ent = torch.zeros(1, device=device)
        loss_rel = torch.zeros(1, device=device)
        loss_static = torch.zeros(1, device=device)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples]).to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
            scores_en = torch.log(score_en)

            if self.use_history_gate:
                scores_en = self._apply_history_validity_adapter(scores_en, all_triples, hva_histories)
                loss_ent += F.cross_entropy(scores_en, all_triples[:, 2])
            else:
                loss_ent += F.nll_loss(scores_en, all_triples[:, 2])

        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        return F.softmax(scores_ob, dim=1)

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        global_index = history_vocabulary.float()
        if self.use_cuda:
            global_index = global_index.to(self.gpu, non_blocking=True)
        else:
            global_index = global_index.cpu()
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        return F.softmax(score_global, dim=1)

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        return F.softmax(scores_re, dim=1)

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        global_index = history_vocabulary.float()
        if self.use_cuda:
            global_index = global_index.to(self.gpu, non_blocking=True)
        else:
            global_index = global_index.cpu()
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        return F.softmax(score_global, dim=1)
