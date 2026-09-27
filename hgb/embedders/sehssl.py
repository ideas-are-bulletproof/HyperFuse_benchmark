"""
SE-HSSL embedder adapter  (structural-entropy hypergraph self-supervised learning).

Faithful reproduction of the official repo's ``main.py`` (model_type='hssl',
is_fair=False):

  * k-hop sample generation via the repo's own ``clique_expansion`` /
    ``search_k_hop_edge`` (vendored, unchanged) -- ``generate_sample`` is
    re-expressed here because it lives in the repo's script, not an importable
    module; the heavy lifting it calls is the vendored code.
  * encoder = ``HyperEncoder`` + ``TriCL`` (vendored tricl_encoder), trained
    with the CCA loss (``cca_loss``, vendored) at node & edge level plus the
    membership-level ``MILNCELoss`` (copied verbatim from main.py), using the
    per-dataset hyperparameters from the repo's ``config.yaml``.
  * embedding = ``model(features, hyperedge_index)[0]``.

Note: ``generate_sample``/``search_k_hop_edge`` materialise dense N×N and N×E
matrices exactly as upstream does, so very large datasets (pubmed, dblp) are
memory-heavy here just as they are upstream -- a warning is logged.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from .. import config as C
from ._vendoring import vendor_path, require_torch

CFG_KEY = {
    "cora_c": "cora", "citeseer": "citeseer", "pubmed": "pubmed",
    "cora_a": "cora_coauthor", "dblp": "dblp_coauthor", "zoo": "zoo",
    "20news": "20newsW100", "mushroom": "Mushroom", "ntu2012": "NTU2012",
    "modelnet40": "ModelNet40",
}
DEFAULTS = "see vendor/sehssl/config.yaml (loaded per-dataset at runtime)"


def _load_cfg(dataset):
    import yaml
    with open(os.path.join(C.VENDOR_ROOT, "sehssl", "config.yaml")) as f:
        allcfg = yaml.safe_load(f)
    key = CFG_KEY.get(dataset)
    if key is None or key not in allcfg:
        raise KeyError(f"no SE-HSSL config for dataset {dataset!r}")
    return allcfg[key], key


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    require_torch()
    import torch
    import torch.nn as nn

    params, key = _load_cfg(sample.name)
    if sample.num_nodes > 8000:
        logger.warning(f"    [sehssl] N={sample.num_nodes}: k-hop sample "
                       "generation builds dense N×N/N×E matrices (upstream "
                       "behaviour) -- this is memory/time heavy.")
    logger.info(f"    [sehssl] cfg={key} hid={params['hid_dim']} "
                f"K={params['K']} d={params['d']} n_epoch={params['n_epoch']}")

    with vendor_path("sehssl"):
        from tricl_encoder import HyperEncoder, TriCL
        from contrast_loss import cca_loss
        from utils import (drop_features, drop_incidence, valid_node_edge_mask,
                           search_k_hop_edge, clique_expansion)

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        params = dict(params)
        params["device"] = dev

        features = torch.FloatTensor(sample.x_dense()).to(dev)
        hyperedge_index = torch.LongTensor(sample.hyperedge_index).to(dev)
        num_nodes, num_edges = sample.num_nodes, sample.num_edges

        # ---- lightweight stand-in for the repo's `data` object ----
        class _Data:
            pass
        data = _Data()
        data.features = features
        data.hyperedge_index = hyperedge_index
        data.num_nodes = num_nodes
        data.num_edges = num_edges
        data.name = sample.name

        # ---- generate_sample (verbatim logic from main.py/sample_generator.py) ----
        clique_index = clique_expansion(hyperedge_index)
        neighbor_list = search_k_hop_edge(data, clique_index, hyperedge_index,
                                          "cpu", K=params["K"])
        np.random.seed(seed)
        sample_list = defaultdict(list)
        for v_i in range(len(neighbor_list)):
            for k in range(len(neighbor_list[v_i])):
                sample_list[v_i].append(
                    np.random.choice(neighbor_list[v_i][k].numpy(), params["d"]).tolist())
        hop_hypernode, hop_hyperedge = {}, {}
        k_hop_n = np.array(list(map(len, sample_list.values())))
        for k in range(params["K"] + 1):
            ids = np.where(k_hop_n == (k + 1))[0]
            if len(ids):
                hop_hypernode[k + 1] = ids
        for k in hop_hypernode.keys():
            hop_hyperedge[k] = torch.tensor([sample_list[i] for i in hop_hypernode[k]])

        # ---- MILNCELoss (verbatim from main.py) ----
        class MILNCELoss(nn.Module):
            def __init__(self, node_dim, edge_dim, d, tau, beta, batch_size, device, mean=True):
                super().__init__()
                self.d = d; self.tau = tau; self.mean = mean
                self.batch_size = batch_size
                self.beta = torch.tensor(beta).to(device)
                self.disc = nn.Bilinear(node_dim, edge_dim, 1).to(device)
                self.device = device

            def f(self, x, tau):
                return torch.exp(x / tau)

            def Listwise_loss(self, n, e, hop_hyperedge, hop_hypernode):
                score_list = defaultdict(None)
                for k in hop_hypernode.keys():
                    score_list[k] = torch.zeros((hop_hypernode[k].shape[0], k)).to(self.device)
                losses = []
                for k in hop_hypernode.keys():
                    num_samples = len(hop_hypernode[k])
                    num_batches = (num_samples - 1) // self.batch_size + 1
                    indices = torch.arange(0, num_samples)
                    for i in range(num_batches):
                        ids = indices[i * self.batch_size:(i + 1) * self.batch_size]
                        node_idx = hop_hypernode[k][ids]
                        anchor = n[node_idx, :]
                        anchor = torch.repeat_interleave(anchor.unsqueeze(1), self.d, dim=1)
                        for j in range(k):
                            contrast_edges_j = hop_hyperedge[k][ids, j]
                            contrast_obj = e[contrast_edges_j, :]
                            hop_score = self.f(torch.sigmoid(
                                self.disc(anchor, contrast_obj).squeeze()), self.tau).sum(dim=1)
                            score_list[k][ids, j] = hop_score
                for k in hop_hypernode.keys():
                    loss_k = torch.zeros(hop_hyperedge[k].shape[0]).to(self.device)
                    for j in range(k - 1):
                        loss_k += -torch.log(torch.min(
                            score_list[k][:, j] / score_list[k][:, j:].sum(dim=1), self.beta))
                    loss_k = loss_k / (k - 1)
                    losses.append(loss_k)
                return torch.cat(losses)

            def forward(self, n, e, hop_hyperedge, hop_hypernode):
                loss = self.Listwise_loss(n, e, hop_hyperedge, hop_hypernode)
                return loss.mean() if self.mean else loss.sum()

        encoder = HyperEncoder(features.shape[1], params["hid_dim"],
                               params["hid_dim"], params["num_layers"])
        model = TriCL(encoder, params["proj_dim"]).to(dev)
        contrast_model = MILNCELoss(params["hid_dim"], params["hid_dim"], params["d"],
                                    params["tau"], params["beta"], params["batch_size"],
                                    dev, mean=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])

        # ---- exact train step from main.py (model_type='hssl') ----
        def train_step():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            hei1 = drop_incidence(hyperedge_index, params["drop_incidence_rate_1"])
            hei2 = drop_incidence(hyperedge_index, params["drop_incidence_rate_2"])
            x1 = drop_features(features, params["drop_feature_rate_1"])
            x2 = drop_features(features, params["drop_feature_rate_2"])
            valid_node_edge_mask(hei1, num_nodes, num_edges)
            valid_node_edge_mask(hei2, num_nodes, num_edges)
            n1, e1 = model(x1, hei1, num_nodes, num_edges)
            n2, e2 = model(x2, hei2, num_nodes, num_edges)
            n, e = model(features, hyperedge_index, num_nodes, num_edges)
            loss_n = cca_loss(n1, n2, num_nodes, params["lambda_n"], dev)
            loss_g = cca_loss(e1, e2, num_edges, params["lambda_g"], dev)
            loss_m = contrast_model(n, e, hop_hyperedge, hop_hypernode)
            loss = loss_n + params["w_g"] * loss_g + params["w_m"] * loss_m
            loss.backward()
            optimizer.step()
            return float(loss.item())

        total = params["n_epoch"]
        for epoch in range(1, total + 1):
            loss = train_step()
            if epoch % C.LOG_EVERY == 0 or epoch == total:
                logger.info(f"    [sehssl] epoch {epoch}/{total} | loss {loss:.4f}")

        model.eval()
        with torch.no_grad():
            n, _ = model(features, hyperedge_index)
        return n.detach().cpu().numpy().astype(np.float32)
