"""
TriCL embedder adapter  (Lee & Shin, AAAI'23).

Faithful reproduction of the official repo's unsupervised training:
``HyperEncoder`` + ``TriCL`` (vendored, unchanged) trained with the exact
per-dataset hyperparameters from the repo's own ``config.yaml`` and the exact
train step from its ``train.py`` (model_type='tricl', num_negs=None).  The
node embedding is ``model(features, hyperedge_index)[0]`` -- identical to how
``node_classification_eval`` reads it upstream.
"""

from __future__ import annotations

import os

import numpy as np

from .. import config as C
from ._vendoring import vendor_path, require_torch

# our dataset name -> TriCL config.yaml key
CFG_KEY = {
    "cora_c": "cora", "citeseer": "citeseer", "pubmed": "pubmed",
    "cora_a": "cora_coauthor", "dblp": "dblp_coauthor", "zoo": "zoo",
    "20news": "20newsW100", "mushroom": "Mushroom", "ntu2012": "NTU2012",
    "modelnet40": "ModelNet40",
}

DEFAULTS = "see vendor/tricl/config.yaml (loaded per-dataset at runtime)"


def _load_cfg(dataset):
    import yaml
    with open(os.path.join(C.VENDOR_ROOT, "tricl", "config.yaml")) as f:
        allcfg = yaml.safe_load(f)
    key = CFG_KEY.get(dataset)
    if key is None or key not in allcfg:
        raise KeyError(f"no TriCL config for dataset {dataset!r}")
    return allcfg[key], key


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    require_torch()
    import torch

    params, key = _load_cfg(sample.name)
    logger.info(f"    [tricl] cfg={key} hid={params['hid_dim']} "
                f"layers={params['num_layers']} epochs={params['epochs']} "
                f"lr={params['lr']}")

    with vendor_path("tricl"):
        from TriCL.models import HyperEncoder, TriCL
        from TriCL.utils import (drop_features, drop_incidence,
                                 valid_node_edge_mask, hyperedge_index_masking)

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        features = torch.FloatTensor(sample.x_dense()).to(dev)
        hyperedge_index = torch.LongTensor(sample.hyperedge_index).to(dev)
        num_nodes, num_edges = sample.num_nodes, sample.num_edges

        encoder = HyperEncoder(features.shape[1], params["hid_dim"],
                               params["hid_dim"], params["num_layers"])
        model = TriCL(encoder, params["proj_dim"]).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])

        # ---- exact train step from TriCL/train.py (model_type='tricl') ----
        def train_step():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            hei1 = drop_incidence(hyperedge_index, params["drop_incidence_rate"])
            hei2 = drop_incidence(hyperedge_index, params["drop_incidence_rate"])
            x1 = drop_features(features, params["drop_feature_rate"])
            x2 = drop_features(features, params["drop_feature_rate"])

            nm1, em1 = valid_node_edge_mask(hei1, num_nodes, num_edges)
            nm2, em2 = valid_node_edge_mask(hei2, num_nodes, num_edges)
            edge_mask = em1 & em2

            n1, e1 = model(x1, hei1, num_nodes, num_edges)
            n2, e2 = model(x2, hei2, num_nodes, num_edges)
            n1, n2 = model.node_projection(n1), model.node_projection(n2)
            e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

            loss_n = model.node_level_loss(n1, n2, params["tau_n"],
                                           batch_size=params["batch_size_1"], num_negs=None)
            loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params["tau_g"],
                                            batch_size=params["batch_size_1"], num_negs=None)
            mi1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, em1)
            mi2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, em2)
            loss_m1 = model.membership_level_loss(n1, e2[em2], mi2, params["tau_m"],
                                                  batch_size=params["batch_size_2"])
            loss_m2 = model.membership_level_loss(n2, e1[em1], mi1, params["tau_m"],
                                                  batch_size=params["batch_size_2"])
            loss_m = (loss_m1 + loss_m2) * 0.5
            loss = loss_n + params["w_g"] * loss_g + params["w_m"] * loss_m
            loss.backward()
            optimizer.step()
            return float(loss.item())

        total = params["epochs"]
        for epoch in range(1, total + 1):
            loss = train_step()
            if epoch % C.LOG_EVERY == 0 or epoch == total:
                logger.info(f"    [tricl] epoch {epoch}/{total} | loss {loss:.4f}")

        model.eval()
        with torch.no_grad():
            n, _ = model(features, hyperedge_index)
        return n.detach().cpu().numpy().astype(np.float32)
