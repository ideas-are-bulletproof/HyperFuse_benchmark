"""
HypeBoy embedder adapter  (Kim et al., ICLR'24 --
"HypeBoy: Generative Self-Supervised Representation Learning on Hypergraphs").

Faithful reproduction of the official repo's unsupervised pretraining (the
``HypeBoy`` routine in ``src.py``): a two-stage generative SSL --

  Stage 1  feature reconstruction   (300 epochs, prob_x=0.5, prob_e=0.2)
  Stage 2  hyperedge filling        (`ssl_epoch` epochs, prob_x=p_x, prob_e=p_e)

using the repo's own ``HyperEncoder`` (UniGCNII-style conv), ``HyperDecoder``,
``key_query_mapper`` heads, ``related_tool`` loss builder and ``augment_*``
functions (all vendored, unchanged).  The two stage-loops are re-expressed here
only so we can log progress every few epochs; every tensor operation inside them
is byte-for-byte the upstream code, and the encoder/decoder/heads/loss they call
are the vendored originals.

The node embedding is ``encoder(X, H, N, E)`` after loading the pretrained
weights -- exactly how the repo extracts representations for its linear-node /
linear-edge evaluation.  Output width is HypeBoy's own node_dim = 128.

Hyperparameters are HypeBoy's ``main.py`` argparse defaults
(epoch=200, p_x=0.4, p_e=0.9, lr=1e-3, weight_decay=1e-6).
"""

from __future__ import annotations

import copy

import numpy as np

from .. import config as C
from ._vendoring import vendor_path, require_torch, _GENERIC

# our dataset name -> HypeBoy's canonical name.  This only affects `related_tool`,
# which switches to a memory-safe *batched* loss for the two largest graphs
# (dblp_coauth, news); any other name uses the plain (non-batched) path, which
# is correct for the smaller datasets HypeBoy doesn't ship (zoo/mushroom/ntu).
CFG_KEY = {
    "cora_c": "cora_cite", "citeseer": "citeseer_cite", "pubmed": "pubmed_cite",
    "cora_a": "cora_ca", "dblp": "dblp_coauth", "20news": "news",
    "modelnet40": "modelnet_40", "ntu2012": "ntu2012", "mushroom": "mushroom",
    "zoo": "zoo",
}

# HypeBoy main.py defaults, for the record
DEFAULTS = dict(epoch1=300, epoch2=200, lr1=1e-3, lr2=1e-3, w_decay=1e-6,
                prob_x1=0.5, prob_e1=0.2, prob_x2=0.4, prob_e2=0.9,
                edge_dim=128, node_dim=128, num_layers=2, drop_p=0.5)


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    require_torch()
    import torch

    name = CFG_KEY.get(sample.name, sample.name)
    P = DEFAULTS
    logger.info(f"    [hypeboy] name={name} node_dim={P['node_dim']} "
                f"layers={P['num_layers']} epoch1={P['epoch1']} "
                f"epoch2={P['epoch2']} p_x={P['prob_x2']} p_e={P['prob_e2']}")

    with vendor_path("hypeboy", evict=_GENERIC + ("HNNs", "src")):
        from HNNs import HyperEncoder, HyperDecoder, key_query_mapper
        from src import fix_seed, related_tool, augment_feature, augment_edge

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        fix_seed(seed)

        X = torch.FloatTensor(sample.x_dense()).to(dev)
        H = torch.LongTensor(sample.hyperedge_index).to(dev)
        n_node = sample.num_nodes

        required_tools = related_tool(X, H, name, dev)

        encoder = HyperEncoder(in_dim=X.shape[1], edge_dim=P["node_dim"],
                               node_dim=P["node_dim"], num_layers=P["num_layers"],
                               drop_p=P["drop_p"], cached=False).to(dev)
        decoder = HyperDecoder(in_dim=P["node_dim"], edge_dim=X.shape[1],
                               node_dim=X.shape[1], drop_p=P["drop_p"],
                               num_layers=P["num_layers"], cached=False,
                               device=dev).to(dev)
        head1 = key_query_mapper(hidden_dim=P["node_dim"]).to(dev)
        head2 = key_query_mapper(hidden_dim=P["node_dim"]).to(dev)

        # ================= Stage 1: feature reconstruction =================
        # (verbatim from src.feature_reconstruction, + logging)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=P["lr1"], weight_decay=P["w_decay"])
        encoder.train(); decoder.train()
        totalX = X.to(dev)
        n_mask = int(n_node * P["prob_x1"])
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        edge_dict = required_tools.edge_dict
        for ep in range(P["epoch1"]):
            opt.zero_grad()
            masked_idx = list(np.random.choice(a=n_node, size=n_mask, replace=False))
            masked_idx.sort()
            curX = torch.clone(totalX)
            curX[masked_idx, :] = decoder.input_mask
            curE = augment_edge(edge_dict, P["prob_e1"], n_node, dev)
            n_edge = int(curE[1][-1]) + 1
            Z1 = encoder(curX, curE, n_node, n_edge)
            Z1[masked_idx, :] = decoder.embedding_mask
            Z2 = decoder(Z1, curE, n_node, n_edge)
            loss = torch.mean((1 - cos(totalX[masked_idx, :], Z2[masked_idx, :])))
            loss.backward()
            opt.step()
            if (ep + 1) % C.LOG_EVERY == 0 or ep == P["epoch1"] - 1:
                logger.info(f"    [hypeboy] stage1 recon {ep+1}/{P['epoch1']} "
                            f"| loss {loss.item():.4f}")
        encoder.load_state_dict(copy.deepcopy(encoder.state_dict()))

        # ================= Stage 2: hyperedge filling =====================
        # (verbatim from src.hyperedge_filling, head_type='head', + logging)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(head1.parameters())
                               + list(head2.parameters()),
                               lr=P["lr2"], weight_decay=P["w_decay"])
        head1.train(); head2.train(); encoder.train()
        np.random.seed(seed)
        fixed_feature = X.to(dev)
        for ep in range(P["epoch2"]):
            opt.zero_grad()
            curX = augment_feature(fixed_feature, P["prob_x2"], dev)
            curE = augment_edge(edge_dict, P["prob_e2"], n_node, dev)
            n_edge = int(curE[1][-1]) + 1
            Z = encoder(curX, curE, n_node, n_edge)
            loss = required_tools.return_loss(Z, head1, head2, "head")
            loss.backward()
            opt.step()
            if (ep + 1) % C.LOG_EVERY == 0 or ep == P["epoch2"] - 1:
                logger.info(f"    [hypeboy] stage2 fill  {ep+1}/{P['epoch2']} "
                            f"| loss {loss.item():.4f}")

        # ================= embedding readout ==============================
        encoder.eval()
        with torch.no_grad():
            E = int(torch.max(H[1]) + 1)
            Z = encoder(X, H, n_node, E)
        return Z.detach().cpu().numpy().astype(np.float32)
