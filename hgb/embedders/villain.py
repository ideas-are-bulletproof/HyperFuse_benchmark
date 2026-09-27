"""
VilLain embedder adapter  (structure-only self-supervised hypergraph embedding).

Faithful reproduction of the official repo's canonical run (main.py + run.sh):
for each ``num_labels`` in {2..8} the ``model`` (vendored, unchanged) is trained
with the argparse defaults (dim=128, lr=1e-2, num_step=4, num_step_gen=100,
epochs=5000 with the repo's post-1000-epoch early stopping), the node embedding
is read via ``get_node_embeds()``, and the per-``nl`` embeddings are merged with
PCA to ``dim`` exactly as ``emb_concat.py`` does.  VilLain uses ONLY the
hypergraph structure (no node features) -- an intentionally different signal
from the feature-aware methods.

The one deviation from upstream is initialising ``cnt_wait=0`` before the loop
(upstream references it before assignment on the first post-1000-epoch check,
a latent bug that would raise on some runs); behaviour on the normal path is
unchanged.  Seeding uses the harness seed rather than the hardcoded 2023 so the
method varies across the benchmark's seeds.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .. import config as C
from ._vendoring import vendor_path, require_torch

DEFAULTS = dict(dim=C.EMB_DIM, lr=1e-2, num_step=4, num_step_gen=100,
                epochs=5000, num_labels_sweep=[2, 3, 4, 5, 6, 7, 8],
                early_stop_patience=20, early_stop_after=1000)


def _train_one(model_module, V_idx, E_idx, V, E, num_labels, logger, seed):
    import torch
    import torch.optim as optim

    dim = DEFAULTS["dim"]
    num_subspace = math.ceil(dim / num_labels)
    m = model_module.model(V_idx, E_idx, V, E, num_subspace, num_labels,
                           DEFAULTS["num_step"], DEFAULTS["num_step_gen"])
    m = m.to(V_idx.device)
    optimizer = optim.AdamW(m.parameters(), lr=DEFAULTS["lr"], weight_decay=0)

    best_loss, best_state = 1e10, None
    pre_loss, patience, cnt_wait = 1e10, DEFAULTS["early_stop_patience"], 0
    total = DEFAULTS["epochs"]
    for epoch in range(1, total + 1):
        m.train()
        loss_local, loss_global = m()
        loss = loss_local + loss_global
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()

        if epoch % C.LOG_EVERY == 0 or epoch == total:
            logger.info(f"    [villain nl={num_labels}] epoch {epoch}/{total} | "
                        f"local {loss_local.item():.4f} global {loss_global.item():.4f}")

        if epoch <= DEFAULTS["early_stop_after"]:
            continue
        if loss.item() < best_loss:
            m.eval()
            best_loss = loss.item()
            best_state = copy.deepcopy(m.state_dict())
        diff = abs(loss.item() - pre_loss) / abs(pre_loss)
        cnt_wait = cnt_wait + 1 if diff < 0.002 else 0
        if cnt_wait == patience:
            break
        pre_loss = loss.item()

    if best_state is not None:
        m.load_state_dict(best_state)
    return m.get_node_embeds().detach().cpu().numpy()


def embed(sample, seed: int, device: str, logger) -> np.ndarray:
    require_torch()
    import torch
    from sklearn.decomposition import PCA

    dim = DEFAULTS["dim"]
    with vendor_path("villain") as _:
        import importlib
        model_module = importlib.import_module("model")  # vendor/villain/model.py

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        V_idx = torch.LongTensor(sample.hyperedge_index[0]).to(dev)
        E_idx = torch.LongTensor(sample.hyperedge_index[1]).to(dev)
        V = int(V_idx.max()) + 1
        E = int(E_idx.max()) + 1

        per_nl = []
        for nl in DEFAULTS["num_labels_sweep"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            emb = _train_one(model_module, V_idx, E_idx, V, E, nl, logger, seed)
            per_nl.append(np.asarray(emb)[:, :dim])

    y = np.concatenate(per_nl, axis=1)
    n_comp = min(dim, y.shape[0], y.shape[1])
    y = PCA(n_components=n_comp).fit_transform(y)
    if y.shape[1] < dim:  # pad tiny graphs to a stable width
        y = np.concatenate([y, np.zeros((y.shape[0], dim - y.shape[1]), np.float32)], 1)
    return y.astype(np.float32)
