"""
AllSet classifier adapter.

Runs the official AllSet-repo classifiers (vendored, unchanged) --
AllSetTransformer, HNHN, HGNN (via HCHA), HyperGCN, UniGCNII -- on top of a
precomputed node embedding used as the node-feature matrix.  The per-method
preprocessing, model construction and training loop are reproduced verbatim
from AllSet's ``train.py``, and the hyperparameters are AllSet's own argparse
defaults (epochs=500, lr=1e-3, wd=0, dropout=0.5, MLP_hidden=64,
All_num_layers=2, heads=1, HNHN alpha/beta=-1.5/-0.5, etc.).  PMA is set True
for AllSetTransformer and False+aggregate='add' for AllDeepSets, exactly as
AllSet's ``parse_method`` does.

For each label split the model is trained for ``epochs`` and the test accuracy
+ macro-F1 are taken at the epoch of best validation accuracy (AllSet's
best-val model-selection protocol).  Metrics are averaged over the split family
for a given (dataset, embedder, seed).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score

from .. import config as C
from ..embedders._vendoring import vendor_path, require_torch


def default_args(num_features: int, num_classes: int, method: str, device_cuda: int):
    """AllSet train.py argparse defaults, plus the method-specific flags that
    parse_method sets (PMA/aggregate for the AllSet variants)."""
    a = SimpleNamespace(
        method=method, epochs=500, dropout=0.5, lr=0.001, wd=0.0,
        All_num_layers=2, MLP_num_layers=2, MLP_hidden=64,
        Classifier_num_layers=2, Classifier_hidden=64,
        aggregate="mean", normtype="all_one", add_self_loop=True,
        normalization="ln", deepset_input_norm=True, GPR=True, LearnMask=True,
        num_features=num_features, num_classes=num_classes, feature_noise="1",
        exclude_self=False, PMA=False, heads=1, output_heads=1,
        HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=True,
        HCHA_symdegnorm=False, cuda=device_cuda,
        HyperGCN_mediators=False, HyperGCN_fast=False,
        UniGNN_use_norm=False, UniGNN_degV=0, UniGNN_degE=0,
    )
    if method == "AllSetTransformer":
        a.PMA = True
    elif method == "AllDeepSets":
        a.PMA = False
        a.aggregate = "add"
    return a


class SimpleData:
    """Minimal attribute container that behaves like the PyG `data` object the
    AllSet models read (they only use attribute access + a .to(device))."""
    def to(self, device):
        import torch
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def build_base_data(emb: np.ndarray, sample, device):
    import torch
    N = sample.num_nodes
    hi = sample.hyperedge_index  # [2, nnz] node ids / he ids (0..E-1)
    V = hi[0].astype(np.int64)
    E = (hi[1] + N).astype(np.int64)           # offset he ids into [N, N+E)
    # AllSet raw format: edge_index = [V|E ; E|V]
    row = np.concatenate([V, E])
    col = np.concatenate([E, V])
    edge_index = torch.as_tensor(np.stack([row, col]), dtype=torch.long)

    d = SimpleData()
    d.x = torch.as_tensor(emb, dtype=torch.float32)
    d.y = torch.as_tensor(sample.y, dtype=torch.long)
    d.n_x = torch.tensor([N])
    d.num_hyperedges = torch.tensor([sample.num_edges])
    d.edge_index = edge_index
    return d


def _preprocess(P, data, method, args, device):
    """Replicate AllSet train.py's per-method preprocessing. P is the vendored
    preprocessing module. Returns (data, extras) where extras carries He_dict
    or (V,E) when a method needs them."""
    import torch
    extras = {}
    if method in ("AllSetTransformer", "AllDeepSets"):
        data = P.ExtractV2E(data)
        if args.add_self_loop:
            data = P.Add_Self_Loops(data)
        if args.exclude_self:
            data = P.expand_edge_index(data)
        data = P.norm_contruction(data, option=args.normtype)
    elif method == "HyperGCN":
        data = P.ExtractV2E(data)
        extras["He_dict"] = P.get_HyperGCN_He_dict(data)
    elif method == "HNHN":
        data = P.ExtractV2E(data)
        if args.add_self_loop:
            data = P.Add_Self_Loops(data)
        H = P.ConstructH_HNHN(data)
        data = P.generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    elif method in ("HCHA", "HGNN"):
        data = P.ExtractV2E(data)
        if args.add_self_loop:
            data = P.Add_Self_Loops(data)
        data.edge_index[1] -= data.edge_index[1].min()
    elif method == "UniGCNII":
        import torch_sparse
        from torch_scatter import scatter
        data = P.ExtractV2E(data)
        if args.add_self_loop:
            data = P.Add_Self_Loops(data)
        data = P.ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        (row, col), _ = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float()
        degE = scatter(degV[V], E, dim=0, reduce="mean").pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV, args.UniGNN_degE = degV, degE
        extras["V"], extras["E"] = V, E
    return data, extras


def _build_model(M, method, data, args, extras, device):
    import torch
    if method == "AllSetTransformer":
        return M.SetGNN(args, data.norm) if args.LearnMask else M.SetGNN(args)
    if method == "AllDeepSets":
        return M.SetGNN(args, data.norm) if args.LearnMask else M.SetGNN(args)
    if method == "HyperGCN":
        return M.HyperGCN(V=data.x.shape[0], E=extras["He_dict"], X=data.x,
                          num_features=args.num_features,
                          num_layers=args.All_num_layers,
                          num_classses=args.num_classes, args=args)
    if method in ("HGNN", "HCHA"):
        return M.HCHA(args)
    if method == "HNHN":
        return M.HNHN(args)
    if method == "UniGCNII":
        V = extras["V"].to(device); E = extras["E"].to(device)
        return M.UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden,
                          nclass=args.num_classes, nlayer=args.All_num_layers,
                          nhead=args.heads, V=V, E=E)
    raise ValueError(method)


def _train_eval_one_split(M, model, data, method, args, split, device, logger, tag):
    """AllSet training loop for one split; return (test_acc, test_macro_f1)."""
    import torch
    import torch.nn.functional as F

    model.reset_parameters()
    if method == "UniGCNII":
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4),
        ], lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_idx = torch.as_tensor(split["train"], dtype=torch.long, device=device)
    val_idx = torch.as_tensor(split["valid"], dtype=torch.long, device=device)
    test_idx = torch.as_tensor(split["test"], dtype=torch.long, device=device)
    y = data.y

    best_val, best_test_acc, best_test_pred = -1.0, 0.0, None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = F.log_softmax(model(data), dim=1)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_idx] == y[val_idx]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()
                best_test_pred = pred[test_idx].detach().cpu().numpy()
        if (epoch + 1) % (C.LOG_EVERY * 5) == 0 or epoch == args.epochs - 1:
            logger.info(f"      [{tag}] epoch {epoch+1}/{args.epochs} | "
                        f"loss {loss.item():.4f} | best_val {best_val:.4f}")

    y_test = y[test_idx].detach().cpu().numpy()
    f1 = f1_score(y_test, best_test_pred, average="macro")
    return best_test_acc, float(f1)


def run_classifier(allset_method: str, emb: np.ndarray, sample, splits,
                   seed: int, device: str, logger, done=None, on_split=None) -> dict:
    """Train/evaluate one AllSet classifier on every split.

    As in AllSet's train.py, preprocessing + model construction happen once and
    the model is re-initialised (reset_parameters) for every split.  The RNG is
    re-seeded with ``seed`` before each split, so a split gives the same result
    whether the run was interrupted and resumed or not.

    done     : {split_id: (acc, f1)} already finished -> skipped
    on_split : callback(split_id, acc, f1) after each newly finished split
    """
    done = done or {}
    todo = [si for si in range(len(splits)) if si not in done]
    accs = {si: done[si][0] for si in done}
    f1s = {si: done[si][1] for si in done}

    if todo:
        require_torch()
        import torch
        from ..seeding import set_global_seed

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        device_cuda = 0 if dev.type == "cuda" else -1

        with vendor_path("allset"):
            import importlib
            M = importlib.import_module("models")
            P = importlib.import_module("preprocessing")

            set_global_seed(seed)
            args = default_args(int(emb.shape[1]), int(sample.num_classes),
                                allset_method, device_cuda)
            # HyperGCN's original AllSet implementation expects dname.
            args.dname = getattr(sample, "name", "")
            data = build_base_data(emb, sample, dev)
            data, extras = _preprocess(P, data, allset_method, args, dev)
            if data is None:
                raise RuntimeError("AllSet preprocessing rejected the hypergraph "
                                   "(num_hyperedges mismatch)")
            model = _build_model(M, allset_method, data, args, extras, dev)
            model = model.to(dev)
            data = data.to(dev)
            if allset_method == "UniGCNII":
                args.UniGNN_degV = args.UniGNN_degV.to(dev)
                args.UniGNN_degE = args.UniGNN_degE.to(dev)

            for si in todo:
                set_global_seed(seed)
                tag = f"{allset_method}[split{si}]"
                acc, f1 = _train_eval_one_split(M, model, data, allset_method, args,
                                                splits[si], dev, logger, tag)
                accs[si], f1s[si] = acc, f1
                if on_split is not None:
                    on_split(si, acc, f1)
                logger.info(f"    [{allset_method}] split {si+1}/{len(splits)} "
                            f"acc={acc:.4f} macroF1={f1:.4f}")

    order = sorted(accs)
    acc_list = [float(accs[i]) for i in order]
    f1_list = [float(f1s[i]) for i in order]
    return {"accuracy": float(np.mean(acc_list)), "macro_f1": float(np.mean(f1_list)),
            "split_accuracy": acc_list, "split_macro_f1": f1_list}
