"""
Reproduce server-side _evaluate_global_gmm locally on the houses checkpoint.
Bypasses the notebook entirely — calls the same train.py API path that
generated test_metrics_global.json on Apr 29. If this script yields the
server numbers (18.04 ≈ 0.883, ha- ≈ 1.000), the notebook has a subtle bug.
If it doesn't, the divergence is environmental (PyTorch/cuDNN, GPU
floating-point) and we need a different fix.
"""
import json
from pathlib import Path

import torch

from src.contrastive.train import (
    _evaluate_global_gmm,
    _load_splits_and_metadata,
    _prepare_dataset_tensors,
    _resolve_selected_features,
    _resolve_train_eval_ids,
)
from src.contrastive.models import build_contrastive_model


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RESULTS = DATA / "contrastive_g5_supcon_v2_ae_houses" / "results"

ENCODER_TRAIN = [
    "10.01.2026", "24.01.2026", "7.02.2026",
    "7.03.2026", "15.03.2026", "5.04.2026", "11.04.2026",
]
GMM_EVAL = ["21.02.2026", "18.04.2026", "ha-25.04.2026", "hb-26.04.2026"]


def main():
    # cfg snapshot matching the houses run
    cfg = json.load(open(REPO / "configs" / "contrastive" / "encoder.json"))
    cfg["cnn_channel_mode"] = "dual"
    cfg["include_std_features"] = True
    selected = _resolve_selected_features(cfg)

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(DATA, "./prepared")
    train_ids, val_ids, test_ids, _, _ = _resolve_train_eval_ids(
        splits_data, metadata_by_id, ENCODER_TRAIN, GMM_EVAL
    )
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    data = _prepare_dataset_tensors(
        train_ids, val_ids, test_ids,
        prepared_base, metadata_by_id,
        selected_features=selected,
        scaler_type=cfg.get("scaler", "standard"),
        dual=(cfg["cnn_channel_mode"] == "dual"),
    )
    print(f"train_patches: {data['train_patches'].shape}, "
          f"NoFlow={(data['train_labels']==0).sum()}, "
          f"Flow={(data['train_labels']==1).sum()}")
    print(f"test_patches:  {data['test_patches'].shape}")

    ckpt = torch.load(RESULTS / "contrastive_model.pt", map_location="cpu", weights_only=False)
    hp = ckpt["hp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_contrastive_model(
        encoder_type=hp["encoder_type"],
        input_height=ckpt["input_height"],
        input_width=ckpt["input_width"],
        channels=hp["cnn_channels"],
        kernel_size=hp["cnn_kernel_size"],
        latent_dim=hp["latent_dim"],
        projection_dim=hp["projection_dim"],
        dropout=hp["dropout"],
        in_channels=ckpt["n_in_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"device={device}, latent_dim={hp['latent_dim']}")

    out = _evaluate_global_gmm(
        model, data, device,
        gmm_n_components=cfg.get("gmm_n_components", 1),
        gmm_covariance_type=cfg.get("gmm_covariance_type", "full"),
    )
    print("\n--- LOCAL _evaluate_global_gmm result ---")
    for k, v in out.items():
        if k == "overall":
            continue
        print(f"  {k:18s}: AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  "
              f"NoFlow={v['n_noflow']} Flow={v['n_flow']}")
    if "overall" in out:
        print(f"  overall mean AUC = {out['overall']['mean_roc_auc']:.4f}, "
              f"gmm_fit_n_noflow = {out['overall']['gmm_fit_n_noflow']}")

    print("\n--- SERVER test_metrics_global.json ---")
    srv = json.load(open(RESULTS / "test_metrics_global.json"))
    for k, v in srv.items():
        if k == "overall":
            continue
        print(f"  {k:18s}: AUC={v['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
