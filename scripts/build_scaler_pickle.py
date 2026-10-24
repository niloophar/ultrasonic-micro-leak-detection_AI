"""
Re-build input_scaler.pkl for an existing contrastive run that was trained
before train.py started persisting the scaler. Mirrors the exact fit logic
from src.contrastive.train._scale_patches (StandardScaler on train-NoFlow only)
and from _resolve_train_eval_ids (so --encoder-train-series matches what the
original training run actually saw).

Usage examples (on the server, from repo root):

  # Default splits.json train (back-compat with very early runs):
  python scripts/build_scaler_pickle.py --output-root ./contrastive_g5_clf

  # Cross-House Phase 1 / Eval-B style (saison-divers, edges held out):
  python scripts/build_scaler_pickle.py \
      --output-root ./contrastive_g5_supcon_v2_ae_houses \
      --encoder-train-series 10.01.2026 24.01.2026 7.02.2026 \
                              7.03.2026 15.03.2026 5.04.2026 11.04.2026 \
      --cnn-channel-mode dual --include-std-features
"""
import argparse
import json
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.contrastive.train import (
    _ids_for_series,
    _load_splits_and_metadata,
    _prepare_patches,
    _resolve_selected_features,
    _resolve_train_eval_ids,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True,
                    help="e.g. ./contrastive_g5_supcon_v2_ae_houses")
    ap.add_argument("--data-root", default="./prepared")
    ap.add_argument("--config", default="configs/contrastive/encoder.json")
    ap.add_argument("--encoder-train-series", nargs="+", default=None,
                    help="Override splits.json train. Mirrors the same flag "
                         "passed to `python -m src.contrastive.main --run train`.")
    ap.add_argument("--cnn-channel-mode", choices=("dual", "single"), default=None)
    ap.add_argument("--include-std-features", action="store_true",
                    help="Pull the __std twins of selected_features (matches "
                         "the same flag in main.py).")
    ap.add_argument("--selected-features", nargs="+", default=None,
                    help="Override cfg selected_features (rarely needed).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data"
    results_dir = data_path / args.output_root.lstrip("./") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(repo_root / args.config))
    if args.cnn_channel_mode is not None:
        cfg["cnn_channel_mode"] = args.cnn_channel_mode
    if args.include_std_features:
        cfg["include_std_features"] = True
    if args.selected_features is not None:
        cfg["selected_features"] = list(args.selected_features)

    selected = _resolve_selected_features(cfg)
    scaler_type = cfg.get("scaler", "standard")
    cnn_channel_mode = cfg.get("cnn_channel_mode", "dual")

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(
        data_path, args.data_root
    )

    # Mirror the same train/val resolution train.py uses for this very run.
    train_ids, val_ids, _, train_resolved, _ = _resolve_train_eval_ids(
        splits_data, metadata_by_id,
        encoder_train_series=args.encoder_train_series,
        gmm_eval_series=None,
    )
    print(f"Encoder train series resolved: {train_resolved}")
    print(f"Train segment ids: {len(train_ids)}, val: {len(val_ids)}")

    patches, labels, _series, feat_names = _prepare_patches(
        train_ids, prepared_base, metadata_by_id, selected
    )
    print(f"Train patches: {patches.shape}, NoFlow={(labels == 0).sum()}, "
          f"Flow={(labels == 1).sum()}")

    noflow = patches[labels == 0]
    if noflow.size == 0:
        raise RuntimeError("No NoFlow training patches to fit scaler on.")
    n, n_feat, w = noflow.shape
    flat = noflow.transpose(0, 2, 1).reshape(-1, n_feat)
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    scaler.fit(flat)

    n_stats = 2 if cfg.get("include_std_features", False) else 1
    if cnn_channel_mode == "dual" and n_feat % 2 == 0:
        input_height = n_feat // (2 * n_stats)
        n_in_channels = 2 * n_stats
    else:
        input_height = n_feat
        n_in_channels = 1

    blob = {
        "scaler": scaler,
        "scaler_type": scaler_type,
        "feature_names": feat_names,
        "n_in_channels": n_in_channels,
        "input_height": input_height,
        "input_width": w,
        "cnn_channel_mode": cnn_channel_mode,
        "include_std_features": cfg.get("include_std_features", False),
    }

    out_path = results_dir / "input_scaler.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(blob, f)
    print(f"Wrote {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
