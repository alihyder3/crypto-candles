from __future__ import annotations
import os, json, argparse, shutil, time, math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from src.phase1.models import LSTMForecaster

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM forecaster (PyTorch) with early stopping.")
    p.add_argument("--dataset", required=True, help=".npz from make_dataset.py")
    p.add_argument("--scaler", required=True, help="Path to scaler JSON to copy next to model")
    p.add_argument("--outdir", required=True, help="Output directory for model + history")
    # Model
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    # Train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    p.add_argument("--clip", type=float, default=1.0, help="Grad clip max norm")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def as_float32(x): return torch.as_tensor(x, dtype=torch.float32)

def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    set_seed(a.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load dataset
    data = np.load(a.dataset, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)   # [N, L, F]
    y_train = data["y_train"].astype(np.float32)   # [N, H]
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.float32)
    # Robust meta loader: works for dict, 0-D object array, or 1-D object array
    raw_meta = data["meta"]
    try:
        import numpy as _np
        if isinstance(raw_meta, dict):
            meta = raw_meta
        elif isinstance(raw_meta, _np.ndarray) and raw_meta.dtype == object:
            if raw_meta.shape == ():            # 0-D object array
                meta = raw_meta.item()
            elif raw_meta.ndim == 1 and len(raw_meta) >= 1:
                meta = raw_meta[0]              # already a dict
            else:
                meta = {}
        else:
            meta = {}
    except Exception:
        meta = {}


    n_features = X_train.shape[2]
    lookback   = X_train.shape[1]
    horizon    = y_train.shape[1]

    # Datasets / loaders
    train_ds = TensorDataset(as_float32(X_train), as_float32(y_train))
    val_ds   = TensorDataset(as_float32(X_val),   as_float32(y_val))
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=a.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # Model / opt / loss
    model = LSTMForecaster(
        n_features=n_features, horizon=horizon,
        hidden_size=a.hidden, num_layers=a.layers,
        dropout=a.dropout, bidirectional=a.bidirectional
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_epoch = -1
    history = {"train_loss": [], "val_loss": []}

    def evaluate():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
        return total / max(1, n)

    # Train loop
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if a.clip and a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = running / max(1, n_seen)
        val_loss = evaluate()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[E{epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "n_features": n_features,
                    "hidden_size": a.hidden,
                    "num_layers": a.layers,
                    "dropout": a.dropout,
                    "bidirectional": a.bidirectional,
                    "lookback": lookback,
                    "horizon": horizon,
                },
                "meta": meta,
            }, os.path.join(a.outdir, "model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= a.patience:
                print(f"[EARLY] No improvement for {a.patience} epochs. Stopping.")
                break

    dur = time.time() - t0
    # Save history
    hist_path = os.path.join(a.outdir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs_run": len(history["train_loss"]),
            "dataset": os.path.abspath(a.dataset),
            "scaler": "scaler.json",
            "runtime_sec": dur,
        }, f, indent=2)
    print(f"[OK] Saved training history -> {hist_path}")

    # Copy scaler next to model for inference pipeline
    dst_scaler = os.path.join(a.outdir, "scaler.json")
    shutil.copyfile(a.scaler, dst_scaler)
    print(f"[OK] Copied scaler -> {dst_scaler}")

    # Quick val batch prediction check (shape only)
    model.load_state_dict(torch.load(os.path.join(a.outdir, "model.pth"), map_location=device)["state_dict"])
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_dl))
        xb = xb.to(device)
        pred = model(xb)
        print(f"[CHECK] Pred shape: {tuple(pred.shape)}  (expected: (batch, {horizon}))")

if __name__ == "__main__":
    main()
