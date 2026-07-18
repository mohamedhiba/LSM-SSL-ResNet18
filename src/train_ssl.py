"""Training loops for self-supervised raster patch pretraining."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.ssl_contrastive import nt_xent_loss
from src.ssl_cross_channel import (
    DEFAULT_MASKED_CHANNEL_INDEX,
    apply_channel_mask,
    cross_channel_loss,
)
from src.ssl_jigsaw import topk_accuracy
from src.ssl_masked_recon import generate_block_mask, masked_reconstruction_loss
from src.ssl_rotation import rotation_accuracy


def _batch_to_tensor(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch


def _batch_to_views(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Contrastive dataloader batches must contain view1 and view2.")
    return batch[0], batch[1]


def _batch_to_jigsaw(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Jigsaw dataloader batches must contain X_jigsaw and y_perm.")
    return batch[0], batch[1]


def _batch_to_rotation(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Rotation dataloader batches must contain X_rot and y_rot.")
    return batch[0], batch[1]


def train_masked_reconstruction_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mask_ratio: float = 0.5,
    block_size: int = 4,
    grad_clip_norm: float | None = 5.0,
) -> float:
    """Train masked reconstruction for one epoch."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        X = _batch_to_tensor(batch).to(device, non_blocking=True)
        terrain, ctx_mask = X[:, :-1], X[:, -1:]
        spatial_mask = generate_block_mask(
            batch_size=X.shape[0],
            height=X.shape[2],
            width=X.shape[3],
            mask_ratio=mask_ratio,
            block_size=block_size,
            device=X.device,
        )
        # Spatially mask terrain only; keep the valid-context mask channel visible
        # so the encoder always knows which pixels are padded.
        X_masked = torch.cat([terrain * (1.0 - spatial_mask), ctx_mask], dim=1)
        optimizer.zero_grad(set_to_none=True)
        X_recon = model(X_masked)
        # Reconstruct terrain over spatially-masked AND valid pixels only.
        loss = masked_reconstruction_loss(X_recon, terrain, spatial_mask * ctx_mask)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.item()) * X.shape[0]
        total_samples += X.shape[0]
    return total_loss / total_samples


@torch.no_grad()
def evaluate_masked_reconstruction(
    model,
    dataloader,
    device,
    mask_ratio: float = 0.5,
    block_size: int = 4,
) -> float:
    """Evaluate masked reconstruction loss."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        X = _batch_to_tensor(batch).to(device, non_blocking=True)
        terrain, ctx_mask = X[:, :-1], X[:, -1:]
        spatial_mask = generate_block_mask(
            batch_size=X.shape[0],
            height=X.shape[2],
            width=X.shape[3],
            mask_ratio=mask_ratio,
            block_size=block_size,
            device=X.device,
        )
        X_masked = torch.cat([terrain * (1.0 - spatial_mask), ctx_mask], dim=1)
        X_recon = model(X_masked)
        loss = masked_reconstruction_loss(X_recon, terrain, spatial_mask * ctx_mask)
        total_loss += float(loss.item()) * X.shape[0]
        total_samples += X.shape[0]
    return total_loss / total_samples


def _save_ssl_checkpoints(
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    channel_means,
    channel_stds,
    full_model_path: Path,
    encoder_path: Path,
) -> None:
    full_model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "config": config,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
    }
    torch.save(payload, full_model_path)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        encoder_path,
    )


def train_masked_reconstruction_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    mask_ratio: float = 0.5,
    block_size: int = 4,
    grad_clip_norm: float | None = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Train masked reconstruction model with early stopping by val loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_masked_reconstruction_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mask_ratio=mask_ratio,
            block_size=block_size,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = evaluate_masked_reconstruction(
            model,
            val_loader,
            device,
            mask_ratio=mask_ratio,
            block_size=block_size,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            _save_ssl_checkpoints(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                config,
                channel_means,
                channel_stds,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
    }


def train_contrastive_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    temperature: float = 0.2,
    grad_clip_norm: float | None = 5.0,
) -> float:
    """Train contrastive SSL for one epoch."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        view1, view2 = _batch_to_views(batch)
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.item()) * view1.shape[0]
        total_samples += view1.shape[0]
    return total_loss / total_samples


@torch.no_grad()
def evaluate_contrastive(
    model,
    dataloader,
    device,
    temperature: float = 0.2,
) -> float:
    """Evaluate contrastive NT-Xent loss."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        view1, view2 = _batch_to_views(batch)
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        total_loss += float(loss.item()) * view1.shape[0]
        total_samples += view1.shape[0]
    return total_loss / total_samples


def train_contrastive_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    temperature: float = 0.2,
    batch_size: int = 128,
    grad_clip_norm: float | None = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Train contrastive model with early stopping by validation NT-Xent loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_contrastive_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            temperature=temperature,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = evaluate_contrastive(
            model,
            val_loader,
            device,
            temperature=temperature,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "temperature": float(temperature),
                "batch_size": int(batch_size),
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            _save_ssl_checkpoints(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                config,
                channel_means,
                channel_stds,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
    }


def train_jigsaw_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = 5.0,
) -> dict[str, float]:
    """Train Jigsaw SSL for one epoch."""

    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    for batch in dataloader:
        X_jigsaw, y_perm = _batch_to_jigsaw(batch)
        X_jigsaw = X_jigsaw.to(device, non_blocking=True)
        y_perm = y_perm.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_jigsaw)
        loss = criterion(logits, y_perm)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = X_jigsaw.shape[0]
        acc = topk_accuracy(logits.detach(), y_perm.detach(), topk=(1, 5))
        total_loss += float(loss.item()) * batch_size
        total_top1 += acc[1] * batch_size
        total_top5 += acc[5] * batch_size
        total_samples += batch_size
    return {
        "loss": total_loss / total_samples,
        "acc_top1": total_top1 / total_samples,
        "acc_top5": total_top5 / total_samples,
    }


@torch.no_grad()
def evaluate_jigsaw(
    model,
    dataloader,
    criterion,
    device,
    return_predictions: bool = False,
) -> dict[str, object]:
    """Evaluate Jigsaw SSL loss and top-k accuracy."""

    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    y_true_parts = []
    y_pred_parts = []
    for batch in dataloader:
        X_jigsaw, y_perm = _batch_to_jigsaw(batch)
        X_jigsaw = X_jigsaw.to(device, non_blocking=True)
        y_perm = y_perm.to(device, non_blocking=True)
        logits = model(X_jigsaw)
        loss = criterion(logits, y_perm)
        batch_size = X_jigsaw.shape[0]
        acc = topk_accuracy(logits, y_perm, topk=(1, 5))
        total_loss += float(loss.item()) * batch_size
        total_top1 += acc[1] * batch_size
        total_top5 += acc[5] * batch_size
        total_samples += batch_size
        if return_predictions:
            y_true_parts.append(y_perm.detach().cpu())
            y_pred_parts.append(torch.argmax(logits.detach().cpu(), dim=1))

    result = {
        "loss": total_loss / total_samples,
        "acc_top1": total_top1 / total_samples,
        "acc_top5": total_top5 / total_samples,
    }
    if return_predictions:
        result["y_true"] = torch.cat(y_true_parts).numpy()
        result["y_pred"] = torch.cat(y_pred_parts).numpy()
    return result


def _save_jigsaw_checkpoints(
    model,
    optimizer,
    epoch: int,
    train_result: dict[str, float],
    val_result: dict[str, float],
    config: dict,
    channel_means,
    channel_stds,
    permutation_bank_path: str | Path,
    full_model_path: Path,
    encoder_path: Path,
) -> None:
    full_model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_result["loss"],
        "train_loss": train_result["loss"],
        "val_acc_top1": val_result["acc_top1"],
        "val_acc_top5": val_result["acc_top5"],
        "config": config,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "permutation_bank_path": str(permutation_bank_path),
    }
    torch.save(payload, full_model_path)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_result["loss"],
            "val_acc_top1": val_result["acc_top1"],
            "val_acc_top5": val_result["acc_top5"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "permutation_bank_path": str(permutation_bank_path),
        },
        encoder_path,
    )


def train_jigsaw_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    permutation_bank_path: str | Path,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
    n_permutation_classes: int = 100,
    grad_clip_norm: float | None = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Train Jigsaw SSL model with early stopping by validation loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    best_val_acc_top1 = 0.0
    best_val_acc_top5 = 0.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_result = train_jigsaw_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_result = evaluate_jigsaw(model, val_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_acc_top1": train_result["acc_top1"],
                "val_acc_top1": val_result["acc_top1"],
                "train_acc_top5": train_result["acc_top5"],
                "val_acc_top5": val_result["acc_top5"],
                "learning_rate": learning_rate,
                "batch_size": int(batch_size),
                "n_permutation_classes": int(n_permutation_classes),
            }
        )
        print(
            "epoch "
            f"{epoch:03d}: train_loss={train_result['loss']:.6f}, "
            f"val_loss={val_result['loss']:.6f}, "
            f"val_top1={val_result['acc_top1']:.4f}, "
            f"val_top5={val_result['acc_top5']:.4f}",
            flush=True,
        )
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_train_loss = train_result["loss"]
            best_val_acc_top1 = val_result["acc_top1"]
            best_val_acc_top5 = val_result["acc_top5"]
            best_epoch = epoch
            _save_jigsaw_checkpoints(
                model,
                optimizer,
                epoch,
                train_result,
                val_result,
                config,
                channel_means,
                channel_stds,
                permutation_bank_path,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "val_acc_top1": rows[-1]["val_acc_top1"],
            "val_acc_top5": rows[-1]["val_acc_top5"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "permutation_bank_path": str(permutation_bank_path),
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "best_val_acc_top1": best_val_acc_top1,
        "best_val_acc_top5": best_val_acc_top5,
    }


def train_rotation_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = 5.0,
) -> dict[str, float]:
    """Train rotation prediction SSL for one epoch."""

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for batch in dataloader:
        X_rot, y_rot = _batch_to_rotation(batch)
        X_rot = X_rot.to(device, non_blocking=True)
        y_rot = y_rot.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_rot)
        loss = criterion(logits, y_rot)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = X_rot.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_acc += rotation_accuracy(logits.detach(), y_rot.detach()) * batch_size
        total_samples += batch_size
    return {"loss": total_loss / total_samples, "acc_top1": total_acc / total_samples}


@torch.no_grad()
def evaluate_rotation(
    model,
    dataloader,
    criterion,
    device,
    return_predictions: bool = False,
) -> dict[str, object]:
    """Evaluate rotation prediction SSL loss and accuracy."""

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    y_true_parts = []
    y_pred_parts = []
    for batch in dataloader:
        X_rot, y_rot = _batch_to_rotation(batch)
        X_rot = X_rot.to(device, non_blocking=True)
        y_rot = y_rot.to(device, non_blocking=True)
        logits = model(X_rot)
        loss = criterion(logits, y_rot)
        batch_size = X_rot.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_acc += rotation_accuracy(logits, y_rot) * batch_size
        total_samples += batch_size
        if return_predictions:
            y_true_parts.append(y_rot.detach().cpu())
            y_pred_parts.append(torch.argmax(logits.detach().cpu(), dim=1))
    result = {"loss": total_loss / total_samples, "acc_top1": total_acc / total_samples}
    if return_predictions:
        result["y_true"] = torch.cat(y_true_parts).numpy()
        result["y_pred"] = torch.cat(y_pred_parts).numpy()
    return result


def _save_rotation_checkpoints(
    model,
    optimizer,
    epoch: int,
    train_result: dict[str, float],
    val_result: dict[str, float],
    config: dict,
    channel_means,
    channel_stds,
    full_model_path: Path,
    encoder_path: Path,
) -> None:
    full_model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_result["loss"],
        "train_loss": train_result["loss"],
        "val_acc_top1": val_result["acc_top1"],
        "config": config,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
    }
    torch.save(payload, full_model_path)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_result["loss"],
            "val_acc_top1": val_result["acc_top1"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        encoder_path,
    )


def train_rotation_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
    n_rotation_classes: int = 4,
    grad_clip_norm: float | None = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Train rotation prediction SSL model with early stopping by val loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    best_val_acc_top1 = 0.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_result = train_rotation_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_result = evaluate_rotation(model, val_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_acc_top1": train_result["acc_top1"],
                "val_acc_top1": val_result["acc_top1"],
                "learning_rate": learning_rate,
                "batch_size": int(batch_size),
                "n_rotation_classes": int(n_rotation_classes),
            }
        )
        print(
            "epoch "
            f"{epoch:03d}: train_loss={train_result['loss']:.6f}, "
            f"val_loss={val_result['loss']:.6f}, "
            f"val_top1={val_result['acc_top1']:.4f}",
            flush=True,
        )
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_train_loss = train_result["loss"]
            best_val_acc_top1 = val_result["acc_top1"]
            best_epoch = epoch
            _save_rotation_checkpoints(
                model,
                optimizer,
                epoch,
                train_result,
                val_result,
                config,
                channel_means,
                channel_stds,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "val_acc_top1": rows[-1]["val_acc_top1"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "best_val_acc_top1": best_val_acc_top1,
    }


def train_cross_channel_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    masked_channel_index: int = DEFAULT_MASKED_CHANNEL_INDEX,
    grad_clip_norm: float | None = 5.0,
    mask_channel_present: bool = True,
) -> float:
    """Train cross-channel masking for one epoch."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        X = _batch_to_tensor(batch).to(device, non_blocking=True)
        X_masked, target = apply_channel_mask(X, masked_channel_index, mask_channel_present)
        optimizer.zero_grad(set_to_none=True)
        X_pred = model(X_masked)
        loss = cross_channel_loss(X_pred, target)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.item()) * X.shape[0]
        total_samples += X.shape[0]
    return total_loss / total_samples


@torch.no_grad()
def evaluate_cross_channel(
    model,
    dataloader,
    device,
    masked_channel_index: int = DEFAULT_MASKED_CHANNEL_INDEX,
    mask_channel_present: bool = True,
) -> float:
    """Evaluate cross-channel masking loss."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        X = _batch_to_tensor(batch).to(device, non_blocking=True)
        X_masked, target = apply_channel_mask(X, masked_channel_index, mask_channel_present)
        X_pred = model(X_masked)
        loss = cross_channel_loss(X_pred, target)
        total_loss += float(loss.item()) * X.shape[0]
        total_samples += X.shape[0]
    return total_loss / total_samples


def train_cross_channel_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    masked_channel_index: int = DEFAULT_MASKED_CHANNEL_INDEX,
    grad_clip_norm: float | None = 5.0,
    mask_channel_present: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Train cross-channel masking model with early stopping by val loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_cross_channel_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            masked_channel_index=masked_channel_index,
            grad_clip_norm=grad_clip_norm,
            mask_channel_present=mask_channel_present,
        )
        val_loss = evaluate_cross_channel(
            model,
            val_loader,
            device,
            masked_channel_index=masked_channel_index,
            mask_channel_present=mask_channel_present,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "masked_channel_index": int(masked_channel_index),
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            _save_ssl_checkpoints(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                config,
                channel_means,
                channel_stds,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
    }


def train_strip_jigsaw_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = 5.0,
) -> dict[str, float]:
    """Train strip jigsaw SSL for one epoch."""

    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    for batch in dataloader:
        X_strip, y_perm = _batch_to_jigsaw(batch)
        X_strip = X_strip.to(device, non_blocking=True)
        y_perm = y_perm.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_strip)
        loss = criterion(logits, y_perm)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = X_strip.shape[0]
        acc = topk_accuracy(logits.detach(), y_perm.detach(), topk=(1, 5))
        total_loss += float(loss.item()) * batch_size
        total_top1 += acc[1] * batch_size
        total_top5 += acc[5] * batch_size
        total_samples += batch_size
    return {
        "loss": total_loss / total_samples,
        "acc_top1": total_top1 / total_samples,
        "acc_top5": total_top5 / total_samples,
    }


@torch.no_grad()
def evaluate_strip_jigsaw(
    model,
    dataloader,
    criterion,
    device,
    return_predictions: bool = False,
) -> dict[str, object]:
    """Evaluate strip jigsaw SSL loss and top-k accuracy."""

    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0
    y_true_parts = []
    y_pred_parts = []
    for batch in dataloader:
        X_strip, y_perm = _batch_to_jigsaw(batch)
        X_strip = X_strip.to(device, non_blocking=True)
        y_perm = y_perm.to(device, non_blocking=True)
        logits = model(X_strip)
        loss = criterion(logits, y_perm)
        batch_size = X_strip.shape[0]
        acc = topk_accuracy(logits, y_perm, topk=(1, 5))
        total_loss += float(loss.item()) * batch_size
        total_top1 += acc[1] * batch_size
        total_top5 += acc[5] * batch_size
        total_samples += batch_size
        if return_predictions:
            y_true_parts.append(y_perm.detach().cpu())
            y_pred_parts.append(torch.argmax(logits.detach().cpu(), dim=1))

    result = {
        "loss": total_loss / total_samples,
        "acc_top1": total_top1 / total_samples,
        "acc_top5": total_top5 / total_samples,
    }
    if return_predictions:
        result["y_true"] = torch.cat(y_true_parts).numpy()
        result["y_pred"] = torch.cat(y_pred_parts).numpy()
    return result


def _save_strip_jigsaw_checkpoints(
    model,
    optimizer,
    epoch: int,
    train_result: dict[str, float],
    val_result: dict[str, float],
    config: dict,
    channel_means,
    channel_stds,
    permutation_bank_path: str | Path,
    full_model_path: Path,
    encoder_path: Path,
) -> None:
    full_model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_result["loss"],
        "train_loss": train_result["loss"],
        "val_acc_top1": val_result["acc_top1"],
        "val_acc_top5": val_result["acc_top5"],
        "config": config,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "permutation_bank_path": str(permutation_bank_path),
    }
    torch.save(payload, full_model_path)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_result["loss"],
            "val_acc_top1": val_result["acc_top1"],
            "val_acc_top5": val_result["acc_top5"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "permutation_bank_path": str(permutation_bank_path),
        },
        encoder_path,
    )


def train_strip_jigsaw_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    full_model_best_path: str | Path,
    encoder_best_path: str | Path,
    last_checkpoint_path: str | Path,
    config: dict,
    channel_means,
    channel_stds,
    permutation_bank_path: str | Path,
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
    n_permutation_classes: int = 6,
    grad_clip_norm: float | None = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Train strip jigsaw SSL model with early stopping by validation loss."""

    full_model_best_path = Path(full_model_best_path).resolve()
    encoder_best_path = Path(encoder_best_path).resolve()
    last_checkpoint_path = Path(last_checkpoint_path).resolve()
    rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = float("inf")
    best_val_acc_top1 = 0.0
    best_val_acc_top5 = 0.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        train_result = train_strip_jigsaw_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_result = evaluate_strip_jigsaw(model, val_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_acc_top1": train_result["acc_top1"],
                "val_acc_top1": val_result["acc_top1"],
                "train_acc_top5": train_result["acc_top5"],
                "val_acc_top5": val_result["acc_top5"],
                "learning_rate": learning_rate,
                "batch_size": int(batch_size),
                "n_permutation_classes": int(n_permutation_classes),
            }
        )
        print(
            "epoch "
            f"{epoch:03d}: train_loss={train_result['loss']:.6f}, "
            f"val_loss={val_result['loss']:.6f}, "
            f"val_top1={val_result['acc_top1']:.4f}, "
            f"val_top5={val_result['acc_top5']:.4f}",
            flush=True,
        )
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_train_loss = train_result["loss"]
            best_val_acc_top1 = val_result["acc_top1"]
            best_val_acc_top5 = val_result["acc_top5"]
            best_epoch = epoch
            _save_strip_jigsaw_checkpoints(
                model,
                optimizer,
                epoch,
                train_result,
                val_result,
                config,
                channel_means,
                channel_stds,
                permutation_bank_path,
                full_model_best_path,
                encoder_best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": rows[-1]["epoch"],
            "val_loss": rows[-1]["val_loss"],
            "train_loss": rows[-1]["train_loss"],
            "val_acc_top1": rows[-1]["val_acc_top1"],
            "val_acc_top5": rows[-1]["val_acc_top5"],
            "config": config,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "permutation_bank_path": str(permutation_bank_path),
        },
        last_checkpoint_path,
    )
    return pd.DataFrame(rows), {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "best_val_acc_top1": best_val_acc_top1,
        "best_val_acc_top5": best_val_acc_top5,
    }
