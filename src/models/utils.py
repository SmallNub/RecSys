import os
import json
import fsspec
import hydra
import torch
from omegaconf import OmegaConf


def load_model_from_path(run_directory: str, protocol: str = "file"):
    """
    Finds the latest checkpoint, instantiates the model architecture via Hydra,
    and loads the trained weights into memory.

    Args:
        run_directory (str): The path to the model's root run folder.
        protocol (str): The filesystem protocol (e.g., 'file', 's3', 'gcs').

    Returns:
        model: The instantiated PyTorch Lightning model with loaded weights.
        cfg: The OmegaConf configuration associated with the model.
    """
    fs = fsspec.filesystem(protocol)
    target_dir = run_directory
    snapshot_path = os.path.join(run_directory, "checkpoint_manager_snapshot.json")

    # 1. Locate the correct sub-directory (Mirrors get_top_cfg logic)
    if fs.exists(snapshot_path):
        with fs.open(snapshot_path, "r") as f:
            snapshot_data = json.load(f)
        latest_dir_name = snapshot_data["latest_checkpoint_result"][
            "checkpoint_dir_name"
        ]
        target_dir = os.path.join(run_directory, latest_dir_name)
    else:
        try:
            subdirs = [
                os.path.basename(x.rstrip("/"))
                for x in fs.glob(os.path.join(run_directory, "checkpoint_*"))
            ]
            if subdirs:
                target_dir = os.path.join(run_directory, sorted(subdirs)[-1])
        except Exception as e:
            print(f"[WARNING] Could not search subdirectories: {e}")

    cfg_path = os.path.join(target_dir, "config.yaml")
    ckpt_path = os.path.join(target_dir, "checkpoint.ckpt")

    if not fs.exists(cfg_path) or not fs.exists(ckpt_path):
        raise FileNotFoundError(
            f"Required config or checkpoint missing in {target_dir}"
        )

    # 2. Load the configuration
    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)

    print(f"[INFO] Instantiating model architecture from config...")
    # 3. Instantiate the model shell via Hydra
    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg

    print(f"[INFO] Loading checkpoint weights from {ckpt_path}...")
    # 4. Extract and load the state dict weights into the shell
    with fs.open(ckpt_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)

        # PyTorch Lightning saves model weights under the 'state_dict' key
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    return model, cfg
