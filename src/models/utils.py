import os
import json
import fsspec
import hydra
import torch
from omegaconf import OmegaConf


def load_underlying_model(run_directory: str, protocol: str = "file"):
    """
    Loads the checkpoint, extracts the core neural network model,
    and discards the PyTorch Lightning wrapper.

    Args:
        run_directory (str): Path to the model's root run folder.
        protocol (str): Filesystem protocol ('file', 's3', etc.).

    Returns:
        underlying_model: The raw PyTorch nn.Module (e.g., SASRec).
    """
    fs = fsspec.filesystem(protocol)
    target_dir = run_directory
    snapshot_path = os.path.join(run_directory, "checkpoint_manager_snapshot.json")

    # 1. Locate the correct sub-directory
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

    # 2. Load configuration
    with fs.open(cfg_path) as f:
        cfg = OmegaConf.load(f)

    # 3. Instantiate the Lightning wrapper shell
    # We do this because the checkpoint weights are saved with a "net." prefix
    lit_module = hydra.utils.instantiate(cfg["model"])

    # 4. Load the weights into the wrapper
    with fs.open(ckpt_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        lit_module.load_state_dict(state_dict)

    # 5. Extract only the raw underlying network
    model = lit_module.net

    return model
