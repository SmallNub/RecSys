import os
import torch

def get_file_size_gb(path):
    return os.path.getsize(path) / (1024 ** 3)

def strip_checkpoint(path):
    size_before = get_file_size_gb(path)
    print(f"Processing: {path} ({size_before:.2f} GB)...")
    
    try:
        # Load the checkpoint (on CPU to avoid GPU OOM)
        ckpt = torch.load(path, map_location="cpu")
        
        # Check if it has optimizer states to remove
        modified = False
        if "optimizer_states" in ckpt:
            del ckpt["optimizer_states"]
            modified = True
            print("  - Removed optimizer_states")
        if "lr_schedulers" in ckpt:
            del ckpt["lr_schedulers"]
            modified = True
            print("  - Removed lr_schedulers")
            
        if modified:
            # Save it back to the same path
            torch.save(ckpt, path)
            size_after = get_file_size_gb(path)
            space_saved = size_before - size_after
            print(f"  -> Done! New size: {size_after:.2f} GB (Saved {space_saved:.2f} GB)")
        else:
            print("  - No optimizer states found; already stripped.")
            
    except Exception as e:
        print(f"  - Error processing {path}: {e}")

def main():
    target_dir = "outputs/checkpoints"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return
        
    print(f"Scanning '{target_dir}' for checkpoints...")
    checkpoint_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_files.append(os.path.join(root, file))
                
    if not checkpoint_files:
        print("No .ckpt files found.")
        return
        
    print(f"Found {len(checkpoint_files)} checkpoint(s). Starting stripping process...")
    for path in checkpoint_files:
        strip_checkpoint(path)
        print("-" * 50)

if __name__ == "__main__":
    main()
