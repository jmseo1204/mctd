import torch
ckpt_path = "/mnt/c/Users/USER/Desktop/test_ogbench/mctd_repo/outputs/downloaded/jmseo1204-seoul-national-university/mctd_eval/en1ddvu7/model.ckpt"
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint Keys:", checkpoint.keys())
    if "hyper_parameters" in checkpoint:
        print("\n--- Hyperparameters ---")
        for k, v in checkpoint["hyper_parameters"].items():
            print(f"{k}: {v}")
    
    if "state_dict" in checkpoint:
        print("\n--- Model State Dict Size (First few) ---")
        for i, (k, v) in enumerate(checkpoint["state_dict"].items()):
            if i > 10: break
            print(f"{k}: {v.shape}")
            
except Exception as e:
    print(f"Error loading checkpoint: {e}")
