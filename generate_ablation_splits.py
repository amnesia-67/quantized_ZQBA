import torch
import os

'''
Author  : Anish Subash
Date    : June 1st, 2026
'''

def generate_splits():
    save_dir = "data_splits"
    os.makedirs(save_dir, exist_ok=True)
    
    # Total CIFAR-10 Training Images = 50,000
    
    # ---------------------------------------------------------
    # Scenario A: 90/20 (10% Overlap)
    # Target: 45k, Surrogate: 10k. Overlap: 5k (10% of total 50k)
    # ---------------------------------------------------------
    torch.save(list(range(0, 45000)), os.path.join(save_dir, "Target_A_90.pt"))
    torch.save(list(range(40000, 50000)), os.path.join(save_dir, "Surrogate_A_20.pt")) # Overlaps [40k-45k]
    
    # ---------------------------------------------------------
    # Scenario B: 80/30 (Smaller Overlap)
    # Target: 40k, Surrogate: 15k. Overlap: 5k
    # ---------------------------------------------------------
    torch.save(list(range(0, 40000)), os.path.join(save_dir, "Target_B_80.pt"))
    torch.save(list(range(35000, 50000)), os.path.join(save_dir, "Surrogate_B_30_Small.pt")) # Overlaps [35k-40k]
    
    # ---------------------------------------------------------
    # Scenario C: 80/30 (Larger Overlap)
    # Target: 40k, Surrogate: 15k. Overlap: 15k (Entire surrogate is inside target)
    # ---------------------------------------------------------
    torch.save(list(range(0, 40000)), os.path.join(save_dir, "Target_C_80.pt"))
    torch.save(list(range(0, 15000)), os.path.join(save_dir, "Surrogate_C_30_Large.pt")) # Overlaps [0-15k]
    
    # ---------------------------------------------------------
    # Scenario D: 50/50 (Completely Disjoint)
    # Target: 25k, Surrogate: 25k. Overlap: 0
    # ---------------------------------------------------------
    torch.save(list(range(0, 25000)), os.path.join(save_dir, "Target_D_50.pt"))
    torch.save(list(range(25000, 50000)), os.path.join(save_dir, "Surrogate_D_50.pt")) # Overlaps None

    print("All deterministic dataset splits generated in ./data_splits/")

if __name__ == "__main__":
    generate_splits()