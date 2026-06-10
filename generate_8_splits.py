import torch
import os

def generate_8_splits():
    save_dir = "data_splits_8"
    os.makedirs(save_dir, exist_ok=True)
    
    # Target always gets the exact same 40,000 images to maintain a constant decision boundary
    target_indices = list(range(0, 40000))
    torch.save(target_indices, os.path.join(save_dir, "Target_Fixed_40k.pt"))
    
    # Overlap percentages to test: 100%, 85%, 70%, 55%, 40%, 25%, 10%, 0%
    overlaps = [100, 85, 70, 55, 40, 25, 10, 0]
    surrogate_size = 10000
    
    for overlap in overlaps:
        overlap_count = int(surrogate_size * (overlap / 100.0))
        disjoint_count = surrogate_size - overlap_count
        
        # Pull overlapping images from inside the target's pool
        overlap_indices = list(range(0, overlap_count))
        # Pull disjoint images from completely outside the target's pool
        disjoint_indices = list(range(40000, 40000 + disjoint_count))
        
        surrogate_indices = overlap_indices + disjoint_indices
        
        filename = f"Surrogate_{overlap}_Overlap.pt"
        torch.save(surrogate_indices, os.path.join(save_dir, filename))
        
    print(f"Generated 8 strict ablation splits in ./{save_dir}/")

if __name__ == "__main__":
    generate_8_splits()