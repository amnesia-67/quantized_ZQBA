import torch
import torch.nn.functional as F

def calc_kl_divergence(target_logits, surrogate_logits):
    """
    Computes behavioral distance via KL-Divergence.
    """
    target_probs = F.softmax(target_logits, dim=1)
    surrogate_log_probs = F.log_softmax(surrogate_logits, dim=1)
    # reduction='batchmean' is mathematically correct for KL Div in PyTorch
    kl_div = F.kl_div(surrogate_log_probs, target_probs, reduction='batchmean')
    return kl_div.item()

def calc_l2_distance(model_a, model_b):
    """
    Computes parametric L2-Norm distance across flattened weight vectors.
    """
    l2_dist = 0.0
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        if param_a.shape == param_b.shape:
            l2_dist += torch.norm(param_a.data - param_b.data, p=2).item()
    return l2_dist