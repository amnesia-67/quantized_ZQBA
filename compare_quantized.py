import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.quantization import resnet18 as qresnet18
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from tqdm import tqdm

'''
Author  : Anish Subash
Purpose : Quantized-vs-quantized transfer. PGD (L-inf) is crafted on each fake-quant
          INT8 ResNet-18 surrogate and transferred to the fake-quant INT8 ResNet-34
          target across the 8 dataset-overlap splits. Emits a single results PDF
          (per-overlap table + ASR-vs-overlap curve). Optionally overlays the
          FP32-target ASR baseline for a direct Q->Q vs Q->FP32 comparison.

          Standalone: no import from the trainer, drop next to your model dir and run.
          transforms match the paper's CIFAR pipeline (ToTensor only, inputs in [0,1]).
'''

CIFAR_EXAMPLE = (torch.randn(1, 3, 32, 32),)


# ------------------------- model builders / loaders -------------------------
def build_qat_resnet34(backend="fbgemm", num_classes=10):
    m = models.resnet34()
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.train()
    return prepare_qat_fx(m, get_default_qat_qconfig_mapping(backend), CIFAR_EXAMPLE)


def load_target(path, device, backend="fbgemm", mode="fakequant"):
    m = build_qat_resnet34(backend=backend)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    if mode == "int8":
        m = convert_fx(m)          # true integer kernels -> CPU only
        device = torch.device("cpu")
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, device


def load_surrogate(path, device, backend="fbgemm"):
    # Rebuild the exact prepared-QAT graph used in train_ablation.py (surrogate_qat role)
    m = qresnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.train()
    m.fuse_model()
    m.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)   # only the input needs grad for PGD
    return m


# ------------------------------- attack ------------------------------------
def pgd(model, x, y, eps, alpha, steps):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1).detach()
    return x_adv


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


def evaluate(target, surrogate, loader, dev_s, dev_t, eps, alpha, steps):
    total = clean_correct = attacked = success = 0
    for x, y in tqdm(loader, desc="  eval", leave=False):
        yt = y.to(dev_t)
        clean_pred = predict(target, x.to(dev_t))
        mask = clean_pred == yt
        total += y.size(0)
        clean_correct += mask.sum().item()
        if mask.any():
            m_cpu = mask.cpu()
            xs = x[m_cpu].to(dev_s)
            ys = y[m_cpu].to(dev_s)               # true labels drive the surrogate PGD
            x_adv = pgd(surrogate, xs, ys, eps, alpha, steps)
            adv_pred = predict(target, x_adv.to(dev_t))
            attacked += ys.size(0)
            success += (adv_pred != ys.to(dev_t)).sum().item()   # target flipped off correct label
    return {
        "clean_acc": 100.0 * clean_correct / max(total, 1),
        "asr": 100.0 * success / max(attacked, 1),
        "attacked": attacked,
    }


# ------------------------------- reporting ----------------------------------
def build_pdf(results, overlaps, out_path, target_path, eps, alpha, steps, fp32_asr=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    xs = sorted(overlaps)
    asr = [results[o]["asr"] for o in xs]

    with PdfPages(out_path) as pdf:
        fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(8.5, 11),
                                         gridspec_kw={"height_ratios": [1, 1.4]})
        # --- table ---
        ax_t.axis("off")
        header = ["Overlap %", "Target Clean Acc", "ASR (Q->Q)"]
        if fp32_asr:
            header.append("ASR (Q->FP32)")
        rows = [header]
        for o in xs:
            row = [str(o), f"{results[o]['clean_acc']:.2f}", f"{results[o]['asr']:.2f}"]
            if fp32_asr:
                row.append(f"{fp32_asr.get(o, float('nan')):.2f}")
            rows.append(row)
        tbl = ax_t.table(cellText=rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
        span = max(asr) - min(asr)
        ax_t.set_title(
            "Quantized-vs-Quantized Transfer (ZK-ML ablation)\n"
            f"PGD L-inf  eps={eps:.4f}  alpha={alpha:.4f}  steps={steps}\n"
            f"target={os.path.basename(target_path)}   flatline span={span:.2f}%",
            fontsize=11, pad=18)

        # --- curve ---
        ax_p.plot(xs, asr, marker="o", linewidth=2, label="INT8 target (Q->Q)")
        if fp32_asr:
            ax_p.plot(xs, [fp32_asr.get(o, float("nan")) for o in xs],
                      marker="s", linestyle="--", label="FP32 target (Q->FP32)")
        for o, a in zip(xs, asr):
            ax_p.annotate(f"{a:.1f}", (o, a), textcoords="offset points", xytext=(0, 7), fontsize=8)
        ax_p.set_xlabel("Surrogate Data Overlap with Target (%)")
        ax_p.set_ylabel("Attack Success Rate (ASR %)")
        ax_p.set_title("Impact of Dataset Overlap on Transferability")
        ax_p.grid(True, alpha=0.3)
        ax_p.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def parse_kv_list(s):
    # "100:60.41,0:53.86" -> {100: 60.41, 0: 53.86}
    out = {}
    if not s:
        return None
    for pair in s.split(","):
        k, v = pair.split(":")
        out[int(k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Quantized-vs-quantized PGD transfer sweep -> results PDF")
    ap.add_argument("--target", required=True, help="Prepared fake-quant target checkpoint from train_quant_target.py")
    ap.add_argument("--surrogate-dir", default="models/ablation_study")
    ap.add_argument("--surrogate-template", default="Surrogate_{}.pth")
    ap.add_argument("--overlaps", type=int, nargs="+", default=[100, 85, 70, 55, 40, 25, 10, 0])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "x86", "qnnpack"])
    ap.add_argument("--target-mode", choices=["fakequant", "int8"], default="fakequant",
                    help="fakequant = STE numerics on GPU (fast); int8 = true integer kernels on CPU")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=8.0 / 255.0)
    ap.add_argument("--alpha", type=float, default=2.0 / 255.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--fp32-asr", type=str, default=None,
                    help='Optional overlay, e.g. "100:60.41,85:59.2,...,0:53.86"')
    ap.add_argument("--out", default="results_quant_vs_quant.pdf")
    args = ap.parse_args()

    dev_s = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)

    target, dev_t = load_target(args.target, dev_s, backend=args.backend, mode=args.target_mode)
    print(f"[TARGET] {args.target} | mode={args.target_mode} | dev={dev_t}")

    results = {}
    for o in args.overlaps:
        spath = os.path.join(args.surrogate_dir, args.surrogate_template.format(o))
        surrogate = load_surrogate(spath, dev_s, backend=args.backend)
        m = evaluate(target, surrogate, loader, dev_s, dev_t, args.eps, args.alpha, args.steps)
        results[o] = m
        print(f"overlap={o:3d}%   clean={m['clean_acc']:.2f}   ASR={m['asr']:.2f}   (n={m['attacked']})")
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    build_pdf(results, args.overlaps, args.out, args.target,
              args.eps, args.alpha, args.steps, fp32_asr=parse_kv_list(args.fp32_asr))
    span = max(r["asr"] for r in results.values()) - min(r["asr"] for r in results.values())
    print("=" * 60)
    print(f"[SUCCESS] {args.out}   |   flatline span = {span:.2f}% ASR")
    print("=" * 60)


if __name__ == "__main__":
    main()