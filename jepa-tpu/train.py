"""Train the tiny I-JEPA on MNIST (CPU), then a linear probe on frozen target-encoder features.

  python3 train.py --epochs 3 --out ckpt.pt
"""
import argparse, copy, time
import numpy as np
import torch
import torch.nn.functional as F
from model import Cfg, Encoder, Predictor


def load_mnist(path):
    d = np.load(path)
    f = lambda x: torch.from_numpy(np.pad(x, ((0, 0), (2, 2), (2, 2)))).float().div(255).sub(0.1307).div(0.3081)[:, None]
    return f(d["x_train"]), torch.from_numpy(d["y_train"]).long(), f(d["x_test"]), torch.from_numpy(d["y_test"]).long()


def masks(B, g, gen):
    """One rectangular target block per sample (same size across the batch); context = rest."""
    h, w = int(torch.randint(2, 5, (1,), generator=gen)), int(torch.randint(2, 5, (1,), generator=gen))
    tgt, keep = [], []
    for _ in range(B):
        r = int(torch.randint(0, g - h + 1, (1,), generator=gen)); c = int(torch.randint(0, g - w + 1, (1,), generator=gen))
        t = torch.tensor([(r + i) * g + c + j for i in range(h) for j in range(w)])
        m = torch.ones(g * g, dtype=torch.bool); m[t] = False
        tgt.append(t); keep.append(m.nonzero().squeeze(1))
    return torch.stack(keep), torch.stack(tgt)


@torch.no_grad()
def features(enc, x, bs=500):
    enc.eval()
    return torch.cat([enc(x[i:i + bs]).mean(1) for i in range(0, len(x), bs)])


def probe(enc, xtr, ytr, xte, yte, n=10000):
    ftr, fte = features(enc, xtr[:n]), features(enc, xte)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    lin = torch.nn.Linear(ftr.shape[1], 10)
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(lin(ftr), ytr[:n]).backward(); opt.step()
    return (lin(fte).argmax(1) == yte).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/data/mnist.npz")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", default="ckpt.pt")
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(0)
    xtr, ytr, xte, yte = load_mnist(a.data)
    c = Cfg()
    ctx = Encoder(c); pred = Predictor(c); tgt_enc = copy.deepcopy(ctx).requires_grad_(False)
    print(f"encoder params: {sum(p.numel() for p in ctx.parameters())/1e6:.3f} M", flush=True)
    print(f"probe acc @ random init: {probe(tgt_enc, xtr, ytr, xte, yte):.3f}", flush=True)
    steps = int(a.epochs * len(xtr) / a.bs)
    opt = torch.optim.AdamW(list(ctx.parameters()) + list(pred.parameters()), 1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for s in range(steps):
        ctx.train(); pred.train()
        idx = torch.randint(0, len(xtr), (a.bs,), generator=gen)
        x = xtr[idx]
        keep, tgt = masks(a.bs, c.img // c.patch, gen)
        with torch.no_grad():
            h = tgt_enc(x)
            h = F.layer_norm(h, h.shape[-1:])
            h = torch.gather(h, 1, tgt[..., None].expand(-1, -1, c.dim))
        p = pred(ctx(x, keep), keep, tgt)
        loss = F.smooth_l1_loss(p, h)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = 0.996 + (1 - 0.996) * s / steps
        with torch.no_grad():
            for pt, pc in zip(tgt_enc.parameters(), ctx.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1 - m)
        if s % 100 == 0 or s == steps - 1:
            print(f"step {s}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    print(f"probe acc after JEPA: {probe(tgt_enc, xtr, ytr, xte, yte):.3f}", flush=True)
    torch.save({"cfg": c.__dict__, "encoder": tgt_enc.state_dict()}, a.out)


if __name__ == "__main__":
    main()
