"""chunk_backbone_grad_check.py - does shared_code.chunk_backbone change outputs or gradients?

Same weights, same batch: whole stack vs chunks of n under activation checkpointing. Runs on the
CPU in fp32 with a small DaViT (same block types as MedImageInsight's DaViT-Huge), so any
difference beyond float rounding is a bug. Also checks eval mode and that parameter names
(and so checkpoints) are unchanged.

    python checks/chunk_backbone_grad_check.py [n]
"""
import copy
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runs"))
import shared_code as sc  # noqa: E402


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("davit_tiny", pretrained=False, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 12)

    def forward(self, x):
        return self.head(self.backbone(x).view(-1, 12, self.backbone.num_features).mean(1))


n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
torch.manual_seed(0)
a = M().train()
b = copy.deepcopy(a)
names = [k for k in b.state_dict()]
sc.chunk_backbone(b, n)
x = torch.randn(24, 3, 96, 96)                       # 2 studies x 12 images
y = torch.rand(2, 12)
res = []
for m in (a, b):
    out = m(x)
    nn.functional.binary_cross_entropy_with_logits(out, y).backward()
    res.append((out.detach(), {k: p.grad.clone() for k, p in m.named_parameters()}))
do = (res[0][0] - res[1][0]).abs().max().item()
dg = max((res[0][1][k] - res[1][1][k]).abs().max().item() / (res[0][1][k].abs().max().item() + 1e-12)
         for k in res[0][1])
a.eval(), b.eval()
with torch.no_grad():
    de = (a(x) - b(x)).abs().max().item()
same_names = names == [k for k in b.state_dict()]
ok = do < 1e-5 and dg < 1e-4 and de < 1e-5 and same_names
print(f"chunk {n}: train output max |diff| {do:.2e}; worst relative grad diff {dg:.2e}; "
      f"eval output max |diff| {de:.2e}; parameter names unchanged {same_names}  {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
