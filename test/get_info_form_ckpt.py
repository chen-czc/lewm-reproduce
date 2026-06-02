import torch

ckpt = torch.load('/HOME/sysu_ntan/sysu_ntan_5/chenzc/le-wm/lewm/d2az0cg8/checkpoints/epoch=41-step=53928.ckpt', map_location='cpu', weights_only=False)
print('Type:', type(ckpt))
print('Keys:', list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A')