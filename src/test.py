import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)  # [C]
        self.std  = torch.tensor(std,  dtype=torch.float32)  # [C]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(t.device, t.dtype)[..., None, None]  # [C,1,1]
        std  = self.std.to(t.device,  t.dtype)[..., None, None]  # [C,1,1]
        if t.dim() == 3:   # [C,H,W]
            return t * std + mean
        elif t.dim() == 4: # [B,C,H,W]
            return t * std.unsqueeze(0) + mean.unsqueeze(0)
        else:
            raise ValueError(f"UnNormalize expects 3D or 4D tensor, got shape {tuple(t.shape)}")

def psnr_from_mse(mse, max_val=1.0):
    mse_t = torch.as_tensor(mse, dtype=torch.float32)
    return 10.0 * torch.log10((max_val ** 2) / (mse_t + 1e-10))

def _to_uint8_img(t):
    """t: [C,H,W] in [0,1] -> numpy uint8 [H,W,C]"""
    t = t.detach().cpu().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")

def test_loop(device, model, test_loader):
    model.to(device)
    model.eval()

    BSDS_unorm = UnNormalize(
        mean=(0.4612, 0.4608, 0.4009),
        std=(0.2511, 0.2495, 0.2706)
    )

    os.makedirs("result", exist_ok=True)

    psnr_list = []
    global_idx = 0  # fileindex for saving images

    with torch.no_grad():
        for batch_idx, imgs in enumerate(test_loader):
            imgs = imgs.to(device, non_blocking=True)       # [B,C,H,W]
            outputs, *_ = model(imgs)                       # [B,C,H,W]

            # Unnormalize to [0,1] for visualization / PSNR
            outs_vis = torch.clamp(BSDS_unorm(outputs), 0.0, 1.0)
            imgs_vis = torch.clamp(BSDS_unorm(imgs),    0.0, 1.0)

            # Per-image MSE & PSNR
            per_pix = F.mse_loss(outs_vis, imgs_vis, reduction='none')   # [B,C,H,W]
            per_img_mse = per_pix.view(per_pix.size(0), -1).mean(dim=1)  # [B]
            per_img_psnr = psnr_from_mse(per_img_mse, max_val=1.0)       # [B]

            psnr_list.extend(per_img_psnr.cpu().tolist())

            # === Save side-by-side comparison for each image ===
            B = imgs_vis.size(0)
            for i in range(B):
                gt_img   = _to_uint8_img(imgs_vis[i])   # [H,W,C]
                pred_img = _to_uint8_img(outs_vis[i])
                p = float(per_img_psnr[i].item())

                plt.figure(figsize=(8, 4))
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(gt_img)
                ax1.set_title("Ground Truth")
                ax1.axis("off")

                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(pred_img)
                ax2.set_title(f"Prediction (PSNR: {p:.2f} dB)")
                ax2.axis("off")

                out_path = os.path.join("result", f"sample_{global_idx:05d}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()
                global_idx += 1

            print(f'Batch {batch_idx+1}, PSNR (avg in batch): {per_img_psnr.mean().item():.2f} dB')

    avg_psnr = sum(psnr_list) / len(psnr_list)
    print(f'Average PSNR over the test set: {avg_psnr:.2f} dB')
    return avg_psnr