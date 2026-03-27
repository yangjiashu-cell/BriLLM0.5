import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR  # Add scheduler
import numpy as np
from tqdm import tqdm
import logging
import nibabel as nib  # for ROI mask
import os

logging.basicConfig(filename='mae_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class ROIMAE(nn.Module):
    def __init__(self, dim=768, patch_size=16, voxels=10000):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, dim))  # 196 patches + CLS for 224x224
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 8, batch_first=True), 6
        )
        self.decoder = nn.Linear(dim, voxels)  # decoder for original 10000 voxels recon

    def forward(self, x, mask=None):  # x: (b,1,100,100) original fMRI ROI
        b = x.shape[0]
        original_flat = x.view(b, -1)  # (b,10000) for recon/loss
        # pad to 224x224 (LEA-inspired ROI pad)
        pad_h = 224 - 100
        pad_w = 224 - 100
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)  # (b,1,224,224)
        patches = self.patch_embed(x_padded).flatten(2).transpose(1,2)  # (b,196,dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, patches], 1) + self.pos_embed  # (b,197,dim)
        enc = self.encoder(x)
        cls_out = enc[:,0]  # [CLS] (b,768)
        recon = self.decoder(cls_out)  # (b,10000) recon original voxels
        if mask is not None:
            return recon[:, mask], cls_out  # masked positions recon
        return recon, cls_out

mae = ROIMAE().cuda()
optimizer = Adam(mae.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)  # Add momentum/betas for smoother descent
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # lr decay every 10 epochs
# Load real data (from preprocess)
god_fmri = np.load('/root/BriLLM0.5/ds001246_fmri_10k.npy')  # (N,10000), N>45 now
god_fmri = (god_fmri - god_fmri.mean(1, keepdims=True)) / (god_fmri.std(1, keepdims=True) + 1e-6)  # Normalize std~0.999 (LEA-inspired)
god_fmri_2d = god_fmri.reshape(len(god_fmri), 1, 100, 100).astype(np.float32)  # (N,1,100,100)

# Load ROI mask (from sourcedata/mask.nii.gz, assume flattened to 10000 bool)
roi_mask_path = '/root/autodl-tmp/ds001246/sourcedata/mask.nii.gz'  # adjust if needed
if os.path.exists(roi_mask_path):
    roi_mask_img = nib.load(roi_mask_path)
    roi_mask = roi_mask_img.get_fdata().flatten()[:10000] > 0  # bool mask for ROI voxels
else:
    roi_mask = np.ones(10000, dtype=bool)  # fallback full mask
    logging.warning('ROI mask not found, using full voxels')

# Train-val split (80/20)
num_samples = len(god_fmri_2d)
train_size = int(0.8 * num_samples)
train_data = torch.from_numpy(god_fmri_2d[:train_size])
val_data = torch.from_numpy(god_fmri_2d[train_size:])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)

best_val_loss = float('inf')
patience = 10
counter = 0
mask_ratio = 0.75  # Move outside for val consistency
enable_early_stop = False  # Option to disable early stop and train full 50 epochs (project suggestion)

for epoch in range(50):
    mae.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f'MAE Epoch {epoch+1}/50')
    logging.info(f'Starting MAE epoch {epoch+1}, train batches: {len(train_loader)}')
    for x in pbar:
        x = x.to('cuda')
        b = x.shape[0]
        # mask 75% within ROI only (LEA-inspired, Page3 ROI embedding + masked loss)
        roi_indices = np.where(roi_mask)[0]  # indices of ROI voxels
        num_roi = len(roi_indices)
        num_mask = int(mask_ratio * num_roi)
        mask_idx = np.random.choice(roi_indices, num_mask, replace=False)  # mask only in ROI
        mask = torch.zeros(10000, dtype=torch.bool, device='cuda')
        mask[mask_idx] = True
        masked_x = x.clone()
        original_flat = x.view(b, -1)  # (b,10000)
        masked_flat = original_flat.clone()
        masked_flat[:, mask] = 0  # mask on flattened
        masked_flat = torch.nan_to_num(masked_flat, nan=0.0)  # handle nan
        masked_x = masked_flat.view_as(x)  # back to (b,1,100,100)
        recon_masked, pred_cls = mae(masked_x, mask=mask)
        # loss on masked voxels only (Eq1 Lfmri MSE)
        loss = F.mse_loss(recon_masked, original_flat[:, mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
        logging.info(f'MAE Train Batch {num_batches}: loss = {loss.item():.4f}')
    avg_train_loss = total_loss / num_batches
    print(f'MAE Epoch {epoch}: Avg train loss = {avg_train_loss:.4f}')
    logging.info(f'MAE Epoch {epoch} train complete: Avg loss = {avg_train_loss:.4f}')

    # Val (unify to masked loss, like train)
    mae.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to('cuda')
            b = x.shape[0]
            # Same mask logic for val
            roi_indices = np.where(roi_mask)[0]
            num_roi = len(roi_indices)
            num_mask = int(mask_ratio * num_roi)
            mask_idx = np.random.choice(roi_indices, num_mask, replace=False)
            mask = torch.zeros(10000, dtype=torch.bool, device='cuda')
            mask[mask_idx] = True
            masked_x = x.clone()
            original_flat = x.view(b, -1)
            masked_flat = original_flat.clone()
            masked_flat[:, mask] = 0
            masked_flat = torch.nan_to_num(masked_flat, nan=0.0)
            masked_x = masked_flat.view_as(x)
            recon_masked, _ = mae(masked_x, mask=mask)
            loss = F.mse_loss(recon_masked, original_flat[:, mask])
            val_loss += loss.item()
            val_batches += 1
    avg_val_loss = val_loss / val_batches
    print(f'MAE Epoch {epoch}: Avg val loss = {avg_val_loss:.4f}')
    logging.info(f'MAE Epoch {epoch} val: Avg loss = {avg_val_loss:.4f}')

    scheduler.step()  # Step scheduler after val

    # Early stopping (if enabled)
    if enable_early_stop:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(mae.state_dict(), 'fMRI_mae.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    else:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(mae.state_dict(), 'fMRI_mae.pth')

# 提取CLS for all (train+val)
mae.eval()
cls_list = []
with torch.no_grad():
    full_loader = torch.utils.data.DataLoader(torch.from_numpy(god_fmri_2d), batch_size=4)
    for x in full_loader:
        x = x.to('cuda')
        _, cls_out = mae(x)  # no mask
        cls_list.append(cls_out.cpu().numpy())
np.save('ds001246_fMRI_cls.npy', np.concatenate(cls_list, axis=0))  # (N,768), update name
print('MAE pretrained and CLS extracted.')