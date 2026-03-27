from open_clip import create_model_and_transforms
from torch.optim import Adam
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm  # 进度条
import json
import os
import random  # subsample
import logging  # 日志

# 设置日志 (输出到train_log.txt)
logging.basicConfig(filename='train_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting projector training on 4090 with OOM-safe batch=4')

# 加载本地safetensors
safetensors_path = '/root/BriLLM0.5/CLIP-ViT-bigG-14-laion2B-39B-b160k/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'
model, _, preprocess = create_model_and_transforms('ViT-bigG-14', pretrained=None)
state_dict = load_file(safetensors_path, device='cpu')

# 修改1: 加打印验证加载（检查是否有missing keys）
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")  # 如果missing>0，可能部分随机
if len(missing_keys) > 0:
    logging.warning(f"Missing keys in load: {missing_keys[:5]}...")  # 日志警告

model = model.visual.cuda().eval()
logging.info('bigG model loaded')

# cond_proj (1280 -> 32)
cond_proj = nn.Linear(1280, 32).cuda()

# 修改2: 加decoder (32 -> 1280)，用于重建
decoder = nn.Linear(32, 1280).cuda()

# 修改3: optimizer加decoder参数
optimizer = Adam(list(cond_proj.parameters()) + list(decoder.parameters()), lr=1e-3)

# Custom COCO Dataset (数据盘路径)
class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        with open(annotation_file, 'r') as f:
            anns = json.load(f)
        self.annotations = anns['annotations']
        self.images = anns['images']
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.images}
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = ann['image_id']
        img_path = os.path.join(self.image_dir, self.image_id_to_file[img_id])
        image = Image.open(img_path).convert('RGB')
        caption = ann['caption']  # 保留，但暂不使用（如果需多模态，可后续加）
        return image, caption

# 加载train2017 (数据盘)
full_dataset = COCODataset(
    image_dir='/root/autodl-tmp/coco2017/train2017',
    annotation_file='/root/autodl-tmp/coco2017/annotations/captions_train2017.json'
)

# Subsample 100k annotations for speed (random, keep diversity)
random.seed(42)
indices = random.sample(range(len(full_dataset)), 100000)
dataset = Subset(full_dataset, indices)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4, collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch]))  # batch=4 OOM-safe, workers=4

# AMP for speed (no precision loss)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(1):  # 修改4: 改成5 epochs，看loss下降趋势（原始是1）
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/1')
    logging.info(f'Starting epoch {epoch+1}, batches: {len(dataloader)}')
    for images, captions in pbar:
        with torch.cuda.amp.autocast():  # AMP forward
            inputs = torch.stack([preprocess(img) for img in images]).to('cuda')  # (4,3,224,224)
            vision_out = model(inputs)  # (4,1280)
            vision_out = vision_out.mean(1) if vision_out.dim() > 2 else vision_out
            
            # 修改5: 计算cond，然后通过decoder重建
            cond = cond_proj(vision_out)  # (4,32)
            recon = decoder(cond)  # (4,1280)
            
            # 修改6: loss改成重建误差（移除random target）
            loss = F.mse_loss(recon, vision_out)  # 现在loss有意义，会下降

        scaler.scale(loss).backward()  # AMP backward
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
        logging.info(f'Batch {num_batches}: loss = {loss.item():.4f}')
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1}: Avg loss = {avg_loss:.4f}')
    logging.info(f'Epoch {epoch+1} complete: Avg loss = {avg_loss:.4f}')

# 修改7: 保存时也save decoder
torch.save({'cond_proj': cond_proj.state_dict(), 'decoder': decoder.state_dict()}, 'cond_proj.pth')
logging.info('Training complete, projector saved.')
print('Saved bigG projector.')