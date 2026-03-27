import nibabel as nib
import numpy as np
from glob import glob
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import os
import logging
import pickle

logging.basicConfig(filename='/root/BriLLM0.5/preprocess_ds001246_log.txt', 
                    level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Starting ds001246 final preprocess with robust image matching and error handling')

base_dir = '/root/autodl-tmp/ds001246'
deriv_dir = os.path.join(base_dir, 'derivatives/preproc-spm/output')  # Preprocessed bold path
stim_base = '/root/autodl-tmp/ds001246/BOLD5000_Stimuli/Scene_Stimuli/Original_Images'  # 你的实际 stimuli 根目录

# 预先遍历所有图像，建立 basename（小写）→ 真实路径 的映射，支持大小写不敏感 + 扩展名差异
print("正在预加载所有图像路径（这可能需要几秒到1分钟，取决于图像数量）...")
image_map = {}
for root, _, files in os.walk(stim_base):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 支持常见图像格式
            key = file.lower()
            full_path = os.path.join(root, file)
            image_map[key] = full_path

logging.info(f'预加载了 {len(image_map)} 张图像到映射表')
print(f'预加载完成，共 {len(image_map)} 张图像')

# Find all preprocessed bold files
bold_files = sorted(glob(f'{deriv_dir}/sub-*/ses-perception*/func/*_task-perception_run-*_bold_preproc.nii.gz'))

def get_events_file(bold_file):
    parts = os.path.basename(bold_file).split('_')
    sub = parts[0]
    ses = parts[1]
    task = parts[2]
    run = parts[3]
    orig_dir = os.path.join(base_dir, sub, ses, 'func')
    events_filename = f'{sub}_{ses}_{task}_{run}_events.tsv'
    return os.path.join(orig_dir, events_filename)

events_files = [get_events_file(f) for f in bold_files]

logging.info(f'Found {len(bold_files)} preprocessed bold files (expected ~76)')
if len(bold_files) == 0:
    print("错误：没有找到任何 preprocessed bold 文件！请检查 deriv_dir 路径和文件名是否正确（是否真的是 *_bold_preproc.nii.gz）")
    exit(1)

fmri_list = []
image_list = []
missing_count = 0
corrupted_count = 0
total_trials = 0

for bold_file, events_file in tqdm(zip(bold_files, events_files), total=len(bold_files)):
    if not os.path.exists(events_file):
        logging.warning(f'No events file found for {bold_file}: {events_file}')
        continue
    
    try:
        img = nib.load(bold_file)
        data = img.get_fdata()  # (x, y, z, time)
        TR = round(img.header.get_zooms()[-1], 1)
        if TR <= 0:
            logging.warning(f'Invalid TR {TR} in {bold_file}, skipping run')
            continue
        
        events = pd.read_csv(events_file, sep='\t')
        logging.info(f'Events columns in {events_file}: {list(events.columns)}')
        
        # BOLD5000 的列名通常是 'onset', 'duration', 'event_type', 'stimulus_name'（或类似）
        required_cols = ['onset', 'event_type', 'stimulus_name']
        if not all(col in events.columns for col in required_cols):
            logging.warning(f'Missing required columns in {events_file}, available: {list(events.columns)}')
            continue
        
        stim_events = events[events['event_type'] == 'stimulus']
        if len(stim_events) == 0:
            logging.warning(f'No stimulus events in {events_file}')
            continue
        
        for _, row in stim_events.iterrows():
            total_trials += 1
            onset = row['onset']
            
            if not np.isfinite(onset):
                logging.warning(f'Skipping trial with non-finite onset: {onset}')
                continue
            
            raw_stim_name = str(row['stimulus_name']).strip()
            if raw_stim_name == '' or raw_stim_name == 'nan':
                logging.warning(f'Empty stimulus_name, skipping')
                continue
            
            # 取文件名（防止 stimulus_name 带路径）
            stim_name = os.path.basename(raw_stim_name)
            key = stim_name.lower()
            
            # 查找策略1：直接匹配（包括扩展名 + 大小写不敏感）
            img_path = image_map.get(key)
            
            # 策略2：如果没找到且没有扩展名，尝试常见扩展名
            if not img_path and '.' not in key:
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate_key = key + ext
                    if candidate_key in image_map:
                        img_path = image_map[candidate_key]
                        break
            
            # 策略3：如果还是没找到，尝试去掉扩展名再加常见扩展名
            if not img_path:
                base_no_ext = os.path.splitext(key)[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate_key = base_no_ext + ext
                    if candidate_key in image_map:
                        img_path = image_map[candidate_key]
                        break
            
            # 加载图像
            if img_path:
                try:
                    pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
                except UnidentifiedImageError:
                    logging.warning(f'Corrupted/unidentifiable image (PIL cannot open): {img_path} (stim_name: {stim_name})')
                    pil_img = Image.new('RGB', (224, 224), (255, 0, 0))  # 红色 dummy
                    corrupted_count += 1
                except Exception as e:
                    logging.warning(f'Error opening image {img_path}: {e}')
                    pil_img = Image.new('RGB', (224, 224), (255, 0, 0))
                    corrupted_count += 1
            else:
                pil_img = Image.new('RGB', (224, 224), (255, 0, 0))  # 红色 dummy
                missing_count += 1
                logging.warning(f'Missing image: {stim_name} (original: {raw_stim_name})')
            
            # fMRI 提取
            start_tr = int((onset + 4) // TR)
            end_tr = int((onset + 10) // TR)
            if start_tr >= data.shape[-1] or end_tr > data.shape[-1] or start_tr >= end_tr:
                logging.warning(f'Invalid TR range: start={start_tr}, end={end_tr}, data_len={data.shape[-1]}')
                continue
            
            vol = data[..., start_tr:end_tr].mean(axis=-1)
            voxels = vol.flatten()[:10000]
            voxels = np.nan_to_num(voxels, nan=0.0, posinf=0.0, neginf=0.0)  # 处理可能的 nan/inf
            
            fmri_list.append(voxels)
            image_list.append(pil_img)
    
    except Exception as e:
        logging.error(f'Error processing {bold_file}: {e}')
        continue

fmri_array = np.array(fmri_list)
logging.info(f'Processed {len(fmri_array)} valid trials, {missing_count} missing, {corrupted_count} corrupted (out of {total_trials} total trials)')

np.save('/root/BriLLM0.5/ds001246_fmri_10k.npy', fmri_array)
with open('/root/BriLLM0.5/ds001246_images_pil.pkl', 'wb') as f:
    pickle.dump(image_list, f)
np.save('/root/BriLLM0.5/ds001246_images_np.npy', np.array([np.array(img) for img in image_list]))

print(f'完成！得到 {len(fmri_array)} 个有效配对')
print(f'缺失图像: {missing_count}，损坏图像: {corrupted_count}')
if missing_count > 50:
    print("警告：缺失图像较多！请检查：1) BOLD5000_Stimuli 是否完整下载并解压到正确路径 2) 图像文件名与 events.tsv 中的 stimulus_name 是否完全匹配（大小写、扩展名）")
if corrupted_count > 10:
    print("警告：损坏图像较多！可能是下载时文件损坏，建议重新下载对应 CSI 文件夹")
logging.info('Preprocess finished')