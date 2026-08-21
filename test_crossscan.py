"""Quick test — no stdout redirect, minimal overhead."""
import os, sys, torch, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
from lib.extract_patches import extract_ordered_overlap, recompone_overlap, load_data
from lib.pre_processing import my_PreProc
from lib.metrics import Evaluate
from lib.common import setpu_seed, dict_round
from lib.visualize import save_img, concat_result
from lib.postprocess import connect_vessel_by_direction
from train_cross_scan import create_model

setpu_seed(2021)

def build_args():
    class A: pass
    a = A()
    a.save = "./experiments/ablation_Phase1/ours_full_crossscan"
    a.vit_name='R50-ViT-B_16'; a.img_size=224; a.vit_patches_size=16
    a.n_skip=3; a.use_hybrid=True; a.use_direction_attention=True
    a.attention_type='channel'; a.attention_reduction=16; a.num_directions=4
    a.hybrid_d_state=16; a.hybrid_d_model=256; a.hybrid_num_layers=2
    a.classes=2; a.n_classes=2
    return a

args = build_args()
save_path = join(args.save)

# Model
print("Creating model...", flush=True)
net = create_model(args, torch.device('cuda'))
ckpt = torch.load(join(save_path, 'top1_model.pth'), map_location='cuda')
net.load_state_dict(ckpt['net'], strict=False)
net = net.cuda().eval()
torch.backends.cudnn.benchmark = True
print(f"Loaded epoch {ckpt['epoch']}, {sum(p.numel() for p in net.parameters()):,} params", flush=True)

# Data
test_path = './prepare_dataset/data_path_list/DRIVE/test.txt'
imgs_ori, gts_ori, fovs_ori = load_data(test_path)
imgs_ori = my_PreProc(imgs_ori)  # Normalization (same as get_data_test_overlap)
gts_ori = gts_ori / 255.0  # Normalize to [0, 1]
fovs_ori = fovs_ori // 255  # Binary FOV mask
N = imgs_ori.shape[0]
H, W = imgs_ori.shape[2], imgs_ori.shape[3]
print(f"Data: {N} imgs, {H}x{W}", flush=True)

ph, pw, sh, sw = 224, 224, 16, 16

all_pred = []
for i in range(N):
    img = imgs_ori[i:i+1]
    pad_h = (sh - (H - ph) % sh) % sh
    pad_w = (sw - (W - pw) % sw) % sw
    new_h, new_w = H + pad_h, W + pad_w
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode='reflect')
    
    patches = extract_ordered_overlap(img, ph, pw, sh, sw)
    print(f"  [{i+1}/{N}] {patches.shape[0]} patches", end=' ', flush=True)
    
    preds = []
    ds = torch.utils.data.TensorDataset(torch.from_numpy(patches).float())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (batch,) in loader:
            out = net(batch.cuda())
            preds.append(out[:, 1].cpu().numpy())
    pred_patch = np.expand_dims(np.concatenate(preds), 1)
    
    pred_img = recompone_overlap(pred_patch, new_h, new_w, sh, sw)
    pred_img = pred_img[:, :, 0:H, 0:W]
    all_pred.append(pred_img[0, 0])
    print(f"→ done", flush=True)

# Post-process: direction-aware vessel connection (dt=3 to avoid hang)
print("Post-processing with direction-aware connection (dt=3)...", flush=True)
pred_imgs = np.array(all_pred)[:, np.newaxis, :, :]
import time
for i in range(N):
    t0 = time.time()
    pred_imgs[i, 0] = connect_vessel_by_direction(pred_imgs[i, 0], distance_threshold=3, angle_threshold=5)
    print(f"  [{i+1}/{N}] {time.time()-t0:.1f}s", flush=True)

# Binarize predictions (matching original test.py pipeline)
pred_imgs_binary = (pred_imgs > 0.5).astype(np.float32)
print("Post-processed + binarized", flush=True)

# Eval on BINARY predictions (same as original test.py)
print("Evaluating...", flush=True)
y_scores = np.concatenate([pred_imgs_binary[i, 0][fovs_ori[i, 0].astype(bool)] for i in range(N)])
y_true = np.concatenate([gts_ori[i, 0][fovs_ori[i, 0].astype(bool)].astype(int) for i in range(N)])

eval_obj = Evaluate(save_path=save_path)
eval_obj.add_batch(y_true, y_scores)
log = eval_obj.save_all_result(plot_curve=False, save_name="performance_crossscan.txt")

print("\n=== CROSS-SCAN FULL RESULTS ===", flush=True)
for k, v in dict_round(log, 6).items():
    print(f"  {k}: {v}")

np.save(join(save_path, 'result_crossscan.npy'), np.asarray([y_true, y_scores]))

# Save imgs
rdir = join(save_path, 'result_img')
os.makedirs(rdir, exist_ok=True)
for i in range(N):
    p = pred_imgs_binary[i].copy()
    p[0] *= fovs_ori[i, 0]
    save_img(concat_result(imgs_ori[i], p, gts_ori[i]), join(rdir, f"Result_{i}.png"))
    save_img(p[0]*255, join(rdir, f"Gray_{i}.png"))
print("ALL DONE", flush=True)
