import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from stgcn import GRU_GCN_Attention
from utils import load_metr_la_data, get_normalized_adj, generate_dataset
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# 参数
# ======================
num_timesteps_input = 30
num_timesteps_output = 30
mask_ratio = 0.3

checkpoint_path = "checkpoints/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_names = [
    "Water_temperature","Water_pH","Water_DO","Water_Mn","Water_NH4","Water_EC","Water_TB","Water_TN","Water_TP",
    "Tmpmean","Tmpmax","Tmpmin","Prec","Pres","PetPM","Rhu","Wind",
    "bio_1","bio_2","bio_3","bio_4","bio_5","bio_6","bio_7","bio_8","bio_9",
    "bio_10","bio_11","bio_12","bio_13","bio_14","bio_15","bio_16","bio_17","bio_18","bio_19",
    "Runoff","Upstream Area","Elevation","Nitrogen Deposition","BD","CEC","CLAY","pH","SOC",
    "TN","TP","Nitrogen Fertilizer","Phosphorus Fertilizer","GDP","Population","Urban Area",
    "NDVI","DIS_AV_CMS"
]

target_idx = feature_names.index("Water_TP")


# ======================
# 1. 数据
# ======================
A, X_norm, means, stds, X_original = load_metr_la_data(target_idx)

T_total = X_norm.shape[2]
split_t = int((T_total - num_timesteps_input + 1) * 0.8)

X_val = X_norm[:, :, split_t - num_timesteps_input:]
X_val_original = X_original[:, :, split_t - num_timesteps_input:]

# ======================
# 时间轴（关键修复）
# ======================
start_date = "2020-11-01"

full_time_index = pd.date_range(
    start=start_date,
    periods=T_total,
    freq="D"
)

# ⭐ test 对齐时间轴（必须和 X_val 一致）
test_time_index = full_time_index[
    split_t - num_timesteps_input :
]

print("X_val time len:", X_val.shape[2])
print("time_index len:", len(test_time_index))
print("time_index:", test_time_index)


val_features, _, val_t_idx, val_ob_mask, val_gt_mask = generate_dataset(
    X_val,
    X_val_original,
    num_timesteps_input,
    num_timesteps_output,
    target_idx=target_idx,
    missing_ratio=mask_ratio,
    stride=num_timesteps_output,
    random_seed=7
)

print("T_total:", T_total)
print("X_val.shape:", X_val.shape)

true_full = X_val_original[:, target_idx, :]
print('X_val_original',X_val_original.shape)

original_missing_mask = (true_full == 0) | np.isnan(true_full)

# ======================
# 2. 模型
# ======================
A_wave = torch.from_numpy(get_normalized_adj(A)).to(device)

net = GRU_GCN_Attention(
    num_nodes=A_wave.shape[0],
    in_features=X_norm.shape[1],
    timesteps_input=num_timesteps_input,
    timesteps_output=num_timesteps_output
).to(device)

model_file=os.path.join(checkpoint_path,'stgcn_interp_best.pth')
checkpoint=torch.load(model_file,map_location=device,weights_only=False)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# ======================
# 3. 全序列插补（核心）
# ======================
def evaluate_all(features, ob_mask):
    pred_list = []

    with torch.no_grad():
        for X, ob in zip(features, ob_mask):

            X = X.unsqueeze(0).to(device)
            ob = ob.numpy()

            # ⭐ 仅用于输入：窗口观测 mask（不能用 global mask）
            cond_mask = ob

            X_masked = X.clone()

            X_masked[:,:,:,target_idx] *= torch.from_numpy(cond_mask).unsqueeze(0).to(device)

            pred = net(A_wave, X_masked).cpu().numpy()[0]
            pred_list.append(pred)

    pred = np.stack(pred_list)
    pred = pred * stds[target_idx] + means[target_idx]

    return pred


# ======================
# 4. 拼接窗口（关键：不使用任何 mask）
# ======================
def aggregate_direct(pred, t_idx, original_shape):
    N, T_total = original_shape
    pred_full = np.full((N, T_total), np.nan)

    B, _, T = pred.shape

    for i in range(B):
        start = t_idx[i]
        end = start + T

        for n in range(N):
            start = t_idx[i]
            end = start + T

            # ⭐ 直接覆盖（最后一个窗口自然会 overwrite）
            pred_full[:, start:end] = pred[i]

    return pred_full

missing_mask_global = np.zeros_like(true_full, dtype=bool)

for i in range(len(val_t_idx)):
    start = val_t_idx[i]
    end = start + num_timesteps_output

    # gt_mask == 0 表示人为mask
    missing_mask_global[:, start:end] |= (val_gt_mask[i].numpy() == 0)

# ======================
# 5. 最终融合（唯一使用 global mask）
# ======================
def build_final_series(true_full, pred_full, missing_mask):

    final = true_full.copy()

    # ⭐ 只填人为mask的位置
    final[missing_mask] = pred_full[missing_mask]

    return final

def plot_result(node_idx, true_full, pred_full, missing_mask,original_missing_mask,time_index):

    plt.figure(figsize=(12,5))

    t = time_index  # ⭐ 使用 test 对齐时间轴

    # ======================
    # 1️⃣ 插补结果（红线）
    # ======================
    imputed_line = true_full[node_idx].copy()
    mask_idx = missing_mask[node_idx] & (~original_missing_mask[node_idx])
    # imputed_line[missing_mask[node_idx]] = pred_full[node_idx][missing_mask[node_idx]]
    imputed_line[mask_idx] = pred_full[node_idx][mask_idx]

    # 红线：填补值（缺失处）
    plt.plot(
        t,
        imputed_line,
        'r--',
        label='Imputed'
    )

    # ======================
    # ⭐ 关键：构造“可断线版本”
    # ======================
    true_plot = true_full[node_idx].copy()

    # ❗把缺失位置变成 NaN（强制断线）
    true_plot[original_missing_mask[node_idx]] = np.nan

    # 蓝线：真实值（非缺失）
    plt.plot(
        t,
        true_plot,
        'b-',
        label='Observed'
    )

    png_path = os.path.join('predictions_interp/node_series_compare2', f'node_{node_idx}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    svg_path = os.path.join('predictions_interp/node_series_compare2', f'node_{node_idx}.svg')
    plt.savefig(svg_path, bbox_inches='tight')


missing_mask = (true_full == 0) | np.isnan(true_full)

print(">>> Generating predictions...")
val_pred_all = evaluate_all(val_features, val_ob_mask)

print(">>> Aggregating (last overwrite)...")
val_pred_full = aggregate_direct(
    val_pred_all,
    val_t_idx,
    original_shape=(X_val.shape[0], X_val.shape[2])
)

print(">>> Building final series...")
final_series = build_final_series(
    true_full,
    val_pred_full,
    missing_mask_global
)

print(">>> Plot...")

for node in range(true_full.shape[0]):
    plot_result(
        node_idx=node,
        true_full=true_full,
        pred_full=val_pred_full,
        missing_mask=missing_mask_global,
        original_missing_mask=original_missing_mask,
        time_index=test_time_index
    )

print("✅ All node plots saved!")


