import os
import pickle as pk
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from scipy import stats

from stgcn import GRU_GCN_Attention
from utils import load_metr_la_data, get_normalized_adj,generate_dataset

# 插补任务参数
num_timesteps_input = 30
num_timesteps_output = 30
mask_ratio = 0.3
train_ratio = 0.8

# 路径和设备
checkpoint_path = "checkpoints/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 目标特征索引
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

# 1. 加载数据和mask
A,X_norm,  means, stds, X_original = load_metr_la_data(target_idx)
# print(X_norm.shape)
# ===== 划分时间 =====
T_total = X_norm.shape[2]
split_t = int((T_total - num_timesteps_input + 1) * 0.8)

X_train = X_norm[:, :, :split_t]
X_train_original=X_original[:, :, :split_t]
X_val = X_norm[:, :, split_t - num_timesteps_input:]
X_val_original = X_original[:, :, split_t - num_timesteps_input:]

# ===== 分别生成 =====
train_features, train_target, train_t_idx, train_ob_mask, train_gt_mask = generate_dataset(
    X_train,
    X_train_original,
    num_timesteps_input,
    num_timesteps_output,
    target_idx=target_idx,
    missing_ratio=mask_ratio,
    random_seed=7
)

val_features, val_target, val_t_idx, val_ob_mask, val_gt_mask = generate_dataset(
    X_val,
    X_val_original,
    num_timesteps_input,
    num_timesteps_output,
    target_idx=target_idx,
    missing_ratio=mask_ratio,
    random_seed=7
)

# 2. 加载插补模型
A_wave=get_normalized_adj(A)
A_wave=torch.from_numpy(A_wave).to(device)

net=GRU_GCN_Attention(
    num_nodes=A_wave.shape[0],
    in_features=X_norm.shape[1],
    timesteps_input=num_timesteps_input,
    timesteps_output=num_timesteps_output
).to(device)
print(X_norm.shape[1])

# 加载最佳模型权重
model_file=os.path.join(checkpoint_path,'stgcn_interp_best.pth')
checkpoint=torch.load(model_file,map_location=device,weights_only=False)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# ======================
# 评估函数
# ======================
def evaluate(features, target, ob_mask, gt_mask):
    pred_list, true_list, mask_list = [], [], []
    real_missing_list = []   # ⭐新增

    with torch.no_grad():
        for X, y, ob, gt in zip(features, target, ob_mask, gt_mask):

            X = X.unsqueeze(0).to(device)
            y = y.numpy()

            ob = ob.numpy()
            gt = gt.numpy()

            # ===== mask =====
            cond_mask = gt
            target_mask = ob * (1 - gt)

            # ⭐真正的原始缺失
            real_missing = (ob == 0)

            X_masked=X.clone()
            X_masked[:,:,:,target_idx] *=  torch.from_numpy(cond_mask).unsqueeze(0).to(device)

            pred = net(A_wave, X_masked).cpu().numpy()[0]

            pred_list.append(pred)

            true_list.append(y)
            mask_list.append(target_mask)
            real_missing_list.append(real_missing)   # ⭐保存

        pred = np.stack(pred_list)
    true = np.stack(true_list)
    mask = np.stack(mask_list)
    real_missing = np.stack(real_missing_list)   # ⭐新增
    
    print(pred.shape)
    # ===== 反归一化 =====
    pred = pred * stds[target_idx] + means[target_idx]
    true = true * stds[target_idx] + means[target_idx]

    # ===== flatten =====
    mask_flat = mask.flatten()
    pred_flat = pred.flatten()[mask_flat == 1]
    true_flat = true.flatten()[mask_flat == 1]

    return pred, true, mask,real_missing,pred_flat, true_flat

def aggregate_predictions(pred, true, mask,real_missing, t_idx, original_shape):
    """
    pred, true, mask: (B, N, T)
    t_idx: (B,)
    original_shape: (N, T_total)

    return:
        pred_full, true_full, valid_mask
    """
    # print(original_shape)
    N, T_total = original_shape

    sum_pred = np.zeros((N, T_total))
    sum_true = np.zeros((N, T_total))
    count = np.zeros((N, T_total))
    real_missing_count = np.zeros((N, T_total))   # ⭐新增

    B, _, T = pred.shape

    for i in range(B):
        start = t_idx[i]
        end = start + T

        for n in range(N):
            idx = (mask[i, n] == 1)

            sum_pred[n, start:end][idx] += pred[i, n][idx]
            sum_true[n, start:end][idx] += true[i, n][idx]
            count[n, start:end][idx] += 1

            rm_idx = (real_missing[i, n] == 1)

            real_missing_count[n, start:end][rm_idx] += 1

    valid = count > 0

    pred_full = np.zeros((N, T_total))
    true_full = np.zeros((N, T_total))

    pred_full[valid] = sum_pred[valid] / count[valid]
    true_full[valid] = sum_true[valid] / count[valid]

    real_missing_mask = real_missing_count > 0   # ⭐关键输出

    return pred_full, true_full, valid, real_missing_mask

# def aggregate_predictions_last(pred, true, mask, real_missing, t_idx, original_shape):
#     """
#     只保留最后一次预测（覆盖策略）
#     """
#     N, T_total = original_shape

#     pred_full = np.zeros((N, T_total))
#     true_full = np.zeros((N, T_total))
#     valid_mask = np.zeros((N, T_total), dtype=bool)
#     real_missing_mask = np.zeros((N, T_total), dtype=bool)

#     B, _, T = pred.shape

#     for i in range(B):
#         start = t_idx[i]
#         end = start + T

#         for n in range(N):
#             idx = (mask[i, n] == 1)
#             rm_idx = (real_missing[i, n] == 1)

#             # ⭐关键：直接覆盖（后面的窗口会覆盖前面的）
#             pred_full[n, start:end][idx] = pred[i, n][idx]
#             true_full[n, start:end][idx] = true[i, n][idx]

#             valid_mask[n, start:end][idx] = True
#             real_missing_mask[n, start:end][rm_idx] = True

#     return pred_full, true_full, valid_mask, real_missing_mask

def smape(y_true,y_pred):
    denom=(np.abs(y_true)+np.abs(y_pred))/2.0
    return np.mean(np.abs(y_pred - y_true) / denom)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8)))

# ======================
# 训练集评估
# ======================
# print("=== Train ===")
# train_pred, train_true, train_mask, train_pred_flat, train_true_flat = evaluate(
#     train_features, train_target, train_ob_mask, train_gt_mask
# )

# train_r2 = r2_score(train_true_flat, train_pred_flat)
# train_rmse = np.sqrt(mean_squared_error(train_true_flat, train_pred_flat))
# train_mae = mean_absolute_error(train_true_flat, train_pred_flat)
# train_smape = smape(train_true_flat, train_pred_flat)
# train_mape = mape(train_true_flat, train_pred_flat)

# print(f"Train R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f},SMAPE={train_smape:.4f},, MAPE={train_mape:.4f}")

# ======================
# 验证集评估
# ======================
print("\n=== Validation ===")
# val_pred, val_true, val_mask, val_pred_flat, val_true_flat = evaluate(
#     val_features, val_target, val_ob_mask, val_gt_mask
# )
val_pred, val_true, val_mask,val_real_missing, _, _ = evaluate(
    val_features, val_target, val_ob_mask, val_gt_mask
)

val_pred_full, val_true_full, valid_mask, real_missing_mask = aggregate_predictions(
    val_pred,
    val_true,
    val_mask,
    val_real_missing,
    val_t_idx,
    original_shape=(val_true.shape[1], X_val.shape[2])
)
# val_pred_full, val_true_full, valid_mask, real_missing_mask = aggregate_predictions_last(
#     val_pred,
#     val_true,
#     val_mask,
#     val_real_missing,
#     val_t_idx,
#     original_shape=(val_true.shape[1], X_val.shape[2])
# )


val_pred_flat = val_pred_full[valid_mask]
val_true_flat = val_true_full[valid_mask]

print('val_pred_flat',val_pred_flat.shape)
print('val_pred_full',val_pred_full.shape)
print('val_true_flat',val_true_flat.shape)
print('val_true_full',val_true_full.shape)

val_r2 = r2_score(val_true_flat, val_pred_flat)
val_rmse = np.sqrt(mean_squared_error(val_true_flat, val_pred_flat))
val_mae = mean_absolute_error(val_true_flat, val_pred_flat)
val_smape = smape(val_true_flat, val_pred_flat)
val_mape = mape(val_true_flat, val_pred_flat)

print(f"Val R2={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f},SMAPE={val_smape:.4f}, MAPE={val_mape:.4f}")

# ======================
# 可视化
# ======================
os.makedirs("predictions_interp", exist_ok=True)

def scatter_plot(true, pred, name):
        # ===== 转为Series方便排序 =====
    x = pd.Series(true)
    y = pd.Series(pred)

    # =========================
    # ⭐ 只在画图时过滤 x > 0.6
    # =========================
    # mask = x <= 0.6
    # x = x[mask].reset_index(drop=True)
    # y = y[mask].reset_index(drop=True)

    # ===== KDE密度 =====
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    idx = z.argsort()
    x = x.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    z = z[idx]

    print(f"{name} density: min={z.min():.4f}, max={z.max():.4f}, median={np.median(z):.4f}")

    # ===== 回归线 =====
    k, b = np.polyfit(x, y, 1)
    regression_line = k * x + b

    # ===== 统一绘图风格 =====
    config = {"font.family": 'Arial', "font.size": 14, "mathtext.fontset": 'stix'}
    plt.rcParams.update(config)

    scale = max(x.max(), y.max()) * 1.01
    # scale=0.65

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    z_norm = (z - z.min()) / (z.max() - z.min())
    scatter = ax.scatter(
        x, y,
        # c=z * 100,
        c=z_norm,
        s=10,
        cmap='RdBu_r',
        alpha=0.8,
        vmin=0, vmax=1
    )

    # ===== 颜色条 =====
    cbar = plt.colorbar(scatter, pad=0.015, aspect=30)
    cbar.set_label('relative density')
    cbar.set_ticks([0, 0.5, 1])

    # ===== 1:1线 =====
    plt.plot([0, scale], [0, scale], 'red', lw=1.5, linestyle='--', label='1:1 line')

    # ===== 回归线 =====
    # plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')

    # ===== 细节 =====
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.axis([0, scale, 0, scale])
    ax.legend(loc='upper left', frameon=False)

    plt.xlabel('Observed TP')
    plt.ylabel('Predicted TP')


    # ===== 保存 =====
    save_path = f"predictions_interp/{name}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    plt.close()

# scatter_plot(train_true_flat, train_pred_flat,
#                "scatter_train1.png")

scatter_plot(val_true_flat, val_pred_flat,
               "scatter_val2.png")


# ======================
# 每个站点指标（Validation）
# ======================
num_nodes = val_pred_full.shape[0]   # 注意：你的维度是 (B, N, T)

r2_list = []
rmse_list = []
mae_list = []
smape_list = []
mape_list=[]
# print('valid_mask shape',valid_mask.shape)  (22,212)
for i in range(num_nodes):

    node_mask = valid_mask[i]

    node_pred = val_pred_full[i][node_mask == 1]
    node_true = val_true_full[i][node_mask == 1]

    if len(node_pred) == 0:
        r2_list.append(np.nan)
        rmse_list.append(np.nan)
        mae_list.append(np.nan)
        smape_list.append(np.nan)
        mape_list.append(np.nan)
    else:
        r2_list.append(r2_score(node_true, node_pred))
        rmse_list.append(np.sqrt(mean_squared_error(node_true, node_pred)))
        mae_list.append(mean_absolute_error(node_true, node_pred))
        smape_list.append(smape(node_true, node_pred))
        mape_list.append(mape(node_true, node_pred))

# 保存
df_metrics = pd.DataFrame({
    "Station_ID": np.arange(num_nodes),
    "R2": r2_list,
    "RMSE": rmse_list,
    "MAE": mae_list,
    "SMAPE": smape_list,
    "MAPE": mape_list
})

df_metrics.to_csv("predictions_interp/metrics_per_station-.csv", index=False)

print("每个站点指标已保存 → predictions_interp/metrics_per_station.csv")

records = []

np.random.seed(7)
N, T_total = val_pred_full.shape
plot_mask = np.zeros((N, T_total))  # 1=mask位置

for n in range(N):
    for t in range(T_total):
        if valid_mask[n, t] == 1:   # 只保存被mask的位置
            records.append({
                "Station": n,
                "Time": t,
                "True": val_true_full[n, t],
                "Imputed": val_pred_full[n, t]
            })

df = pd.DataFrame(records)
df.to_csv("predictions_interp/validation_imputed_values1.csv", index=False)


'''Val R2=0.8870, RMSE=0.0242, MAE=0.0148,SMAPE=0.2860, MAPE=0.3290'''