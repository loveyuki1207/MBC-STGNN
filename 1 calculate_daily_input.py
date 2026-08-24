import os
import numpy as np
import torch
import pandas as pd
from utils import load_metr_la_data,get_normalized_adj
from main import upstream_mass_flux
from stgcn import GRU_GCN_Attention

# 与训练脚本保持一致的参数
num_timesteps_input = 30
num_timesteps_output = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径配置
checkpoint_path = "checkpoints/"
data_path = "data/"
output_path = "daily_inputs_results/"
os.makedirs(output_path, exist_ok=True)

seconds_per_day = 24 * 3600

# 目标特征（Water_TP）
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

# ====================== 加载基础数据 ======================
# 1. 加载原始归一化数据和统计量
A, X_norm, means, stds, X_original = load_metr_la_data(target_idx)

X_norm=np.nan_to_num(X_norm,nan=0.0)

num_nodes = X_norm.shape[0]  # 站点数
num_features=X_norm.shape[1]
T_total = X_norm.shape[2]    # 总时间步数（天数）

X_norm=torch.from_numpy(X_norm).float().to(device)

# 初始化mask
ob_mask_full = ~np.isnan(X_original[:,target_idx,:])

# # 单独处理 Water_TP（0 也算缺失）
# ob_mask_full = (ob_mask_full & (X_original[:, target_idx, :] != 0))

# 转 tensor
ob_mask_full = torch.from_numpy(ob_mask_full.astype(np.float32)).to(device)
ob_mask_full = ob_mask_full.unsqueeze(1)

# 2. 加载物理驱动数据（流量、面源输入）
Q_node_daily = torch.tensor(
    np.load(os.path.join(data_path, "Q_node_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

print("Q mean:", Q_node_daily.mean().item())
print("Q min:", Q_node_daily.min().item())

L_industry_daily = torch.tensor(
    np.load(os.path.join(data_path, "L_industry_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

L_crop_farming_daily = torch.tensor(
    np.load(os.path.join(data_path, "L_crop_farming_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

L_livestock_breeding_daily = torch.tensor(
    np.load(os.path.join(data_path, "L_livestock_breeding_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

L_urban_daily=torch.tensor(
    np.load(os.path.join(data_path, "L_urban_life_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

L_rural_daily=torch.tensor(
    np.load(os.path.join(data_path, "L_rural_life_daily.npy")),
    dtype=torch.float32, device=device
)  # [T, N, 1]

# 3. 加载模型训练好的可学习参数（再悬浮项、衰减系数）
best_model_path = os.path.join(checkpoint_path, "stgcn_interp_best.pth")
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

model=GRU_GCN_Attention(
    num_nodes=num_nodes,
    in_features=num_features,
    timesteps_input=num_timesteps_input,
    timesteps_output=num_timesteps_output
)

model.to(device).load_state_dict(checkpoint["model_state_dict"])
model.eval()

a_resusp = checkpoint["a_resusp"].to(device)  # [T_total, num_nodes, 1]
lambda_raw = checkpoint["lambda_raw"].to(device)
lambda_decay = torch.nn.functional.softplus(lambda_raw)
print('lambda_raw shape:',lambda_raw.shape,'lambda_decay shape:',lambda_decay.shape)

# ======================
# 入河系数（关键修复）
# ======================
def bounded_param(x, min_v, max_v):
    return min_v + torch.sigmoid(x) * (max_v - min_v)

# coeff_mlp = CoeffMLP().to(device)
# coeff_mlp.load_state_dict(checkpoint["coeff_mlp_state_dict"])
# coeff_mlp.eval()

crop_raw = checkpoint["crop_raw"].to(device)
livestock_raw = checkpoint["livestock_raw"].to(device)
rural_raw = checkpoint["rural_raw"].to(device)

k_crop = bounded_param(crop_raw, 0.1, 0.3)
k_livestock = bounded_param(livestock_raw, 0.3, 0.5)
k_rural=bounded_param(rural_raw,0.2,0.5)

# 4. 加载上下游关系矩阵
A_up = np.load(os.path.join(data_path, "A_upstream.npy"))
A_up = torch.from_numpy(A_up).float().to(device)
D_up = np.load(os.path.join(data_path, "D_upstream.npy"))
D_up = torch.from_numpy(D_up).float().to(device)
D_up = torch.where(A_up == 1, D_up, torch.full_like(D_up, 1e6))

# print("lambda_decay:", lambda_decay[:,0,0])
print("D_up mean:", D_up[D_up < 1e5].mean().item())

test = torch.exp(-D_up/1000 / lambda_decay)
print("weight max:", test.max().item())
print("weight min:", test.min().item())


A_wave = torch.from_numpy(get_normalized_adj(A)).to(device)

X_input = X_norm.unsqueeze(0)           # [1,N,F,T]
mask_input = ob_mask_full.unsqueeze(0)

C_full_series = torch.zeros((num_nodes, T_total), device=device)

with torch.no_grad():
    for start in range(T_total - num_timesteps_input + 1):
        end = start + num_timesteps_input

        X_win = X_input[:, :, :, start:end].permute(0,1,3,2)
        mask_win = mask_input[:, :, :, start:end].permute(0,1,3,2)
        # print(X_input[:, :, target_idx, start:end])
        mask_tp=mask_win.squeeze(-1)

        X_masked = X_win.clone()
        X_masked[:, :, :, target_idx] = X_win[:, :, :, target_idx] * mask_tp
        
        C_pred = model(A_wave, X_masked)  # [1,N,T]

        C_obs = X_win[:, :,  :,target_idx]
        
        C_full = C_obs * mask_tp + C_pred * (1 - mask_tp)

        C_full_series[:, start:end] = C_full.squeeze(0)

# ======================
# 7. 反归一化
# ======================
C_full_phys = C_full_series * stds[target_idx] + means[target_idx]

# temp_idx = feature_names.index("Tmpmean")
# prec_idx = feature_names.index("Prec")

# temp = X_norm[:, temp_idx, :]   # (N,T)
# prec = X_norm[:, prec_idx, :]

# 转成 (1,T,N)
# temp = temp.permute(1,0).unsqueeze(0)
# prec = prec.permute(1,0).unsqueeze(0)

# meteo = torch.stack([prec], dim=-1)  # (1,T,N,2)

# ===== MLP预测 =====
# k_all = coeff_mlp(meteo)

# k_crop = bounded_param(k_all[...,0:1], 0.1, 0.3)
# k_livestock = bounded_param(k_all[...,1:2], 0.3, 0.5)
# k_domestic = bounded_param(k_all[...,2:3], 0.8, 1.0)

# 去掉batch维度 → (T,N)
# k_crop_t = k_crop.squeeze(0).squeeze(-1)
# k_livestock_t = k_livestock.squeeze(0).squeeze(-1)
# k_rural_t = k_rural.squeeze(0).squeeze(-1)
k_crop_val = k_crop.item()
k_livestock_val = k_livestock.item()
k_rural_val = k_rural.item()

print("k_crop_val:", k_crop_val)
print("k_livestock_val:", k_livestock_val)
print("k_rural_val:", k_rural_val)

# ====================== 计算逐日各项输入 ======================
# 初始化存储数组（逐日、逐站点）
daily_upstream = np.zeros((T_total, num_nodes))  # 上游来源 [T, N]

daily_industry=(L_industry_daily.squeeze(-1)).detach().cpu().numpy()  # 工业源输入 [T, N]
daily_crop_farming = (k_crop_val * L_crop_farming_daily.squeeze(-1)).detach().cpu().numpy()  # 面源输入 [T, N]
daily_livestock_breeding = (k_livestock_val * L_livestock_breeding_daily.squeeze(-1)).detach().cpu().numpy()
daily_urban = (L_urban_daily.squeeze(-1)).detach().cpu().numpy()     # 城镇生活源输入 [T, N]
daily_rural = (k_rural_val * L_rural_daily.squeeze(-1)).detach().cpu().numpy()

daily_resusp = a_resusp.squeeze(-1).cpu().numpy()    # 再悬浮 [T, N]

with torch.no_grad():
    C_series_phys = C_full_phys.unsqueeze(0)

    # for t in range(T_total - num_timesteps_input + 1):
    t_range = torch.arange(0, T_total, device=device).unsqueeze(0)

    C_series_phys=torch.clamp(C_series_phys,min=0.0)

    flux_up = upstream_mass_flux(C_series_phys, t_range,lambda_decay,Q_node_daily,
                                 A_up, D_up, seconds_per_day)
    
    print("flux_up has nan:", torch.isnan(flux_up).any().item())
    flux_up_no_nan = flux_up[~torch.isnan(flux_up)]

    if flux_up_no_nan.numel() > 0:
        print("flux_up min:", flux_up_no_nan.min().item())
        print("flux_up max:", flux_up_no_nan.max().item())
    else:
        print("All values are NaN!")

    daily_upstream[:, :] = flux_up.squeeze(0).cpu().numpy().T


# ====================== 数据整理与保存 ======================
# 1. 构造日期维度（假设从第0天开始，可根据实际情况替换为真实日期）
dates = [f"Day_{i}" for i in range(T_total)]

# 2. 整理为DataFrame（逐站点、逐日、各项输入）
df_long = []
for node in range(num_nodes):
    for day in range(T_total):
        df_long.append({
            "Station_ID": node,
            "Date": dates[day],
            "Upstream_Input": daily_upstream[day, node],  # 上游来源（吨/天）
            "Industry_Input": daily_industry[day, node],
            "Crop_farming_Input": daily_crop_farming[day, node],  # 面源输入（原始单位）
            "Livestock_breeding_Input": daily_livestock_breeding[day, node],
            "Urban_Input": daily_urban[day, node],
            "Rural_Input": daily_rural[day, node],
            "Resuspension": daily_resusp[day, node]       # 再悬浮（模型学习值）
        })
df_long = pd.DataFrame(df_long)

df_long["Total_Input"] = (
    df_long["Upstream_Input"]
    + df_long["Industry_Input"]
    + df_long["Crop_farming_Input"]
    + df_long["Livestock_breeding_Input"]
    + df_long["Urban_Input"]
    + df_long["Rural_Input"]
    + df_long["Resuspension"]
)

df_long.to_csv(os.path.join(output_path, "1 daily_inputs_long.csv"), index=False)

# ====================== 输出统计信息 ======================
print("="*50)
print("✅ DONE")
print(f"T_total={T_total}, Nodes={num_nodes}")
print(f"Upstream mean={np.mean(daily_upstream):.6f}")
print("="*50)
print('a_susp shape',a_resusp.shape)

T, N = a_resusp.shape[:2]

param_list = []

for node in range(N):
    for time in range(T):
        param_list.append({
            "node": node,
            "time": time,

            "a_resusp": a_resusp[time, node, 0].item(),
            "lambda_decay": lambda_decay[time, node, 0].item(),

            "k_crop": k_crop_val,
            "k_livestock": k_livestock_val,
            "k_rural": k_rural_val,
        })

param_df = pd.DataFrame(param_list)

param_df.to_csv(os.path.join(output_path, "parameter.csv"), index=False)
