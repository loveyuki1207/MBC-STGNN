import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from stgcn import GRU_GCN_Attention
from utils import generate_dataset, load_metr_la_data, get_normalized_adj

# 插补任务参数
num_timesteps_input = 30  # 输入30天（masked）
num_timesteps_output = 30 # 输出30天（插补值）
mask_ratio = 0.3    # 人为mask比例

epochs = 100
batch_size = 32
plot_every=10
random_seed=7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================
# 上下游关系矩阵
# ======================
A_up=np.load('data/A_upstream.npy')
A_up=torch.from_numpy(A_up).float().to(device)

D_up=np.load('data/D_upstream.npy')
D_up=torch.from_numpy(D_up).float().to(device)
D_up = torch.where(A_up==1,D_up,torch.full_like(D_up,1e6))

seconds_per_day=24*3600

#上游质量通量函数（适配插补任务，时间步调整）
def upstream_mass_flux(C_series_phys,t_range,lambda_decay,Q_node_daily,
                       A_up,D_up,seconds_per_day):
    """
    计算上游质量通量（适配30天窗口）
    C_series_phys: [B, N, T]  —— 输入序列的物理值
    t_range: [B,T]          —— 当前窗口起始时间索引
    返回: [B, N, T]    —— 上游质量通量
    """

    # 上游流量（t-1 时刻）
    Qj = Q_node_daily[t_range]   # (B,T,N,1)
    Qj = Qj.permute(0,2,1,3).squeeze(-1)  # → (B,N,T)

    # 浓度转单位 mg/m3
    C_m3 = C_series_phys * 1000

    # 上游原始通量
    flux = Qj[:,:,:-1] * C_m3[:,:,:-1]  # (B,N,T) (t-1)

    lambda_decay=lambda_decay[t_range]

    weight = torch.exp(-D_up.unsqueeze(0).unsqueeze(0)/1000 / lambda_decay)
    weight =weight * A_up.unsqueeze(0).unsqueeze(0)

    # 加权汇总
    flux_up = torch.einsum("btij,bit->bjt", weight[:,1:], flux)
    # pad回T
    flux_up = torch.nn.functional.pad(flux_up, (1,0))  # 前面补0
    flux_up=flux_up *seconds_per_day/1e9 #mg->t

    return flux_up

def bounded_param(x, min_v, max_v):
    y = torch.nn.functional.softplus(x)
    y = y / (y + 1.0)   # 压到 (0,1)，但不饱和
    return min_v + y * (max_v - min_v)

def pde_residual(C_pred,t_idx,lambda_decay,Q_node_daily,
                  A_up, D_up, seconds_per_day):
    """
    插补任务的PDE残差（30天窗口）
    C_pred: 插补预测值 (B,N,T)
    t_idx: 窗口起始时间索引 (B)
    """
    # 反标准化到物理空间
    C_pred_phys = C_pred * stds[target_idx] + means[target_idx]

    # ===== 时间索引（核心替换）=====
    t_range = t_idx.unsqueeze(-1) + torch.arange(num_timesteps_input, device=device)
    # shape: (B, T)
    Qi = Q_node_daily[t_range]   # (B,T,N,1)
    Qi = Qi.permute(0,2,1,3).squeeze(-1)  # (B,N,T)

    # ===== 入河系数（可学习 + 物理约束）=====
    #工业源和城镇生活源不需要设置系数，直接是1
    # k_crop = bounded_param(crop_raw[t_range], 0.1, 0.3)
    k_crop = bounded_param(crop_raw, 0.1, 0.3)
    k_livestock = bounded_param(livestock_raw, 0.3, 0.5)
    k_rural=bounded_param(rural_raw,0.2,0.5)

    L_industry = industry_raw[t_range] * L_industry_daily[t_range]
    L_industry = L_industry.permute(0,2,1,3).squeeze(-1)

    L_crop_farming = k_crop * L_crop_farming_daily[t_range]
    L_crop_farming = L_crop_farming.permute(0,2,1,3).squeeze(-1)

    L_livestock_breeding = k_livestock * L_livestock_breeding_daily[t_range]
    L_livestock_breeding = L_livestock_breeding.permute(0,2,1,3).squeeze(-1)

    L_urban = urban_raw[t_range] * L_urban_daily[t_range]
    L_urban = L_urban.permute(0,2,1,3).squeeze(-1)

    L_rural = k_rural * L_rural_daily[t_range]
    L_rural = L_rural.permute(0,2,1,3).squeeze(-1)

    resusp = a_resusp[t_range]
    resusp = resusp.permute(0,2,1,3).squeeze(-1)

    C_pred_m3=C_pred_phys*1000    #mg/L->mg/m3
    out_flux=Qi*C_pred_m3
    out_flux=out_flux*seconds_per_day/1e9

    upstream=upstream_mass_flux(C_pred_phys,t_range,lambda_decay,Q_node_daily,
                                 A_up, D_up, seconds_per_day)

    residual=(out_flux-upstream-L_industry-L_crop_farming-L_livestock_breeding-L_urban-L_rural-resusp)
    return residual

lambda_pde=0.01
def train_epoch(train_loader,lambda_raw,Q_node_daily,
                 A_up, D_up, seconds_per_day):
    """
    插补任务训练一个epoch（结合mask计算损失）
    """
    net.train()
    epoch_losses=[]

    for X, y, ob_mask, gt_mask, t_idx in train_loader:
        X=X.to(device=device)
        y=y.to(device=device)
        ob_mask = ob_mask.to(device)
        gt_mask = gt_mask.to(device)
        t_idx = t_idx.to(device).long()

        optimizer.zero_grad()

         # ===== mask =====
        cond_mask = gt_mask
        target_mask = ob_mask * (1 - gt_mask)

        # masked 输入
        X_masked=X.clone()
        X_masked[:, :, :, target_idx] *= cond_mask

        # 预测
        C_pred = net(A_wave, X_masked)  # (B,N,T)

        # ===== 数据损失 =====
        loss_data = ((C_pred - y) ** 2 * target_mask).sum() / (target_mask.sum() + 1e-8)
        
        assert torch.all(gt_mask <= ob_mask)

        #构造混合物理场
        C_obs=X[:,:,:,target_idx].detach()   
        C_full = C_obs * cond_mask + C_pred * (1 - cond_mask)

        lambda_decay = torch.nn.functional.softplus(lambda_raw)
        # PDE物理残差
        res=pde_residual(C_full,t_idx,lambda_decay,Q_node_daily,
                          A_up, D_up, seconds_per_day)
        res=res[:,:,1:]
        # mask_pde=target_mask_tp[:,:,1:]

        # loss_pde=(res ** 2 * mask_pde).sum()/(mask_pde.sum()+1e-8)
        loss_pde=torch.mean(res ** 2 )

        # 再悬浮正则
        loss_resusp_reg = torch.mean(a_resusp**2)

        # 可学习参数正则
        lambda_resusp = 1e-4
        lambda_lambda = 1e-4

        loss_lambda_reg = torch.mean(lambda_decay**2)

        # 非负惩罚项
        lambda_nonneg=0.1
        C_phys = C_pred * stds[target_idx] + means[target_idx]

        loss_noneg=torch.mean(torch.nn.functional.relu(-C_phys)**2)

        # 总损失
        loss=(
            loss_data
            + lambda_pde * loss_pde
            + lambda_resusp * loss_resusp_reg
            + lambda_nonneg * loss_noneg
            + lambda_lambda * loss_lambda_reg
        )
        loss.backward()

        optimizer.step()

        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))

if __name__ == '__main__':
    print('using device',device)
    torch.manual_seed(7)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(7)

    # 加载原始数据
    Q_node_daily=torch.tensor(np.load('data/Q_node_daily.npy'),
                        dtype=torch.float32,device=device)  #[T,N,1]
    
    L_industry_daily=torch.tensor(np.load('data/L_industry_daily.npy'),
                                 dtype=torch.float32,device=device)
    L_crop_farming_daily=torch.tensor(np.load('data/L_crop_farming_daily.npy'),
                                 dtype=torch.float32,device=device)
    L_livestock_breeding_daily=torch.tensor(np.load('data/L_livestock_breeding_daily.npy'),
                                 dtype=torch.float32,device=device)
    L_urban_daily=torch.tensor(np.load('data/L_urban_life_daily.npy'),
                                 dtype=torch.float32,device=device)
    L_rural_daily=torch.tensor(np.load('data/L_rural_life_daily.npy'),
                                 dtype=torch.float32,device=device)
    
    # 指定目标特征
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
    temp_idx = feature_names.index("Tmpmean")
    prec_idx = feature_names.index("Prec")

    # 加载数据并生成mask
    A,X_norm,  means, stds,X_original = load_metr_la_data(target_idx)
 
    num_nodes=X_norm.shape[0]
    T_total=X_norm.shape[2]

    # 可学习参数
    a_resusp=nn.Parameter(torch.zeros(T_total,num_nodes,1,device=device))
    # alpha = nn.Parameter(torch.tensor(0.1, device=device))      
    lambda_raw = nn.Parameter(torch.zeros(T_total,num_nodes,1, device=device))  

    # ======================
    # 可学习入河系数（带物理区间约束）
    # ======================
    industry_raw=torch.ones(T_total,num_nodes,1,device=device)  #工业源入河系数固定为1
    # crop_raw = nn.Parameter(torch.zeros( T_total,1, 1, device=device))
    crop_raw = nn.Parameter(torch.tensor( 0.0, device=device))
    livestock_raw = nn.Parameter(torch.tensor(0.0, device=device))
    urban_raw=torch.ones(T_total,num_nodes,1,device=device)     #城镇生活源入河系数固定为1
    rural_raw = nn.Parameter(torch.tensor(0.0, device=device))

    # 划分训练/验证数据（80%训练）
    split_line = int((X_norm.shape[2] - num_timesteps_input + 1) * 0.8)
    X_train = X_norm[:, :, :split_line]
    X_train_original=X_original[:, :, :split_line]
    X_val = X_norm[:, :, split_line - num_timesteps_input:]  # 保证窗口连续
    X_val_original = X_original[:, :, split_line - num_timesteps_input:]
    
    # ======================
    # 分别生成数据集
    # ======================
    train_features, train_target, train_t_idx, train_ob_mask, train_gt_mask = generate_dataset(
        X_train,
        X_train_original,
        num_timesteps_input,
        num_timesteps_output,
        target_idx=target_idx,
        missing_ratio=mask_ratio,
        random_seed=random_seed
    )

    val_features, val_target, val_t_idx, val_ob_mask, val_gt_mask = generate_dataset(
        X_val,
        X_val_original,
        num_timesteps_input,
        num_timesteps_output,
        target_idx=target_idx,
        missing_ratio=mask_ratio,
        random_seed=random_seed
    )

    # ======================
    # DataLoader
    # ======================
    train_dataset = TensorDataset(
        train_features, train_target, train_ob_mask, train_gt_mask, train_t_idx
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        val_features, val_target, val_ob_mask, val_gt_mask, val_t_idx
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 邻接矩阵
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).to(device=device)

    # 初始化模型
    net = GRU_GCN_Attention(
        num_nodes=A_wave.shape[0],
        in_features=train_features.shape[3],
        timesteps_input=num_timesteps_input,
        timesteps_output=num_timesteps_output
    ).to(device=device)

    # 优化器（包含物理参数）
    optimizer=torch.optim.Adam([
        {
            "params": net.parameters(),
            "lr": 0.001,
            "weight_decay": 0.0005
        },
        {
            "params": [crop_raw, livestock_raw, rural_raw],
            "lr": 0.1,
            "weight_decay": 0.0005
        },
        {
            "params": [a_resusp, lambda_raw],
            "lr": 0.001,
            "weight_decay": 0.0005
        }
    ])

    # 记录损失
    training_losses = []
    validation_losses = []
    validation_maes = []

    # 保存路径
    checkpoint_path = "checkpoints/"
    os.makedirs(checkpoint_path, exist_ok=True)
    best_val_loss=float('inf')

    for epoch in range(1,epochs+1):
        # 训练
        train_loss = train_epoch(train_loader,lambda_raw,Q_node_daily,
                                  A_up, D_up, seconds_per_day)
        training_losses.append(train_loss)

        # 验证
        net.eval()
        val_loss_list=[]
        val_mae_list=[]

        with torch.no_grad():
            for Xv, yv, ob_mask_v, gt_mask_v, t_idx_v in val_loader:

                Xv = Xv.to(device)
                yv = yv.to(device)
                ob_mask_v = ob_mask_v.to(device)
                gt_mask_v = gt_mask_v.to(device)

                cond_mask = gt_mask_v
                target_mask = ob_mask_v * (1 - gt_mask_v)

                Xv_masked = Xv.clone()

                Xv_masked[:, :, :, target_idx] *=  cond_mask

                C_pred = net(A_wave, Xv_masked)
                
                loss_data = ((C_pred - yv) ** 2 * target_mask).sum() / (target_mask.sum() + 1e-8)
                val_loss_list.append(loss_data.item())

        val_loss_epoch = float(np.mean(val_loss_list)) 
        validation_losses.append(val_loss_epoch)

        # 打印信息
        print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.6f}, Val Loss: {val_loss_epoch:.6f}")

        # 保存模型
        model_file=os.path.join(checkpoint_path,f'weights/stgcn_interp_epoch_{epoch}.pth')
        torch.save(net.state_dict(),model_file)

        # 保存最佳模型
        if val_loss_epoch<best_val_loss:
            best_val_loss=val_loss_epoch
            best_epoch=epoch
            best_model_file=os.path.join(checkpoint_path,'stgcn_interp_best.pth')
            torch.save({
                "model_state_dict": net.state_dict(),
                "a_resusp": a_resusp.detach().cpu(),
                "lambda_raw": lambda_raw.detach().cpu(),

                "crop_raw": crop_raw.detach().cpu(),
                "livestock_raw": livestock_raw.detach().cpu(),
                "rural_raw": rural_raw.detach().cpu(),

                "means": means,
                "stds": stds,
                "mask_ratio": mask_ratio
            },best_model_file)
            print(f"✅ Best model updated at epoch {epoch} — Val Loss: {val_loss_epoch:.6f}")

        # 绘制损失曲线
        if epoch % plot_every == 0 or epoch == epochs:
            plt.figure(figsize=(8,5))
            plt.plot(training_losses, label="training loss")
            plt.plot(validation_losses, label="validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt_path = os.path.join(checkpoint_path, f"loss_curve_interp_epoch_{epoch}.png")
            plt.savefig(plt_path)
            plt.close()

        # 保存损失记录
        with open(os.path.join(checkpoint_path, "losses_interp.pk"), "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)

        # 清空CUDA缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n🏆 Best interpolation model is from epoch {best_epoch} with Val Loss = {best_val_loss:.6f}")
    print("Training finished. Results saved in:", checkpoint_path)

