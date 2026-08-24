import numpy as np
import torch
import os
import zipfile


def load_metr_la_data(target_idx):
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")  # adjacency
    X = np.load("data/node_values.npy").transpose((1, 2, 0))  # (N,F,T)
    X = X.astype(np.float32)

    # ======================
    # 将 target_idx 中为0的值置为 NaN
    # ======================
    target_data = X[:, target_idx, :]
    mask = (target_data == 0)
    target_data[mask] = np.nan
    X[:, target_idx, :] = target_data

    X_original=X.copy()

    # 归一化（Z-score, 保留原始缺失值不参与计算）
    means = np.nanmean(X, axis=(0, 2))
    stds = np.nanstd(X, axis=(0, 2))

    X = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)

    return A, X, means, stds,X_original


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    """
    A[source, target] = 1
    source = upstream
    target = downstream

    GCN requires:
    A_msg[target, source] = 1
    """

    A = A.T  # 转置，保证A[target, source] = 1
    # A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  暂时不加对角线，数据中已经存在对角线
    #注意axis的数值，原始的数据行是下游，也就是被影响的数据，而我的数据行是上游，是源节点
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))

    return A_wave


def generate_dataset(X,X_original,  num_timesteps_input, num_timesteps_output,
                      target_idx,missing_ratio, random_seed=7,stride=1):
    """
    1. 生成原始观测掩码（标记非NaN位置）
    2. 对观测值随机遮蔽指定比例，生成人为缺失掩码
    :param X: 节点特征 (num_vertices, num_features, num_timesteps)
    :param missing_ratio: 人为遮蔽的观测值比例
    :param random_seed: 随机种子（保证可复现）
    :return: 新增 data_ob_masks/data_gt_masks 掩码张量
    """
    # 1. 生成原始观测掩码（标记非NaN的位置，True=观测到/非缺失，False=原始缺失）
    tp_ob_mask = ~np.isnan(X[:,target_idx,:])  # (num_vertices, num_timesteps)

    # tp_ob_mask = tp_ob_mask & (X_original[:, target_idx, :] != 0)
    # 填充原始NaN值（避免后续计算报错）
    X = np.nan_to_num(X, nan=0.0)

    # 固定随机种子
    np.random.seed(random_seed)

    # 2. 生成滑动窗口索引（保留原逻辑）
    indices = [(i, i + (num_timesteps_input)) for i
               in range(0, X.shape[2] - (num_timesteps_input) + 1,stride)]

    # 3. 保存样本和掩码
    features, target, t_idx = [], [], []
    data_ob_masks, data_gt_masks = [], []  # 新增掩码列表

    for i, j in indices:
        # -------- 处理输入特征 --------
        # 截取输入窗口特征 (num_vertices, num_features, num_timesteps_input)
        feat = X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1))

        # -------- 处理观测掩码 --------
        # 截取输入窗口的原始观测掩码
        ob_mask = tp_ob_mask[:,  i: i + num_timesteps_input]
        
        if np.sum(ob_mask) == 0:
            continue

        # ===== 人为 mask（只针对 TP）=====
        gt_mask = ob_mask.copy()

        # ==================================================
        # ✅ 核心修改：只对 TP 做 mask
        # ==================================================
        tp_mask_flat = gt_mask.reshape(-1)

        # 只选择原始观测到的位置（True）进行随机遮蔽
        obs_indices = np.where(tp_mask_flat)[0]

        if len(obs_indices) > 0:
            miss_indices = np.random.choice(
                obs_indices,
                int(len(obs_indices) * missing_ratio),
                replace=False
            )
            tp_mask_flat[miss_indices] = False

        # reshape 回去
        gt_mask = tp_mask_flat.reshape(gt_mask.shape)
        
        # ===== 保存 =====
        features.append(feat)
        data_ob_masks.append(ob_mask)
        data_gt_masks.append(gt_mask)

        # ===== 目标（TP）=====
        targ = X[:, target_idx, i:i + num_timesteps_input]
        target.append(targ)

        t_idx.append(i)
        
    # 4. 转换为张量并返回（新增掩码返回值）
    return (
        torch.from_numpy(np.array(features)),
        torch.from_numpy(np.array(target)),
        torch.tensor(t_idx, dtype=torch.long),
        torch.from_numpy(np.array(data_ob_masks, dtype=np.float32)),  # 原始观测掩码
        torch.from_numpy(np.array(data_gt_masks, dtype=np.float32))   # 人为缺失掩码
    )
