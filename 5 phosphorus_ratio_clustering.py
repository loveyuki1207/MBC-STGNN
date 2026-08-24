import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['svg.fonttype'] = 'none'

# 读取数据
file_path = r"STGCN_mass_balance (Imputation)\daily_inputs_results\3 phosphorus_site_average.csv"
df = pd.read_csv(file_path)

station_info=pd.read_csv(r"STGCN_mass_balance (Imputation)\daily_inputs_results\站点信息.csv")

df=df.merge(
    station_info[['FID','Station_Number']],
    left_on='Station_ID',
    right_on='FID',
    how='left'
)

# ===== 构造比例 =====
df["Upstream_ratio"] = df["Upstream_Input"] / df["Total_Input"]
df['Point_ratio'] = (df['Industry_Input']+df['Urban_Input']) / df['Total_Input']
df["Nonpoint_ratio"] = (df["Crop_farming_Input"] + df["Livestock_breeding_Input"]+df['Rural_Input']) / df["Total_Input"]

# 取特征
X = df[["Upstream_ratio","Point_ratio","Nonpoint_ratio"]].fillna(0)

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# ===== 层次聚类 =====
Z = linkage(X_scaled, method='ward')

# ===== 树状图 =====
plt.figure(figsize=(10,5))
dendrogram(Z,labels=df['Station_Number'].values)
# plt.title("Dendrogram")
plt.xlabel("Stations")
plt.ylabel("Distance")

# ===== 保存 =====
plt.savefig(
    "STGCN_mass_balance (Imputation)\daily_inputs_results/phosphorus_ratio_clustering.png",
    dpi=300,              # 分辨率（论文必须300+）
    bbox_inches='tight'   # 👈 关键：防止被裁剪
)

plt.savefig(
    "STGCN_mass_balance (Imputation)\daily_inputs_results/phosphorus_ratio_clustering.svg",
    format="svg",              # 分辨率（论文必须300+）
    bbox_inches='tight'   # 👈 关键：防止被裁剪
)
plt.show()

