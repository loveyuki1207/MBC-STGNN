# calculate_average_phosphorus.py

import pandas as pd

# -----------------------------
# 1. 读取月度磷数据
# -----------------------------
input_file = r"STGCN_mass_balance (Imputation)\daily_inputs_results\2 phosphorus_monthly.csv"
df_monthly = pd.read_csv(input_file)

# -----------------------------
# 2. 计算每个站点各磷来源的平均值
# -----------------------------
# 按站点分组，对磷来源列求平均
site_avg = df_monthly.groupby('Station_ID')\
    [['Upstream_Input','Industry_Input','Crop_farming_Input',
      'Livestock_breeding_Input','Urban_Input', 'Rural_Input', 
      'Resuspension','Total_Input']].mean().reset_index()

# 保留站点经纬度信息，方便查看
site_info = df_monthly[['Station_ID','Longitude','Latitude']].drop_duplicates()
site_avg = pd.merge(site_avg, site_info, on='Station_ID', how='left')

# -----------------------------
# 3. 输出站点平均值 CSV
# -----------------------------
site_avg.to_csv(r"STGCN_mass_balance (Imputation)\daily_inputs_results\3 phosphorus_site_average.csv", index=False)
print("生成文件: phosphorus_site_average.csv")