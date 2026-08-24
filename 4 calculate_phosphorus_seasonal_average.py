# calculate_phosphorus_seasonal_average.py

import pandas as pd

# -----------------------------
# 1. 读取月度磷数据
# -----------------------------
input_file = r"STGCN_mass_balance (Imputation)\daily_inputs_results\2 phosphorus_monthly.csv"
df_monthly = pd.read_csv(input_file)

# -----------------------------
# 2. 提取月份 + 定义季节
# -----------------------------
# 假设 date 列是类似 "2020-05-01"
df_monthly['YearMonth'] = pd.to_datetime(df_monthly['YearMonth'], format='%Y-%m')

df_monthly['month'] = df_monthly['YearMonth'].dt.month

# 定义季节
def get_season(month):
    if month in [5, 6, 7, 8, 9, 10]:
        return 'Wet'
    else:
        return 'Dry'

df_monthly['Season'] = df_monthly['month'].apply(get_season)

# -----------------------------
# 3. 分别计算干季 / 湿季平均
# -----------------------------
cols = [
    'Upstream_Input',
    'Industry_Input',
    'Crop_farming_Input',
    'Livestock_breeding_Input',
    'Urban_Input',
    'Rural_Input',
    'Resuspension',
    'Total_Input'
]

season_avg = df_monthly.groupby(['Station_ID', 'Season'])[cols].mean().reset_index()

# -----------------------------
# 4. 保留经纬度
# -----------------------------
site_info = df_monthly[['Station_ID','Longitude','Latitude']].drop_duplicates()
season_avg = pd.merge(season_avg, site_info, on='Station_ID', how='left')

# -----------------------------
# 5. 拆成两个表
# -----------------------------
dry_avg = season_avg[season_avg['Season'] == 'Dry']
wet_avg = season_avg[season_avg['Season'] == 'Wet']

# -----------------------------
# 6. 输出
# -----------------------------
dry_avg.to_csv(r"STGCN_mass_balance (Imputation)\daily_inputs_results\4 phosphorus_dry_season.csv", index=False)
wet_avg.to_csv(r"STGCN_mass_balance (Imputation)\daily_inputs_results\4 phosphorus_wet_season.csv", index=False)
