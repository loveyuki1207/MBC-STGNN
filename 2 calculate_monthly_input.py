import pandas as pd

# -----------------------------
# 1. 读取每日磷来源数据
# -----------------------------
daily_file = r"STGCN_mass_balance (Imputation)\daily_inputs_results\1 daily_inputs_long.csv"
df_daily = pd.read_csv(daily_file)

# 将Day_0、Day_1...转为真实日期
start_date = pd.to_datetime("2020-11-01")  # 第0天对应的日期
# 提取Day列数字并加上起始日期
df_daily['Date'] = df_daily['Date'].str.replace('Day_', '').astype(int)
df_daily['Date'] = df_daily['Date'].apply(lambda x: start_date + pd.Timedelta(days=x))

# -----------------------------
# 2. 读取站点信息
# -----------------------------
station_file = r"STGCN_mass_balance (Imputation)\predictions_interp\站点信息.csv"
df_station = pd.read_csv(station_file)

# df_station中可能有多个列，取需要的列
df_station = df_station.rename(columns={
    "FID": "Station_ID",
    "Station_Nu": "Station_Nu",
    "Longitude": "Longitude",
    "Latitude": "Latitude"
})
df_station = df_station[['Station_ID', "Station_Nu",'Longitude', 'Latitude']]

# -----------------------------
# 3. 按月汇总磷来源
# -----------------------------
df_daily['YearMonth'] = df_daily['Date'].dt.to_period('M')  # 月份字段
# 按站点和月份求和
df_monthly = df_daily.groupby(['Station_ID', 'YearMonth'])\
    [['Upstream_Input', 'Industry_Input', 'Crop_farming_Input', 
      'Livestock_breeding_Input', 'Urban_Input', 'Rural_Input', 
      'Resuspension', 'Total_Input']].sum().reset_index()

# 将YearMonth恢复成字符串形式 YYYY-MM
df_monthly['YearMonth'] = df_monthly['YearMonth'].astype(str)

# -----------------------------
# 4. 合并站点信息
# -----------------------------
df_result = pd.merge(df_monthly, df_station, on='Station_ID', how='left')

# -----------------------------
# 5. 输出 CSV
# -----------------------------
output_file = r"STGCN_mass_balance (Imputation)\daily_inputs_results\2 phosphorus_monthly.csv"
df_result.to_csv(output_file, index=False)

print(f"生成文件: {output_file}")