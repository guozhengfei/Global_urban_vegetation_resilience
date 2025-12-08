import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime, timedelta
import pyeto
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class UrbanETCalculator:
    """
    使用高分辨率ERA5-Land数据计算城市月尺度ET的类
    模拟OpenET中多种方法的思路
    """

    def __init__(self, era5_data, cities_gdf):
        """
        初始化计算器

        参数:
        era5_data: xarray.Dataset, 高分辨率ERA5-Land数据
        cities_gdf: geopandas.GeoDataFrame, 城市边界数据
        """
        self.era5 = era5_data
        self.cities = cities_gdf
        self.results = {}

    def fao_56_penman_monteith(self, temp_min, temp_max, temp_mean, rh_mean,
                               solar_rad, wind_speed, elevation, lat_rad):
        """
        FAO-56 Penman-Monteith方法 (模拟OpenET中的精密算法)
        参考: Allen et al., 1998
        """
        # 计算饱和水汽压曲线斜率 (Δ)
        delta = 4098 * (0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3))) / ((temp_mean + 237.3) ** 2)

        # 心理测量常数 (γ)
        gamma = 0.000665 * self.era5['sp'] / 1000  # 基于气压计算

        # 净辐射估算 (简化版)
        rn = solar_rad * 0.77 - 50  # 简单的净辐射估算

        # 土壤热通量 (G)
        g = 0.1 * rn if rn > 0 else 0.7 * rn

        # FAO-56 Penman-Monteith公式
        et = (0.408 * delta * (rn - g) +
              gamma * (900 / (temp_mean + 273)) * wind_speed *
              (0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3)) - rh_mean / 100 *
               0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3)))) / \
             (delta + gamma * (1 + 0.34 * wind_speed))

        return np.maximum(et, 0)


    def priestley_taylor_method(self, temp_mean, solar_rad, rh_mean):
        """
        Priestley-Taylor方法 (模拟OpenET中的PT-JPL方法)
        参考: Fisher et al., 2008
        """
        # Priestley-Taylor系数
        alpha = 1.26

        # 饱和水汽压曲线斜率
        delta = 4098 * (0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3))) / ((temp_mean + 237.3) ** 2)

        # 心理测量常数
        gamma = 0.665 * 10 ** -3 * self.era5['sp']

        # 净辐射估算
        rn = solar_rad * 0.77

        # Priestley-Taylor公式
        et = alpha * (delta / (delta + gamma)) * rn * 0.035  # 单位转换系数
        return np.maximum(et, 0)

    def calculate_monthly_et_city(self, city_name, city_geometry, year, month):
        """
        为单个城市计算月尺度ET
        """
        print(f"计算城市 {city_name} {year}-{month} 的ET...")

        try:
            # 提取城市区域的ERA5-Land数据
            city_data = self.era5.sel(
                longitude=slice(city_geometry.bounds[0], city_geometry.bounds[2]),
                latitude=slice(city_geometry.bounds[3], city_geometry.bounds[1])
            )

            if city_data.sizes['longitude'] == 0 or city_data.sizes['latitude'] == 0:
                print(f"警告: 城市 {city_name} 范围内无ERA5-Land数据")
                return None

            # 计算月平均气象数据
            monthly_data = city_data.sel(time=f"{year}-{month:02d}").mean(dim='time')

            # 基本气象变量
            temp_min = monthly_data['t2m_min'].values if 't2m_min' in monthly_data else monthly_data['t2m'].min().values
            temp_max = monthly_data['t2m_max'].values if 't2m_max' in monthly_data else monthly_data['t2m'].max().values
            temp_mean = monthly_data['t2m'].mean().values
            rh_mean = monthly_data['d2m'].mean().values  # 简化处理
            solar_rad = monthly_data['ssrd'].mean().values if 'ssrd' in monthly_data else 15  # 默认值
            wind_speed = monthly_data['u10'].mean().values if 'u10' in monthly_data else 2.0  # 默认值

            # 城市平均纬度（弧度）
            city_centroid = city_geometry.centroid
            lat_rad = np.radians(city_centroid.y)

            # 计算一年中的天数（用于地外辐射计算）
            doy = datetime(year, month, 15).timetuple().tm_yday

            # 应用不同方法计算ET
            et_results = {}

            # 1. FAO-56 Penman-Monteith
            et_results['FAO56_PM'] = self.fao_56_penman_monteith(
                temp_min, temp_max, temp_mean, rh_mean, solar_rad, wind_speed, 100, lat_rad)

            # 2. Hargreaves
            et_results['Hargreaves'] = self.hargreaves_method(
                temp_min, temp_max, temp_mean, lat_rad, doy)

            # 3. Priestley-Taylor
            et_results['Priestley_Taylor'] = self.priestley_taylor_method(
                temp_mean, solar_rad, rh_mean)

            return et_results

        except Exception as e:
            print(f"计算城市 {city_name} 时出错: {e}")
            return None

    def calculate_global_urban_et(self, year, month, city_subset=None):
        """
        计算全球城市月尺度ET
        """
        target_cities = self.cities
        if city_subset is not None:
            target_cities = target_cities.head(city_subset)

        monthly_results = {}

        for idx, city in target_cities.iterrows():
            city_name = city['name'] if 'name' in city.columns else f"City_{idx}"
            city_geometry = city.geometry

            et_results = self.calculate_monthly_et_city(city_name, city_geometry, year, month)

            if et_results is not None:
                monthly_results[city_name] = {
                    'geometry': city_geometry,
                    'et_methods': et_results,
                    'avg_et': np.mean(list(et_results.values()))
                }

        self.results[f"{year}-{month:02d}"] = monthly_results
        return monthly_results

    def visualize_results(self, year, month, method='avg_et'):
        """
        可视化ET结果
        """
        if f"{year}-{month:02d}" not in self.results:
            print("无可用结果进行可视化")
            return

        results = self.results[f"{year}-{month:02d}"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # 各方法ET值分布
        methods = ['FAO56_PM', 'Hargreaves', 'Priestley_Taylor']
        et_values = [[] for _ in range(len(methods))]
        city_names = []
        avg_et_values = []

        for city_name, data in results.items():
            city_names.append(city_name)
            avg_et_values.append(data['avg_et'])
            for i, method in enumerate(methods):
                if method in data['et_methods']:
                    et_values[i].append(data['et_methods'][method])

        # 方法比较箱型图
        for i, method in enumerate(methods):
            if et_values[i]:
                axes[0].boxplot(et_values[i], positions=[i], labels=[method])
        axes[0].set_title('各方法ET值分布比较')
        axes[0].set_ylabel('ET (mm/day)')

        # 城市平均ET排名
        sorted_indices = np.argsort(avg_et_values)[::-1]
        sorted_cities = [city_names[i] for i in sorted_indices[:10]]
        sorted_et = [avg_et_values[i] for i in sorted_indices[:10]]

        axes[1].bar(range(len(sorted_cities)), sorted_et)
        axes[1].set_xticks(range(len(sorted_cities)))
        axes[1].set_xticklabels(sorted_cities, rotation=45)
        axes[1].set_title('ET最高的10个城市')
        axes[1].set_ylabel('平均ET (mm/day)')

        # 方法间相关性
        import seaborn as sns
        correlation_data = {}
        for i, method in enumerate(methods):
            if et_values[i]:
                correlation_data[method] = et_values[i]

        if len(correlation_data) > 1:
            corr_df = pd.DataFrame(correlation_data)
            sns.heatmap(corr_df.corr(), annot=True, ax=axes[2])
            axes[2].set_title('方法间相关性')

        plt.tight_layout()
        plt.show()


# 示例使用
def main():
    """
    主函数示例
    """
    # 1. 加载ERA5-Land数据 (示例)
    # 假设era5_data是已经加载的xarray Dataset
    # 包含变量: t2m(温度), d2m(露点), ssrd(辐射), u10/v10(风速)等

    # 2. 加载城市边界数据
    # 可以使用geopandas读取shapefile或GeoJSON
    world_cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

    # 3. 初始化ET计算器
    et_calculator = UrbanETCalculator(era5_data, world_cities)

    # 4. 计算特定年月全球城市ET (先测试前5个城市)
    results = et_calculator.calculate_global_urban_et(2023, 7, city_subset=5)

    # 5. 可视化结果
    et_calculator.visualize_results(2023, 7)

    # 6. 保存结果
    output_df = pd.DataFrame.from_dict({
        city: data for city, data in results.items()
    }, orient='index')
    output_df.to_csv(f'global_urban_et_202307.csv')


if __name__ == "__main__":
    main()