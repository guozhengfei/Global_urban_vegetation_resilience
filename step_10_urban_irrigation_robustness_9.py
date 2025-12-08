import math
from datetime import datetime


def calculate_monthly_et(t_mean, t_max, t_min, Rs, Rnl,rh_mean, u2, altitude):
    """
    使用FAO Penman-Monteith方程计算月参考蒸散量(ET0)

    参数:
    t_mean -- 月平均气温(摄氏度)
    t_max -- 月最高气温(摄氏度)
    t_min -- 月最低气温(摄氏度)
    rh_mean -- 月平均相对湿度(%)
    u2 -- 2米高处月平均风速(m/s)
    altitude -- 站点海拔高度(米)

    返回:
    et0_monthly -- 月参考蒸散量(mm/month)
    """
    # 1. 计算饱和水汽压(kPa)
    e0_tmax = 0.6108 * math.exp((17.27 * t_max) / (t_max + 237.3))
    e0_tmin = 0.6108 * math.exp((17.27 * t_min) / (t_min + 237.3))
    es = (e0_tmax + e0_tmin) / 2

    # 2. 计算实际水汽压(kPa)
    ea = (rh_mean / 100) * es

    # 3. 计算饱和水汽压曲线斜率(kPa/°C)
    delta = 4098 * (0.6108 * math.exp((17.27 * t_mean) / (t_mean + 237.3))) / ((t_mean + 237.3) ** 2)

    # 4. 计算大气压(kPa)
    P = 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26

    # 5. 计算心理常数(kPa/°C)
    gamma = 0.000665 * P

    # 7. 计算净辐射(MJ/m²/day)
    Rns = 0.77 * Rs  # 净短波辐射
    Rn = Rns + Rnl
    # 8. 计算土壤热通量(G) - 月尺度通常忽略
    G = 0

    # 9. 计算Penman-Monteith ET0 (mm/day)
    et0_daily = (0.408 * delta * (Rn - G) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)) / (
                delta + gamma * (1 + 0.34 * u2))

    # 10. 转换为月尺度(mm/month)
    et0_monthly = et0_daily * 30
    return et0_monthly


# 测试用例
if __name__ == "__main__":
    # 示例数据 (北京气象站)
    t_mean = 25.5  # 月平均气温 (°C)
    t_max = 30.2  # 月最高气温 (°C)
    t_min = 20.8  # 月最低气温 (°C)
    rh_mean = 65  # 月平均相对湿度 (%)
    u2 = 2.5  # 2米高月平均风速 (m/s)
    n = 7.2  # 月实际日照时数 (小时/天)
    altitude = 43.5  # 海拔 (m)
    lat = 39.9  # 纬度 (°)
    month = 7  # 7月
    year = 2023  # 年份

    et0 = calculate_monthly_et(t_mean, t_max, t_min, rh_mean, u2, n, altitude, lat, month, year)
    print(f"计算得到的月参考蒸散量(ET0): {et0:.2f} mm/month")
