#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
哈尔滨冰雪大世界 — 多业态融合仿真数据生成器
==============================================================================
用途：为"多业态融合智慧文旅管控系统"生成仿真运营数据，支持 AI 客流预测
      模型的训练与验证。

生成内容：
  1. data/tourist_flow.csv    — 每日客流、时段分布、预售票量、天气、节假日标签
  2. data/revenue.csv          — 各业态营收（门票、酒店、餐饮、零售）
  3. data/reviews.csv          — 舆情评论样本（正面/负面/中性 + 文本内容）
  4. data/travel_agency.csv    — 旅行社数据（团队数量、人数、来源地）
  5. data/queue_times.csv      — 各项目排队时长

季节特征：
  - 旺季：11月 — 3月（日均 8000~35000 人）
  - 峰值：12月24日 — 2月15日（春节前后，日均 20000~50000 人）
  - 淡季：4月 — 10月（园区关闭或极低客流）

兼容性：Python 3.8+
作者：智慧城市大赛团队
日期：2026-09
==============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ========================= 全局随机种子 =========================
# 保证每次生成的数据可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========================= 输出路径 =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ========================= 参数配置 =========================
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"           # 2 年共 730 天
NUM_DAYS = 730

# 哈尔滨大致经纬度（用于天气模拟的季节参考）
HARBIN_LAT = 45.75

# ========================= 工具函数 =========================

def is_weekend(d):
    """判断是否为周末（周六/周日）"""
    return 1 if d.weekday() >= 5 else 0


def is_spring_festival(d):
    """
    判断是否为春节假期区间（简化处理）。
    2024 年春节：2月10日（正月初一）；2025 年春节：1月29日（正月初一）
    假期按除夕前一两天到初七左右计算，峰值客流还会延续到元宵前后。
    """
    y = d.year
    if y == 2024:
        start = datetime(2024, 2, 8)
        end = datetime(2024, 2, 24)    # 到元宵
    elif y == 2025:
        start = datetime(2025, 1, 27)
        end = datetime(2025, 2, 12)
    else:
        return 0
    return 1 if start <= d <= end else 0


def is_national_day(d):
    """国庆假期 10.1 — 10.7"""
    m, day = d.month, d.day
    return 1 if (m == 10 and 1 <= day <= 7) else 0


def is_new_year(d):
    """元旦假期 1.1 — 1.3"""
    m, day = d.month, d.day
    return 1 if (m == 1 and 1 <= day <= 3) else 0


def is_qingming(d):
    """清明节 4.4 — 4.6"""
    m, day = d.month, d.day
    return 1 if (m == 4 and 4 <= day <= 6) else 0


def is_labor_day(d):
    """劳动节 5.1 — 5.5"""
    m, day = d.month, d.day
    return 1 if (m == 5 and 1 <= day <= 5) else 0


def is_mid_autumn(d):
    """
    中秋节（简化日期）。
    2024 年：9月17日；2025 年：10月6日（与国庆重合）
    """
    if d.year == 2024 and d.month == 9 and 15 <= d.day <= 19:
        return 1
    if d.year == 2025 and d.month == 10 and 4 <= d.day <= 8:
        return 1
    return 0


def is_dragon_boat(d):
    """端午节（简化日期）。2024：6.10；2025：5.31"""
    if d.year == 2024 and d.month == 6 and 7 <= d.day <= 13:
        return 1
    if d.year == 2025 and d.month == 5 and 28 <= d.day <= 31:
        return 1
    return 0


def is_holiday(d):
    """综合节假日判断（任意一个节日即为1）"""
    return int(any([
        is_spring_festival(d),
        is_national_day(d),
        is_new_year(d),
        is_qingming(d),
        is_labor_day(d),
        is_mid_autumn(d),
        is_dragon_boat(d),
    ]))


def is_peak_season(d):
    """
    峰值季：12月24日 — 2月15日
    这是冰雪大世界客流的绝对高峰期（圣诞 + 元旦 + 春节）
    """
    m, day = d.month, d.day
    if m == 12 and day >= 24:
        return 1
    if m == 1:
        return 1
    if m == 2 and day <= 15:
        return 1
    return 0


def is_operating_season(d):
    """
    运营季：11月 — 3月（冰雪大世界开放月份）
    其余月份园区基本关闭
    """
    m = d.month
    return 1 if m in [11, 12, 1, 2, 3] else 0


# ========================= 1. 主数据生成 =========================

def generate_tourist_flow():
    """
    生成每日客流主数据 (data/tourist_flow.csv)

    列：
      date, visitors, is_weekend, is_holiday, is_peak_season,
      is_operating_season, temp_high, temp_low, precipitation, wind_speed,
      pre_sale_tickets, hour_0 ~ hour_23

    思路：
      1. 基础客流 = 季节性基线 × 星期因子 × 节假日因子 × 噪声
      2. 天气 = 正弦年周期 + 日随机波动
      3. 预售票 ≈ 客流 × 0.25~0.45（存在随机提前量）
      4. 时段分布 = 根据季节不同的 24 小时比例
    """
    print("[1/5] 正在生成每日客流数据...")

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(NUM_DAYS)]
    records = []

    for d in dates:
        m = d.month
        weekday = d.weekday()          # 0=Mon ... 6=Sun
        year_day = d.timetuple().tm_yday

        # ---- 1a. 天气模拟 ----
        # 哈尔滨冬季严寒、夏季温和。使用正弦函数模拟年平均温度。
        # 年周期最低温在1月中旬（约 day 15），最高温在7月中旬（约 day 196）
        # temp_avg: 年均约 4°C，振幅约 33°C（-29 ~ 37）
        temp_low_base = -18 - 17 * np.cos(2 * np.pi * (year_day - 15) / 365)
        temp_high_base = -8 - 16 * np.cos(2 * np.pi * (year_day - 15) / 365)
        temp_low = temp_low_base + np.random.normal(0, 3)
        temp_high = temp_high_base + np.random.normal(0, 3)
        temp_low = np.clip(temp_low, -38, 28)
        temp_high = np.clip(temp_high, -28, 38)

        # 降水：夏季降水多，冬季降雪（折算为 mm 等效水量）
        precip_base = 2 + 4 * (1 + np.cos(2 * np.pi * (year_day - 196) / 365)) / 2
        if m in [12, 1, 2]:
            precip_base *= 0.6     # 冬季降雪量较少
        precipitation = max(0, np.random.exponential(precip_base))

        # 风速 (m/s)：冬季风大
        wind_base = 3 + 1.5 * (1 - np.cos(2 * np.pi * (year_day - 15) / 365)) / 2
        wind_speed = max(0, np.random.normal(wind_base, 1.2))

        # ---- 1b. 客流模拟 ----
        if not is_operating_season(d):
            visitors = 0
        else:
            if is_peak_season(d):
                base = 22000
            elif m == 11:
                base = 8000        # 11月刚开业
            elif m == 3:
                base = 10000       # 3月接近闭园
            else:
                base = 0

            # 星期因子：周末 ×1.35
            weekend_factor = 1.35 if is_weekend(d) else 1.0

            # 节假日因子
            if is_spring_festival(d):
                holiday_factor = 1.8
            elif is_national_day(d):
                holiday_factor = 1.0    # 国庆非冰雪旺季
            elif is_new_year(d):
                holiday_factor = 1.5
            elif is_labor_day(d) or is_qingming(d):
                holiday_factor = 1.0
            elif is_mid_autumn(d) or is_dragon_boat(d):
                holiday_factor = 1.0
            else:
                holiday_factor = 1.0

            # 天气影响：极端低温(-25°C以下)或大风天减少客流
            weather_penalty = 1.0
            if temp_high < -22:
                weather_penalty -= 0.15
            if temp_low < -30:
                weather_penalty -= 0.10
            if wind_speed > 7:
                weather_penalty -= 0.10
            weather_penalty = max(weather_penalty, 0.55)

            # 峰值季内的增长趋势（越靠近春节越高）
            if is_peak_season(d):
                # 12月24日起逐渐攀升
                days_since_peak_start = (d - datetime(d.year, 12, 24)).days
                if days_since_peak_start < 0:
                    days_since_peak_start = (d - datetime(d.year - 1, 12, 24)).days
                peak_ramp = min(1.0, days_since_peak_start / 20.0)
                if d.month == 2:
                    peak_ramp = max(0.6, 1.0 - (d.day / 28.0) * 0.5)
                base *= (0.5 + 0.7 * peak_ramp)

            visitors = int(base * weekend_factor * holiday_factor * weather_penalty)
            visitors += int(np.random.normal(0, visitors * 0.08))  # 8% 随机噪声
            visitors = max(0, visitors)

        # ---- 1c. 预售票量 ----
        if visitors > 0:
            pre_sale_ratio = np.random.uniform(0.25, 0.45)
            pre_sale_tickets = int(visitors * pre_sale_ratio)
            # 节假日预售更多
            if is_holiday(d) or is_weekend(d):
                pre_sale_tickets = int(pre_sale_tickets * np.random.uniform(1.1, 1.3))
        else:
            pre_sale_tickets = 0

        # ---- 1d. 时段客流分布（24小时比例） ----
        if visitors > 0:
            if is_peak_season(d):
                # 峰值季：上午10点开门，下午2-6点高峰，晚上8-10点次高峰（冰灯夜景）
                hour_weights = np.array([
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  # 0-7
                    0.005, 0.015, 0.040, 0.065, 0.085, 0.100, 0.115, 0.120,   # 8-15
                    0.110, 0.100, 0.080, 0.060, 0.045, 0.030, 0.020, 0.010,   # 16-23
                ])
            else:
                # 平季：集中于 10:00-18:00
                hour_weights = np.array([
                    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                    0.005, 0.020, 0.055, 0.090, 0.110, 0.120, 0.115, 0.100,
                    0.090, 0.075, 0.065, 0.050, 0.040, 0.030, 0.020, 0.015,
                ])
            # 添加一定随机波动
            hour_weights = hour_weights + np.random.uniform(-0.008, 0.008, 24)
            hour_weights = np.clip(hour_weights, 0, None)
            hour_weights = hour_weights / hour_weights.sum()
            hour_counts = (visitors * hour_weights).astype(int)
            # 修正由于取整带来的偏差
            diff = visitors - hour_counts.sum()
            hour_counts[np.argmax(hour_weights)] += diff
        else:
            hour_counts = np.zeros(24, dtype=int)

        # ---- 1e. 组装记录 ----
        record = {
            "date": d.strftime("%Y-%m-%d"),
            "visitors": visitors,
            "is_weekend": is_weekend(d),
            "is_holiday": is_holiday(d),
            "is_peak_season": is_peak_season(d),
            "is_operating_season": is_operating_season(d),
            "temp_high": round(temp_high, 1),
            "temp_low": round(temp_low, 1),
            "precipitation": round(precipitation, 2),
            "wind_speed": round(wind_speed, 1),
            "pre_sale_tickets": pre_sale_tickets,
        }
        for h in range(24):
            record[f"hour_{h}"] = hour_counts[h]

        records.append(record)

    df = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "tourist_flow.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    -> 已保存 {output_path} ({len(df)} 行)")
    return df


# ========================= 2. 营收数据 =========================

def generate_revenue(flow_df):
    """
    生成各业态营收数据 (data/revenue.csv)

    列：
      date, ticket_revenue, hotel_occupancy_rate, hotel_revenue,
      dining_revenue, retail_revenue, total_revenue

    业态关系：
      - 门票：visitors × 均价 180 元（淡旺季浮动）
      - 酒店入住率：与客流正相关，峰值季 85%-95%
      - 餐饮消费：visitors × 人均 60~120 元
      - 零售消费：visitors × 人均 40~100 元（纪念品等）
    """
    print("[2/5] 正在生成各业态营收数据...")

    records = []
    for _, row in flow_df.iterrows():
        d = datetime.strptime(row["date"], "%Y-%m-%d")
        visitors = int(row["visitors"])
        is_peak = int(row["is_peak_season"])
        is_operating = int(row["is_operating_season"])

        if visitors == 0:
            records.append({
                "date": row["date"],
                "ticket_revenue": 0,
                "hotel_occupancy_rate": 0.0,
                "hotel_revenue": 0,
                "dining_revenue": 0,
                "retail_revenue": 0,
                "total_revenue": 0,
            })
            continue

        # 门票均价（旺季上浮）
        if is_peak:
            ticket_price = np.random.uniform(260, 330)
        elif is_operating:
            ticket_price = np.random.uniform(150, 220)
        else:
            ticket_price = 0
        ticket_revenue = int(visitors * ticket_price)

        # 酒店入住率
        base_occupancy = 0.35
        if is_peak:
            base_occupancy = 0.85 + np.random.uniform(-0.08, 0.10)
        elif is_operating:
            base_occupancy = 0.45 + 0.30 * (visitors / 35000)
        else:
            base_occupancy = 0.08
        occupancy = np.clip(base_occupancy + np.random.uniform(-0.03, 0.03), 0, 1)
        occupancy = round(occupancy, 4)

        # 酒店营收（假设周边合作酒店共 2000 间房，均价 350~600 元/间）
        room_price = np.random.uniform(450, 650) if is_peak else np.random.uniform(280, 420)
        hotel_revenue = int(2000 * occupancy * room_price)

        # 餐饮消费（人均）
        dining_per_capita = np.random.uniform(80, 150) if is_peak else np.random.uniform(40, 90)
        dining_revenue = int(visitors * dining_per_capita)

        # 零售消费（人均）
        retail_per_capita = np.random.uniform(50, 120) if is_peak else np.random.uniform(25, 70)
        retail_revenue = int(visitors * retail_per_capita)

        total_revenue = ticket_revenue + hotel_revenue + dining_revenue + retail_revenue

        records.append({
            "date": row["date"],
            "ticket_revenue": ticket_revenue,
            "hotel_occupancy_rate": occupancy,
            "hotel_revenue": hotel_revenue,
            "dining_revenue": dining_revenue,
            "retail_revenue": retail_revenue,
            "total_revenue": total_revenue,
        })

    df = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "revenue.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    -> 已保存 {output_path} ({len(df)} 行)")
    return df


# ========================= 3. 舆情评论 =========================

def generate_reviews(flow_df):
    """
    生成舆情评论样本 (data/reviews.csv)

    列：
      date, rating, sentiment, review_text

    评论生成规则：
      - 大致 60% 正面 / 25% 中性 / 15% 负面
      - 旺季评论量 ≈ visitors × 0.003
      - 极端天气或排队过长 → 负评概率增加
      - 评论文本从模板库中随机拼凑
    """
    print("[3/5] 正在生成舆情评论数据...")

    # ---- 评论模板库 ----
    pos_templates = [
        "太美了！冰雕非常震撼，下次还来。",
        "带孩子来的，小朋友玩得很开心，推荐！",
        "夜场的冰灯特别漂亮，像童话世界一样。",
        "虽然冷但值得，哈尔滨的冬天名不虚传。",
        "大滑梯超刺激，排队时间也在可接受范围内。",
        "今年冰雕规模比去年更大了，很壮观。",
        "服务态度很好，园区管理有序。",
        "冰上龙舟很有创意，适合全家一起玩。",
        "极地馆的企鹅太可爱了，拍了好多照片。",
        "冰酒吧体验独特，零下30度喝热饮很特别。",
        "冰雪大世界的灯光秀太震撼了，音乐配合得绝佳！",
        "适合拍照打卡，随手一拍都是大片。",
        "主办方今年安排得很用心，动线合理不绕路。",
        "虽然人多但是氛围感很棒，冰雪的快乐。",
        "冰滑梯排队虽久但值得，尖叫声不断。",
    ]

    neutral_templates = [
        "总体还行，就是排队太久了。",
        "票价有点贵，不过冰雕确实不错。",
        "人太多了，建议工作日去。",
        "园区很大，走得腿都酸了，建议多设休息点。",
        "冰雕很美，但是餐厅价格偏高。",
        "中规中矩的一次体验，和预期差不多。",
        "白天看一般，晚上亮灯后效果才好。",
        "停车场离入口有点远，走了好久。",
    ]

    neg_templates = [
        "排队排了2个小时，体验很差。",
        "太冷了，园区取暖设施不够。",
        "餐饮又贵又难吃，建议自带食物。",
        "人挤人，根本没心情看冰雕。",
        "门票太贵了，性价比低。",
        "地上太滑了，差点摔倒好几次。",
        "工作人员态度不好，问路都不理。",
        "冰雕有些已经开始融化，维护不及时。",
        "卫生间太少了，排队比看冰雕还久。",
        "停车费贵得离谱，而且不好找车位。",
    ]

    records = []
    for _, row in flow_df.iterrows():
        visitors = int(row["visitors"])
        if visitors == 0:
            continue

        # 当日评论数（约为客流的 0.2%~0.5%）
        review_count = max(1, int(visitors * np.random.uniform(0.002, 0.005)))

        # 当天天气是否恶劣（用于调整情感分布）
        bad_weather = (float(row["temp_high"]) < -22) or (float(row["wind_speed"]) > 7)

        for _ in range(review_count):
            # 情感分布
            if bad_weather:
                prob = [0.45, 0.28, 0.27]
            else:
                prob = [0.60, 0.25, 0.15]
            sentiment = np.random.choice(["positive", "neutral", "negative"], p=prob)

            if sentiment == "positive":
                text = random.choice(pos_templates)
                rating = np.random.randint(4, 6)
            elif sentiment == "neutral":
                text = random.choice(neutral_templates)
                rating = 3
            else:
                text = random.choice(neg_templates)
                rating = np.random.randint(1, 3)

            records.append({
                "date": row["date"],
                "rating": rating,
                "sentiment": sentiment,
                "review_text": text,
            })

    # 限制总评论数，避免文件过大
    if len(records) > 15000:
        records = random.sample(records, 15000)

    df = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "reviews.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    -> 已保存 {output_path} ({len(df)} 行)")
    return df


# ========================= 4. 旅行社数据 =========================

def generate_travel_agency(flow_df):
    """
    生成旅行社数据 (data/travel_agency.csv)

    列：
      date, agency_count, total_group_people, avg_group_size,
      top_origin_1, top_origin_2, top_origin_3

    来源地（国内主要客源城市 / 省份）：
      北京、上海、广东、浙江、江苏、四川、辽宁、吉林、山东、内蒙古
    """
    print("[4/5] 正在生成旅行社数据...")

    origins = ["北京", "上海", "广东", "浙江", "江苏", "四川",
               "辽宁", "吉林", "山东", "内蒙古", "天津", "河北",
               "河南", "湖北", "湖南", "福建", "安徽", "陕西"]

    records = []
    for _, row in flow_df.iterrows():
        d = datetime.strptime(row["date"], "%Y-%m-%d")
        visitors = int(row["visitors"])
        is_peak = int(row["is_peak_season"])
        is_hol = int(row["is_holiday"])

        if visitors == 0:
            records.append({
                "date": row["date"],
                "agency_count": 0,
                "total_group_people": 0,
                "avg_group_size": 0,
                "top_origin_1": "",
                "top_origin_2": "",
                "top_origin_3": "",
            })
            continue

        # 旅行社团队客约占总客流的 15%~35%
        group_ratio = np.random.uniform(0.15, 0.35)
        if is_peak:
            group_ratio += 0.08
        total_group_people = int(visitors * group_ratio)

        # 平均团规模 15~45 人
        avg_size = np.random.uniform(15, 45)
        agency_count = max(1, int(total_group_people / avg_size))

        # 随机选前三大客源地
        top3 = random.sample(origins, min(3, len(origins)))

        records.append({
            "date": row["date"],
            "agency_count": agency_count,
            "total_group_people": total_group_people,
            "avg_group_size": round(avg_size, 1),
            "top_origin_1": top3[0],
            "top_origin_2": top3[1] if len(top3) > 1 else "",
            "top_origin_3": top3[2] if len(top3) > 2 else "",
        })

    df = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "travel_agency.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    -> 已保存 {output_path} ({len(df)} 行)")
    return df


# ========================= 5. 排队时长数据 =========================

def generate_queue_times(flow_df):
    """
    生成各项目排队时长数据 (data/queue_times.csv)

    列：
      date, ice_slide_wait, ice_bar_wait, ice_sculpture_walk_wait,
      polar_pavilion_wait, snow_world_wait, ferris_wheel_wait,
      dragon_boat_wait, avg_wait

    项目说明：
      - ice_slide：超级冰滑梯（最热门）
      - ice_bar：冰酒吧
      - ice_sculpture_walk：冰雕长廊
      - polar_pavilion：极地馆
      - snow_world：雪世界
      - ferris_wheel：冰雪摩天轮
      - dragon_boat：冰上龙舟

    排队时间与当日客流正相关，同时各项目热度不同。
    """
    print("[5/5] 正在生成排队时长数据...")

    records = []
    for _, row in flow_df.iterrows():
        visitors = int(row["visitors"])

        if visitors == 0:
            records.append({
                "date": row["date"],
                "ice_slide_wait": 0,
                "ice_bar_wait": 0,
                "ice_sculpture_walk_wait": 0,
                "polar_pavilion_wait": 0,
                "snow_world_wait": 0,
                "ferris_wheel_wait": 0,
                "dragon_boat_wait": 0,
                "avg_wait": 0,
            })
            continue

        # 客流归一化到 0~1（35000 为饱和线）
        load = min(visitors / 35000.0, 1.2)

        # 各项目热度系数
        popularity = {
            "ice_slide_wait":            3.2,   # 最热门
            "ice_bar_wait":              1.0,
            "ice_sculpture_walk_wait":   0.3,   # 基本不排队
            "polar_pavilion_wait":       1.8,
            "snow_world_wait":           1.3,
            "ferris_wheel_wait":         2.2,
            "dragon_boat_wait":          1.5,
        }

        waits = {}
        for proj, coeff in popularity.items():
            base_wait = load * coeff * 35     # 基准等待时间（分钟）
            wait = max(0, np.random.normal(base_wait, base_wait * 0.25))
            waits[proj] = round(wait, 1)

        avg_wait = round(np.mean(list(waits.values())), 1)

        records.append({
            "date": row["date"],
            **waits,
            "avg_wait": avg_wait,
        })

    df = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "queue_times.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    -> 已保存 {output_path} ({len(df)} 行)")
    return df


# ========================= 主流程 =========================

def main():
    print("=" * 60)
    print("  哈尔滨冰雪大世界 — 仿真运营数据生成器")
    print("=" * 60)
    print(f"  日期范围: {START_DATE} ~ {END_DATE}")
    print(f"  数据天数: {NUM_DAYS} 天")
    print(f"  输出目录: {DATA_DIR}")
    print("-" * 60)

    # 1. 生成客流数据
    flow_df = generate_tourist_flow()

    # 2. 生成营收数据
    revenue_df = generate_revenue(flow_df)

    # 3. 生成评论数据
    reviews_df = generate_reviews(flow_df)

    # 4. 生成旅行社数据
    agency_df = generate_travel_agency(flow_df)

    # 5. 生成排队时长数据
    queue_df = generate_queue_times(flow_df)

    # ---- 数据统计摘要 ----
    print("\n" + "=" * 60)
    print("  数据生成完成！统计摘要：")
    print("-" * 60)

    total_visitors = flow_df["visitors"].sum()
    peak_day = flow_df.loc[flow_df["visitors"].idxmax()]
    total_revenue = revenue_df["total_revenue"].sum() / 1e8     # 亿元

    print(f"  2年总客流: {total_visitors / 1e4:.1f} 万人次")
    print(f"  客流峰值日: {peak_day['date']} ({int(peak_day['visitors'])} 人)")
    print(f"  2年预估总收入: {total_revenue:.2f} 亿元")
    print(f"  生成评论数: {len(reviews_df)} 条")

    pos_ratio = (reviews_df["sentiment"] == "positive").mean() * 100
    print(f"  评论正面率: {pos_ratio:.1f}%")

    print(f"\n  所有 CSV 文件已保存至: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()