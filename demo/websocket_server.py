#!/usr/bin/env python3
"""
WebSocket 数据推送服务器 — 多业态融合智慧文旅管控系统
从仿真CSV数据中读取并流式推送到前端3D大屏

启动方式: python websocket_server.py
监听端口: ws://localhost:8765
"""

import asyncio
import websockets
import json
import csv
import random
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── 加载CSV数据 ──
def load_csv(filename):
    """读取CSV并返回行列表"""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[WARN] {filename} 不存在，跳过")
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

print("[INIT] 加载仿真数据...")
tourist_flow = load_csv("tourist_flow.csv")
revenue_data = load_csv("revenue.csv")
reviews_data = load_csv("reviews.csv")
travel_agency = load_csv("travel_agency.csv")
queue_times = load_csv("queue_times.csv")
print(f"  tourist_flow: {len(tourist_flow)} rows")
print(f"  revenue:      {len(revenue_data)} rows")
print(f"  reviews:      {len(reviews_data)} rows")
print(f"  travel_agency:{len(travel_agency)} rows")
print(f"  queue_times:  {len(queue_times)} rows")

# ── 数据索引 ──
flow_idx = 0
revenue_idx = 0
review_idx = 0

# 景区功能区定义
ZONES = [
    {"name": "冰晶城堡主景区", "capacity": 500},
    {"name": "冰雪宫殿区", "capacity": 400},
    {"name": "极速冰滑梯区", "capacity": 200},
    {"name": "梦幻雪圈区", "capacity": 300},
    {"name": "冰酒吧休闲区", "capacity": 250},
    {"name": "迎宾广场", "capacity": 600},
    {"name": "望雪塔观景区", "capacity": 200},
    {"name": "冰雕艺术展区", "capacity": 350},
]

# 旅行社数据
AGENCIES = [
    {"name": "中青旅国际旅行社", "groups": 3, "people": 156, "guide": "张明远", "origin": "北京"},
    {"name": "哈尔滨康辉旅行社", "groups": 5, "people": 238, "guide": "李雪琴", "origin": "沈阳"},
    {"name": "携程旅行网(团队部)", "groups": 4, "people": 192, "guide": "王建华", "origin": "上海"},
    {"name": "春秋国际旅行社", "groups": 2, "people": 87, "guide": "赵丽华", "origin": "广州"},
    {"name": "龙江之星旅行社", "groups": 3, "people": 124, "guide": "孙志强", "origin": "成都"},
    {"name": "途牛旅游(地接部)", "groups": 2, "people": 95, "guide": "陈晓东", "origin": "深圳"},
]

# 评论池
REVIEW_POOL = [
    {"text": "冰雪大世界太震撼了！冰雕晶莹剔透，灯光秀绝美！", "sentiment": "pos", "source": "携程"},
    {"text": "排队时间太长了，零下20度冻得受不了……", "sentiment": "neg", "source": "美团"},
    {"text": "冰滑梯太刺激了，孩子玩了三遍还不肯走！", "sentiment": "pos", "source": "大众点评"},
    {"text": "整体不错，性价比还可以，就是餐饮有点贵", "sentiment": "neutral", "source": "携程"},
    {"text": "夜景比白天美十倍，强烈建议晚上去！", "sentiment": "pos", "source": "抖音"},
    {"text": "工作人员服务态度有待提高，问路爱理不理", "sentiment": "neg", "source": "微博"},
    {"text": "冰雪大舞台的演出太精彩了，值回票价！", "sentiment": "pos", "source": "小红书"},
    {"text": "停车场指示不清晰，找了半小时才找到车", "sentiment": "neg", "source": "美团"},
    {"text": "建议自带热水和暖宝宝，园内物价偏高", "sentiment": "neutral", "source": "知乎"},
    {"text": "第一次看到这么大规模的冰建筑，太壮观了", "sentiment": "pos", "source": "携程"},
    {"text": "电子导览很方便，跟着地图走不迷路", "sentiment": "pos", "source": "小程序"},
    {"text": "厕所卫生状况一般，高峰期还需要排队", "sentiment": "neg", "source": "大众点评"},
]


def generate_snapshot():
    """生成一帧完整的数据快照"""
    global flow_idx, revenue_idx, review_idx

    # ── 从CSV读取或模拟 ──
    if tourist_flow and flow_idx < len(tourist_flow):
        row = tourist_flow[flow_idx]
        flow_idx += 1
        current_visitors = int(float(row.get("visitor_count", 8500)))
    else:
        current_visitors = random.randint(7500, 10500)

    if revenue_data and revenue_idx < len(revenue_data):
        row = revenue_data[revenue_idx]
        revenue_idx += 1
        rev_total = float(row.get("total_revenue", 386.5))
        rev_ticket = float(row.get("ticket_revenue", rev_total * 0.45))
        rev_hotel = float(row.get("hotel_revenue", rev_total * 0.28))
        rev_dining = float(row.get("dining_revenue", rev_total * 0.17))
        rev_retail = float(row.get("retail_revenue", rev_total * 0.09))
        rev_parking = float(row.get("parking_revenue", 1.0))
    else:
        rev_total = 380 + random.random() * 30
        rev_ticket = rev_total * (0.42 + random.random() * 0.06)
        rev_hotel = rev_total * (0.25 + random.random() * 0.06)
        rev_dining = rev_total * (0.15 + random.random() * 0.04)
        rev_retail = rev_total - rev_ticket - rev_hotel - rev_dining - 1.0
        rev_parking = 0.8 + random.random() * 0.4

    # 各区域客流
    zones = []
    for z in ZONES:
        visitors = random.randint(30, z["capacity"])
        queue_time = max(0, round(visitors / z["capacity"] * 55 + (random.random() - 0.5) * 15))
        zones.append({
            "name": z["name"],
            "visitors": visitors,
            "capacity": z["capacity"],
            "queueTime": queue_time,
        })

    # 旅行社微调
    for a in AGENCIES:
        a["people"] = max(20, a["people"] + random.randint(-3, 5))

    # 评论
    reviews = random.sample(REVIEW_POOL, 4)
    for r in reviews:
        r["time"] = f"{random.randint(1, 45)}分钟前"

    # 游客画像
    source_cities = [
        {"name": "北京", "pct": 22}, {"name": "上海", "pct": 16}, {"name": "广州", "pct": 12},
        {"name": "沈阳", "pct": 10}, {"name": "成都", "pct": 8}, {"name": "其他", "pct": 32},
    ]

    return {
        "type": "snapshot",
        "data": {
            "timestamp": datetime.now().isoformat(),
            "currentVisitors": current_visitors,
            "totalToday": 32000 + random.randint(0, 2000),
            "entryRate": random.randint(120, 280),
            "revenue": round(rev_total, 1),
            "revenueTicket": round(rev_ticket, 1),
            "revenueHotel": round(rev_hotel, 1),
            "revenueDining": round(rev_dining, 1),
            "revenueRetail": round(rev_retail, 1),
            "revenueParking": round(rev_parking, 2),
            "satisfaction": round(91 + random.random() * 5, 1),
            "parkingUsed": random.randint(250, 450),
            "parkingTotal": 500,
            "hotelOccupancy": random.randint(65, 92),
            "diningTurnover": round(1.5 + random.random() * 2.5, 1),
            "retailAvgPrice": random.randint(45, 110),
            "weatherTemp": random.randint(-18, -10),
            "weatherType": random.choice(["小雪", "多云", "晴", "阵雪", "阴"]),
            "visitorHistory": [8000 + random.random() * 2500 for _ in range(30)],
            "zones": zones,
            "sourceCities": source_cities,
            "ageGroups": [
                {"name": "18-25岁", "pct": 28}, {"name": "26-35岁", "pct": 35},
                {"name": "36-45岁", "pct": 20}, {"name": "46-60岁", "pct": 12}, {"name": "60+", "pct": 5},
            ],
            "travelTypes": [
                {"name": "家庭出游", "pct": 40}, {"name": "情侣出行", "pct": 25},
                {"name": "朋友结伴", "pct": 20}, {"name": "独自旅行", "pct": 10}, {"name": "跟团游", "pct": 5},
            ],
            "agencies": AGENCIES,
            "reviews": reviews,
        },
    }


async def handler(websocket):
    """WebSocket 连接处理"""
    print(f"[CONNECT] 客户端已连接: {websocket.remote_address}")
    try:
        while True:
            snapshot = generate_snapshot()
            await websocket.send(json.dumps(snapshot, ensure_ascii=False))
            await asyncio.sleep(2)  # 每2秒推送一帧
    except websockets.exceptions.ConnectionClosed:
        print(f"[DISCONNECT] 客户端已断开: {websocket.remote_address}")
    except Exception as e:
        print(f"[ERROR] {e}")


async def main():
    print("[START] WebSocket 数据推送服务器启动中...")
    print("[START] 监听 ws://localhost:8765")
    print("[START] 按 Ctrl+C 停止")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] 服务器已停止")