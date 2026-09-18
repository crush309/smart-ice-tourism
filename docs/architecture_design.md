# 系统架构设计文档

> 项目：多业态融合智慧文旅管控系统
> 版本：v1.0 | 日期：2026-09-15

---

## 1. 架构总览

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          展示层 Presentation                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 3D可视化大屏  │  │ 管理仪表盘    │  │ 游客小程序 │  │ 商户终端App   │   │
│  │ Vue3+Three.js│  │ Vue3+Element │  │ Uni-App   │  │ Flutter      │   │
│  └──────────────┘  └──────────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                            WebSocket / REST API
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                          业务层 Business                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ ①智能运营     │  │ ②智能规划预判 │  │ ③服务优化     │  │ ④业态融合   │ │
│  │ Spring Boot  │  │ Python AI   │  │ Python NLP  │  │ Spring+AI  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              Kafka 消息队列
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据层 Data                                    │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐  │
│  │ Kafka  │  │ Flink  │  │ TDengine │  │PostgreSQL│  │ MinIO/Neo4j │  │
│  │ 消息队列│  │ 流处理  │  │ 时序存储  │  │ 业务存储 │  │ 对象/图存储  │  │
│  └────────┘  └────────┘  └──────────┘  └────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              MQTT / HTTP
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                          感知层 Perception                               │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐  │
│  │IoT传感器│  │ 摄像头  │  │ WiFi探针  │  │ POS终端 │  │ 电子围栏基站  │  │
│  └────────┘  └────────┘  └──────────┘  └────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **分层解耦** | 四层架构各层独立演进，通过定义清晰的接口通信 |
| **模块化** | 四大业务模块可独立部署、独立扩展 |
| **数据驱动** | 所有决策基于实时数据流，而非事后批处理 |
| **弹性伸缩** | 旺季自动扩容，淡季自动缩容 |
| **安全可靠** | 数据传输加密、访问控制、灾备方案 |

---

## 2. 感知层设计

### 2.1 数据采集设备

| 设备类型 | 采集内容 | 通信协议 | 数据频率 |
|----------|---------|---------|---------|
| 高清摄像头 | 人流视频流 | RTSP | 实时(25fps) |
| WiFi探针 | 设备MAC地址、信号强度 | MQTT | 每30秒 |
| 电子围栏基站 | 人员进出事件 | MQTT | 实时 |
| POS终端 | 交易记录 | HTTP | 实时 |
| 温湿度传感器 | 环境温湿度 | MQTT | 每5分钟 |
| 票务闸机 | 入园/出园记录 | HTTP | 实时 |

### 2.2 边缘计算

在景区本地部署边缘网关，进行：
- 视频流本地AI推理（人数统计），仅上传计数结果
- 数据预处理和压缩
- 断网缓存（本地存储，恢复后补传）

---

## 3. 数据层设计

### 3.1 数据流

```
IoT设备 → MQTT Broker → Kafka → Flink 实时处理 → TDengine(时序)
                                   ↓
                            PostgreSQL(业务)
                                   ↓
                            Python AI引擎
                                   ↓
                            结果回写到DB → WebSocket推送 → 前端
```

### 3.2 存储选型

| 数据库 | 用途 | 选型理由 |
|--------|------|---------|
| **TDengine** | 时序数据（客流、环境监测） | 10倍于通用DB的时序查询性能 |
| **PostgreSQL** | 业务数据（订单、用户、配置） | 成熟稳定，支持JSON/PostGIS |
| **MinIO** | 对象存储（视频截图、日志） | S3兼容，私有部署 |
| **Neo4j** | 关系图谱（游客-消费-商户） | 图数据库，适合关联分析 |
| **Redis** | 缓存（实时排名、会话） | 极低延迟 |

### 3.3 核心数据表设计

```sql
-- 客流记录表 (TDengine)
CREATE TABLE tourist_flow (
    ts TIMESTAMP,
    area_id VARCHAR(32),
    device_id VARCHAR(32),
    count INT,
    density FLOAT
) TAGS (area_name VARCHAR(64));

-- 游客画像表 (PostgreSQL)
CREATE TABLE tourist_profile (
    id BIGSERIAL PRIMARY KEY,
    device_mac VARCHAR(64),
    visit_date DATE,
    stay_duration INT,        -- 停留时长(分钟)
    visit_areas TEXT[],        -- 到访区域列表
    consumption_total DECIMAL, -- 消费总额
    source_city VARCHAR(64),  -- 来源城市
    age_group VARCHAR(16),    -- 年龄段
    created_at TIMESTAMP DEFAULT NOW()
);

-- 商户交易表 (PostgreSQL)
CREATE TABLE merchant_transaction (
    id BIGSERIAL PRIMARY KEY,
    merchant_id BIGINT,
    category VARCHAR(32),    -- 业态类型: hotel/restaurant/retail
    amount DECIMAL(12,2),
    payment_method VARCHAR(32),
    trans_time TIMESTAMP,
    tourist_profile_id BIGINT REFERENCES tourist_profile(id)
);
```

---

## 4. 业务层设计

### 4.1 模块① 智能运营

**API接口：**

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/ops/realtime/flow` | GET | 实时客流数据 |
| `/api/v1/ops/realtime/heatmap` | GET | 热力图数据 |
| `/api/v1/ops/tourist/profile` | GET | 游客画像分析 |
| `/api/v1/ops/travel-agency` | GET | 旅行社入园数据 |
| `/api/v1/ops/dashboard` | GET | 驾驶舱聚合数据 |
| `/api/v1/ops/alerts` | GET | 告警列表 |

**核心类设计：**

```
FlowMonitorService
├── getRealtimeFlow(areaId) → FlowData
├── getHeatmapData() → List<HeatPoint>
└── updateFlowFromSensor(sensorData) → void

TouristProfileService
├── analyzeProfile(dateRange) → ProfileReport
├── getSourceDistribution() → Map<City, Count>
└── getAgeGroupDistribution() → Map<AgeGroup, Count>

DashboardService
├── getOverview() → DashboardVO
└── getRevenueSummary(dateRange) → RevenueReport
```

### 4.2 模块② 智能规划与预判

**预测管线：**

```
[数据加载] → [特征工程] → [模型推理] → [后处理] → [结果存储] → [API输出]
```

**API接口：**

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/predict/flow` | GET | 未来7天客流预测 |
| `/api/v1/predict/peak` | GET | 峰值时段预警 |
| `/api/v1/predict/queue` | GET | 排队时长预估 |
| `/api/v1/predict/resource` | GET | 资源调度建议 |

**模型管理：**
- 模型版本控制：每次训练保存版本号
- A/B测试：新旧模型并行运行对比
- 自动重训练：每月用最新数据重新微调

### 4.3 模块③ 服务智能优化

**NLP分析管线：**

```
[评论采集] → [文本清洗] → [BERT编码] → [情感分类] → [关键词提取] → [服务报告]
```

**API接口：**

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/optimize/sentiment` | GET | 情感趋势分析 |
| `/api/v1/optimize/complaints` | GET | 投诉分类统计 |
| `/api/v1/optimize/hotspots` | GET | 服务短板热图 |
| `/api/v1/optimize/suggestions` | GET | AI改进建议 |

### 4.4 模块④ 业态融合

**推荐系统架构：**

```
[用户画像] + [消费历史] + [知识图谱] → [协同过滤] + [关联规则] → [推荐列表]
```

**API接口：**

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/v1/fusion/recommend/packages` | GET | 智能套票推荐 |
| `/api/v1/fusion/recommend/products` | GET | 个性化产品推荐 |
| `/api/v1/fusion/analysis/behavior` | GET | 消费行为分析 |
| `/api/v1/fusion/graph/query` | GET | 知识图谱查询 |

---

## 5. 展示层设计

### 5.1 3D可视化大屏（Demo核心）

**技术栈：** Vue3 + Three.js + ECharts + WebSocket

**界面布局：**
```
┌─────────────────────────────────────────────────────┐
│            多业态融合智慧文旅管控系统                    │
├──────────┬──────────────────────┬────────────────────┤
│ 📊 客流   │                      │ 📍 景区概览          │
│ 实时数据  │                      │ • 冰晶城堡           │
│          │   Three.js 3D场景     │ • 极速冰滑梯         │
│ ┌──────┐ │   (冰雪大世界)        │ • 梦幻雪圈           │
│ │ 8,523│ │                      │                     │
│ └──────┘ │   可旋转/缩放/点击     │ 🟢 在线: 12         │
│          │                      │ 🔴 离线: 1          │
│ ┌──────┐ │                      │                     │
│ │折线图 │ │                      │ ⚠️ 冰晶城堡区域      │
│ │迷你流 │ │                      │   人流密集(85%)      │
│ └──────┘ │                      │                     │
├──────────┴──────────────────────┴────────────────────┤
│ 🕐 14:30:25 │ 👥 在线: 8,523 │ 🌨 -15°C │ 📅 2026-01-15 │
└─────────────────────────────────────────────────────┘
```

### 5.2 管理者仪表盘

**页面结构：**
- 首页概览：KPI 卡片（日客流、营收、满意度、预警数）
- 客流分析：趋势图、对比图、预测图
- 营收分析：按业态/时段/区域多维度下钻
- 舆情监控：实时评论流、情感趋势、词云

---

## 6. 部署架构

```
┌─────────────────────────────────────────────┐
│                  Nginx (反向代理+负载均衡)      │
├──────────────┬──────────────┬────────────────┤
│   前端服务    │   后端服务    │   AI服务        │
│   (Nginx)   │  (Spring Boot)│  (FastAPI)     │
│   Port: 80  │  Port: 8080  │  Port: 8000    │
├──────────────┴──────────────┴────────────────┤
│              数据中间件集群                     │
│   Kafka → Flink → TDengine → PostgreSQL      │
│                    Redis                     │
└─────────────────────────────────────────────┘
```

---

## 7. 非功能性需求

| 类别 | 要求 | 实现方案 |
|------|------|---------|
| 性能 | 大屏刷新率 ≥ 30fps | Three.js 场景 ≤ 5000面，Web Worker 数据处理 |
| 可用性 | 核心服务 99.9% | 微服务+健康检查+自动重启 |
| 安全 | 数据传输加密 | HTTPS + JWT认证 + API限流 |
| 扩展性 | 支持10倍数据增长 | Kafka分区扩展 + DB读写分离 |
| 可维护性 | 模块独立部署 | Docker Compose + 标准化API文档(Swagger) |