# 多业态融合智慧文旅管控系统 — AI客流预测模块

## Phases

- [x] Phase 2: data_generator.py 仿真数据生成器 — **Status:** complete
- [x] Phase 3: predict_model.py LSTM客流预测模型 — **Status:** complete
- [x] Phase 4: 集成测试与验证 — **Status:** complete
- [ ] Phase 3: predict_model.py LSTM客流预测模型 — **Status:** pending
- [ ] Phase 4: 集成测试与验证 — **Status:** pending

## 关键设计决策

1. 数据周期：2年(730天)，2024-01-01 至 2025-12-31
2. 旺季：11月-3月；峰值：12月24日-2月15日(春节)；淡季：4月-10月
3. 模型：LSTM Encoder(128) → Attention → LSTM Decoder(64) → Dense
4. 预测输出：未来7天客流 + 峰值时段预警
5. 数据划分：70/15/15 训练/验证/测试