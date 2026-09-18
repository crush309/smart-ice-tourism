#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
多业态融合智慧文旅管控系统 — LSTM 客流预测模型
==============================================================================
用途：基于历史运营数据，使用 PyTorch 构建 LSTM 编码器-解码器模型，
      预测未来 7 天的每日客流量，并给出峰值时段预警。

模型架构：
  LSTM Encoder (128 units) -> Attention -> LSTM Decoder (64 units) -> Dense(7)

输入特征（7维）：
  1. daily_visitors       — 历史日客流
  2. day_of_week          — 星期几 (0=Mon, 6=Sun)
  3. is_weekend           — 是否周末
  4. is_holiday           — 是否节假日
  5. temp_high            — 最高气温 (°C)
  6. precipitation        — 降水量 (mm)
  7. pre_sale_tickets     — 预售票量

输出：
  - 未来 7 天每日客流预测值
  - 峰值时段预警（预测期间是否有 >20000 人/天 的客流高峰）

评估指标：RMSE, MAE, MAPE
输出文件：
  - data/prediction_result.png    — 预测结果可视化
  - data/lstm_model.pth           — 训练好的模型权重

依赖：需先运行 data_generator.py 生成训练数据。

兼容性：Python 3.8+ / PyTorch 1.10+
作者：智慧城市大赛团队
日期：2026-09
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ========================= 全局配置 =========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] 使用: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ========================= 超参数 =========================
LOOK_BACK = 14              # 用过去 14 天预测
PRED_HORIZON = 7            # 预测未来 7 天
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
ENCODER_HIDDEN = 128        # 编码器 LSTM 隐藏单元
DECODER_HIDDEN = 64         # 解码器 LSTM 隐藏单元
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PATIENCE = 25               # 早停耐心值

# ========================= 中文绘图字体设置 =========================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ========================= 1. 数据加载与预处理 =========================

def load_and_preprocess():
    """
    从 data/tourist_flow.csv 加载数据，构造特征矩阵和标签。

    返回:
      X: np.ndarray, shape = (num_samples, LOOK_BACK, num_features)
      y: np.ndarray, shape = (num_samples, PRED_HORIZON)
      scaler_y: StandardScaler (客流反归一化用)
      dates: 每个样本对应的目标日期列表
      df: 原始 DataFrame（供后续绘图使用）
    """
    print("\n[1/5] 加载数据...")

    csv_path = os.path.join(DATA_DIR, "tourist_flow.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 {csv_path}。请先运行 data_generator.py 生成数据。"
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # ---- 特征工程 ----
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    # 季节正弦编码：将月份映射到 [-1, 1]，11月=冬季开始
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 11) / 12)

    # 滚动窗口特征：前7天客流均值和前14天客流均值
    df["visitors_7d_avg"] = df["visitors"].rolling(7, min_periods=1).mean()
    df["visitors_14d_avg"] = df["visitors"].rolling(14, min_periods=1).mean()
    # 客流变化趋势：7天均值 - 14天均值
    df["visitors_trend"] = df["visitors_7d_avg"] - df["visitors_14d_avg"]

    # 选择特征列
    feature_cols = [
        "visitors",             # 日客流
        "day_of_week",          # 星期几 (0-6)
        "is_weekend",           # 是否周末
        "is_holiday",           # 是否节假日
        "is_peak_season",       # 是否峰值季（12/24-2/15）
        "is_operating_season",  # 是否运营季（11-3月）
        "month_sin",            # 月份正弦编码
        "temp_high",            # 最高温
        "temp_low",             # 最低温
        "precipitation",        # 降水
        "wind_speed",           # 风速
        "pre_sale_tickets",     # 预售票
        "visitors_7d_avg",      # 7天客流均值
        "visitors_14d_avg",     # 14天客流均值
        "visitors_trend",       # 客流趋势
    ]

    data = df[feature_cols].values.astype(np.float32)

    # ---- 分别归一化特征和目标 ----
    # 客流是第0列，其余是特征
    y_raw = data[:, 0].reshape(-1, 1)                  # 客流
    X_raw = data.copy()                                 # 全部特征（含客流）

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw).flatten()

    # ---- 滑动窗口构建（同时记录每个样本是否在运营季）----
    X_list, y_list, date_list, op_mask_list = [], [], [], []

    for i in range(len(df) - LOOK_BACK - PRED_HORIZON + 1):
        # 输入：过去 LOOK_BACK 天的全部特征
        X_list.append(X_scaled[i : i + LOOK_BACK])
        # 输出：未来 PRED_HORIZON 天的客流
        y_list.append(y_scaled[i + LOOK_BACK : i + LOOK_BACK + PRED_HORIZON])
        # 记录目标日期（用于后续可视化）
        date_list.append(df["date"].iloc[i + LOOK_BACK : i + LOOK_BACK + PRED_HORIZON].tolist())
        # 标记该窗口是否涉及运营季（预测窗口内至少有一天在运营季）
        in_season = int(df["is_operating_season"].iloc[
            i + LOOK_BACK : i + LOOK_BACK + PRED_HORIZON
        ].sum() > 0)
        op_mask_list.append(in_season)

    X = np.array(X_list)
    y = np.array(y_list)
    op_mask = np.array(op_mask_list)

    print(f"    特征矩阵 X 形状: {X.shape}  (样本数, 时间步, 特征数)")
    print(f"    标签矩阵 y 形状: {y.shape}  (样本数, 预测天数)")
    print(f"    总可用样本: {len(X)}")
    print(f"    涉及运营季的样本: {op_mask.sum()} / {len(op_mask)}")

    # ---- 按时间顺序划分（不打乱，保证时序性）----
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    dates_test = date_list[val_end:]
    op_mask_test = op_mask[val_end:]

    print(f"    训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    print(f"    测试集运营季样本: {op_mask_test.sum()} / {len(op_mask_test)}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler_y, dates_test, df, op_mask_test)


# ========================= 2. 数据集类 =========================

class TourismDataset(Dataset):
    """PyTorch Dataset：将 numpy 数组转为 tensor"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========================= 3. 注意力机制 =========================

class Attention(nn.Module):
    """
    Bahdanau 风格的加性注意力（Additive Attention）。

    对编码器各个时间步的输出计算注意力权重，生成上下文向量。
    """

    def __init__(self, encoder_hidden, decoder_hidden):
        """
        参数:
          encoder_hidden: 编码器 LSTM 隐藏维度
          decoder_hidden: 解码器 LSTM 隐藏维度
        """
        super().__init__()
        self.attn = nn.Linear(encoder_hidden + decoder_hidden, decoder_hidden)
        self.v = nn.Linear(decoder_hidden, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        参数:
          encoder_outputs: (batch, seq_len, encoder_hidden)
          decoder_hidden:  (batch, decoder_hidden)
        返回:
          context: (batch, encoder_hidden) — 上下文向量
          attn_weights: (batch, seq_len) — 注意力权重
        """
        seq_len = encoder_outputs.shape[1]
        # 将 decoder_hidden 复制 seq_len 次
        decoder_hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # 拼接并计算能量
        energy = torch.tanh(self.attn(torch.cat([encoder_outputs, decoder_hidden_rep], dim=2)))
        attn_scores = self.v(energy).squeeze(2)          # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1) # (batch, seq_len)
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attn_weights


# ========================= 4. LSTM Encoder-Decoder 模型 =========================

class LSTMPredictor(nn.Module):
    """
    LSTM 编码器-解码器客流预测模型。

    架构:
      Encoder LSTM(128) -> Attention -> Decoder LSTM(64) -> Dense(PRED_HORIZON)
    """

    def __init__(self, input_size, encoder_hidden, decoder_hidden, pred_horizon,
                 num_layers=1, dropout=0.2):
        """
        参数:
          input_size:     输入特征维度
          encoder_hidden: 编码器隐藏单元数
          decoder_hidden: 解码器隐藏单元数
          pred_horizon:   预测步长（天数）
          num_layers:     LSTM 层数
          dropout:        dropout 比率
        """
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.pred_horizon = pred_horizon

        # ---- 编码器 LSTM ----
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- 注意力层 ----
        self.attention = Attention(encoder_hidden, decoder_hidden)

        # ---- 编码器隐藏状态 → 解码器隐藏状态投影 ----
        self.enc_to_dec_hidden = nn.Linear(encoder_hidden, decoder_hidden)

        # ---- 解码器 LSTM ----
        self.decoder_lstm = nn.LSTM(
            input_size=encoder_hidden,      # 注意力上下文作为输入
            hidden_size=decoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ---- 输出层 ----
        self.fc = nn.Sequential(
            nn.Linear(decoder_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, pred_horizon),
        )

    def forward(self, x):
        """
        参数:
          x: (batch, LOOK_BACK, input_size)
        返回:
          output: (batch, pred_horizon)
        """
        batch_size = x.shape[0]

        # Step 1: 编码器 LSTM
        enc_out, (enc_h, enc_c) = self.encoder_lstm(x)
        # enc_out: (batch, seq_len, encoder_hidden)
        # enc_h:   (num_layers, batch, encoder_hidden)

        # 使用编码器最后一个时间步的隐藏状态作为注意力 query
        # 需要将其投影到 decoder_hidden 维度，以匹配注意力层的期望
        enc_last_hidden = enc_h[-1]                         # (batch, encoder_hidden)
        dec_query = self.enc_to_dec_hidden(enc_last_hidden)  # (batch, decoder_hidden)

        # Step 2: Attention — 计算上下文向量
        context, attn_weights = self.attention(enc_out, dec_query)
        # context: (batch, encoder_hidden)

        # Step 3: 解码器 LSTM — 将 context 扩展为解码序列
        # 重复 context 作为解码器每一步的输入
        dec_input = context.unsqueeze(1).repeat(1, self.pred_horizon, 1)
        # dec_input: (batch, pred_horizon, encoder_hidden)

        dec_out, _ = self.decoder_lstm(dec_input)
        # dec_out: (batch, pred_horizon, decoder_hidden)

        # Step 4: 全连接输出层（对每个预测步分别映射）
        output = self.fc(dec_out[:, -1, :])       # 取最后一个时间步
        # output: (batch, pred_horizon)

        return output


# ========================= 5. 训练与评估 =========================

def train_model(model, train_loader, val_loader, scaler_y):
    """
    模型训练循环，包含早停机制。

    返回:
      model: 训练好的模型（已加载最优权重）
      history: 训练历史（train_loss, val_loss）
    """
    print(f"\n[2/5] 开始训练 (EPOCHS={EPOCHS}, LR={LEARNING_RATE})...")

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器：当验证损失不再下降时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        # ---- 训练阶段 ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- 验证阶段 ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # 打印训练进度
        if epoch % 20 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

        # ---- 早停 + 保存最优模型 ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最优权重
            torch.save(model.state_dict(),
                       os.path.join(DATA_DIR, "lstm_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    => 早停触发！最优 epoch: {best_epoch}")
            break

    # 加载最优模型
    model.load_state_dict(torch.load(
        os.path.join(DATA_DIR, "lstm_model.pth"),
        weights_only=True,
    ))
    print(f"    => 训练完成！最优验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")

    return model, history


def evaluate_model(model, test_loader, scaler_y, op_mask_test=None):
    """
    在测试集上评估模型，计算 RMSE / MAE / MAPE。

    参数:
      op_mask_test: 运营季标记，如果提供则分别评估全部样本和运营季样本
    返回反归一化后的真实值和预测值。
    """
    print("\n[3/5] 模型评估...")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).cpu().numpy()
            all_preds.append(pred)
            all_true.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true_scaled = np.concatenate(all_true, axis=0)

    # 反归一化到原始客流规模
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)

    # 确保非负
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # ---- 计算各评估指标（全部样本）----
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    # MAPE (避免除零)
    mask = y_true.flatten() > 1
    mape = np.mean(np.abs((y_true.flatten()[mask] - y_pred.flatten()[mask])
                           / y_true.flatten()[mask])) * 100

    print(f"    [全部测试集] RMSE: {rmse:.1f} | MAE: {mae:.1f} | MAPE: {mape:.2f}%")

    # ---- 仅在运营季样本上评估 ----
    if op_mask_test is not None and op_mask_test.sum() > 0:
        y_true_op = y_true[op_mask_test == 1]
        y_pred_op = y_pred[op_mask_test == 1]

        rmse_op = np.sqrt(mean_squared_error(y_true_op.flatten(), y_pred_op.flatten()))
        mae_op = mean_absolute_error(y_true_op.flatten(), y_pred_op.flatten())
        mask_op = y_true_op.flatten() > 1
        mape_op = np.mean(np.abs((y_true_op.flatten()[mask_op] - y_pred_op.flatten()[mask_op])
                                  / y_true_op.flatten()[mask_op])) * 100

        print(f"    [仅运营季]   RMSE: {rmse_op:.1f} | MAE: {mae_op:.1f} | MAPE: {mape_op:.2f}%")

    return y_true, y_pred, rmse, mae, mape


# ========================= 6. 可视化 =========================

def plot_training_history(history):
    """绘制训练/验证损失曲线"""
    print("\n[4/5] 绘制训练历史...")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="训练损失 (Train Loss)", linewidth=1.5)
    ax.plot(history["val_loss"], label="验证损失 (Val Loss)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM 客流预测模型 — 训练收敛曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    history_path = os.path.join(DATA_DIR, "training_history.png")
    fig.savefig(history_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> 已保存 {history_path}")


def plot_prediction_result(y_true, y_pred, dates_test, rmse, mae, mape):
    """
    绘制预测 vs 真实对比图。

    包含三个子图：
      1. 整个测试集的 7 天预测 vs 真实（取第1天预测）
      2. 散点图：预测值 vs 真实值
      3. 误差分布直方图
    """
    print("[5/5] 绘制预测结果可视化...")

    # 展平所有预测（用于散点图和误差直方图）
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    fig = plt.figure(figsize=(16, 10))

    # ---- 子图 1：时序预测对比 ----
    ax1 = fig.add_subplot(2, 2, (1, 2))

    # 取测试集每个样本的 第1天 预测，构成连续时间序列
    day1_true = y_true[:, 0]
    day1_pred = y_pred[:, 0]
    # 获取对应日期
    first_dates = [d[0] for d in dates_test]

    ax1.plot(first_dates, day1_true, "b-", label="真实客流", linewidth=1.2, alpha=0.8)
    ax1.plot(first_dates, day1_pred, "r--", label="预测客流", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("日客流量（人）")
    ax1.set_title(f"哈尔滨冰雪大世界 客流预测 vs 真实 (测试集)\n"
                  f"RMSE={rmse:.0f} | MAE={mae:.0f} | MAPE={mape:.1f}%")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ---- 子图 2：散点图 ----
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.scatter(y_true_flat, y_pred_flat, alpha=0.4, s=8, c="steelblue", edgecolors="none")
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax2.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="理想预测 y=x")
    ax2.set_xlabel("真实客流（人/天）")
    ax2.set_ylabel("预测客流（人/天）")
    ax2.set_title("预测值 vs 真实值 散点图")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3：误差分布 ----
    ax3 = fig.add_subplot(2, 2, 4)
    errors = y_pred_flat - y_true_flat
    ax3.hist(errors, bins=50, color="seagreen", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    ax3.set_xlabel("预测误差（人/天）")
    ax3.set_ylabel("频次")
    ax3.set_title(f"预测误差分布 (均值={errors.mean():.0f}, 标准差={errors.std():.0f})")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    result_path = os.path.join(DATA_DIR, "prediction_result.png")
    fig.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> 已保存 {result_path}")


# ========================= 7. 峰值预警分析 =========================

def peak_warning_analysis(y_true, y_pred, dates_test):
    """
    分析预测期间峰值时段（客流 > 20000 人/天）的预警准确性。

    打印：
      - 峰值日检测的精确率 / 召回率 / F1
      - 预测峰值日列表
    """
    print("\n" + "=" * 60)
    print("  峰值时段预警分析")
    print("=" * 60)

    THRESHOLD = 20000     # 峰值客流阈值

    true_peak = y_true.flatten() > THRESHOLD
    pred_peak = y_pred.flatten() > THRESHOLD

    tp = np.sum(true_peak & pred_peak)
    fp = np.sum(~true_peak & pred_peak)
    fn = np.sum(true_peak & ~pred_peak)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  峰值阈值: {THRESHOLD} 人/天")
    print(f"  真实峰值样本: {np.sum(true_peak)}")
    print(f"  预测峰值样本: {np.sum(pred_peak)}")
    print(f"  准确率 (Precision): {precision:.2%}")
    print(f"  召回率 (Recall):    {recall:.2%}")
    print(f"  F1 分数:           {f1:.2%}")

    # 输出若干峰值预警示例
    print(f"\n  示例预警（测试集前10个峰值日）:")
    flat_pred = y_pred.flatten()
    flat_true = y_true.flatten()
    all_dates = [d for sublist in dates_test for d in sublist]

    peak_indices = np.where(true_peak)[0]
    shown = 0
    for idx in peak_indices:
        if shown >= 10:
            break
        print(f"    {all_dates[idx].strftime('%Y-%m-%d')}  "
              f"真实: {flat_true[idx]:.0f} 人  "
              f"预测: {flat_pred[idx]:.0f} 人  "
              f"{'[预警正确]' if pred_peak[idx] else '[漏报]'}")
        shown += 1

    return precision, recall, f1


# ========================= 8. 主流程 =========================

def main():
    print("=" * 60)
    print("  多业态融合智慧文旅管控系统 — LSTM 客流预测模型")
    print("=" * 60)
    print(f"  预测窗口: 过去 {LOOK_BACK} 天 -> 未来 {PRED_HORIZON} 天")
    print(f"  模型结构: LSTM({ENCODER_HIDDEN}) -> Attention -> LSTM({DECODER_HIDDEN}) -> Dense")
    print(f"  设备: {DEVICE}")
    print("-" * 60)

    # ---- Step 1: 数据加载 ----
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_y, dates_test, df, op_mask_test) = load_and_preprocess()

    # ---- Step 2: 构建 DataLoader ----
    train_dataset = TourismDataset(X_train, y_train)
    val_dataset = TourismDataset(X_val, y_val)
    test_dataset = TourismDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Step 3: 构建模型 ----
    input_size = X_train.shape[2]      # 特征维度

    model = LSTMPredictor(
        input_size=input_size,
        encoder_hidden=ENCODER_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        pred_horizon=PRED_HORIZON,
        num_layers=1,
        dropout=0.2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  模型总参数量: {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")
    print(f"  输入特征维度: {input_size}")
    print(f"  预测步长:     {PRED_HORIZON} 天")

    # ---- Step 4: 训练 ----
    model, history = train_model(model, train_loader, val_loader, scaler_y)

    # ---- Step 5: 评估 ----
    y_true, y_pred, rmse, mae, mape = evaluate_model(
        model, test_loader, scaler_y, op_mask_test
    )

    # ---- Step 6: 可视化 ----
    plot_training_history(history)
    plot_prediction_result(y_true, y_pred, dates_test, rmse, mae, mape)

    # ---- Step 7: 峰值预警分析 ----
    peak_warning_analysis(y_true, y_pred, dates_test)

    # ---- 最终输出 ----
    print("\n" + "=" * 60)
    print("  模型训练与评估完成！")
    print("-" * 60)
    print(f"  模型文件: {os.path.join(DATA_DIR, 'lstm_model.pth')}")
    print(f"  训练曲线: {os.path.join(DATA_DIR, 'training_history.png')}")
    print(f"  预测图表: {os.path.join(DATA_DIR, 'prediction_result.png')}")
    print(f"\n  核心指标: RMSE={rmse:.0f} | MAE={mae:.0f} | MAPE={mape:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()