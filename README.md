# Kronos 台股預測實驗 (kronos-tw-stock)

用開源金融時序基礎模型 [Kronos](https://github.com/shiyu-coder/Kronos)（MIT License）預測台股日 K，並做滾動回測，檢查它到底有沒有預測力。

本 repo 只放我自己寫的腳本與結果，**不包含 Kronos 模型原始碼**。

## 結論先講：在台積電上沒有預測力

台積電 (2330)、Kronos-small、最近 30 個不重疊的 20 日窗口（`rolling_2330_small_summary.csv`）：

| 指標 | 結果 | 怎麼看 |
|---|---|---|
| 整段方向命中率 | **33%** | 永遠猜「漲」就有 73%，模型遠遠輸給這個基準 |
| 逐日方向命中率 | 47% | 跟丟銅板差不多 |
| 平均預測報酬 / 實際報酬 | **−19%** / +4% | 模型有嚴重的系統性偏空 |
| Spearman IC | +0.08 | 30 個樣本下和 0 沒有統計上的差別 |
| 只在模型看多時持有 / 一直持有（每窗平均） | +0.4% / +4.1% | 照模型做會錯過大部分漲幅 |

所以這份實驗的結論是**不能直接拿 Kronos-small 的預測來交易台積電**。
平均預測 −19% 這種偏差很大，可能的原因（還沒驗證）：模型沒看過這種長期單邊上漲、價格創新高的序列；抽樣溫度與 lookback 的設定；或成交金額是用「收盤 × 成交量」近似的。

| 單次回測 | 滾動回測 |
|---|---|
| ![](2330_TW_backtest_small.png) | ![](rolling_2330_small.png) |

## 這份實驗的限制

- **樣本很少**：一檔股票、30 個窗口。要下更一般的結論，至少要跑多檔股票（`--tickers` 可以一次給很多檔）
- **可能有資料洩漏**：回測的區間可能和 Kronos 預訓練用的資料重疊（模型也許「看過」這段行情）。就算如此結果還是不好，但這表示好的結果也不能直接相信
- **沒有算交易成本**：「看多才持有」的淨值曲線沒有扣手續費和證交稅
- **價格沒有還原權息**（`auto_adjust=False`）：除息日的跳空會被當成真的下跌，實際報酬會少算配息
- **未來日期只排除週末**，沒有排除台灣的國定假日（只當時間特徵用，影響很小）

## 使用方式

```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt          # 上游的依賴 (torch 等)
# 把本 repo 的 run_tw_stock.py、rolling_backtest.py 複製到 Kronos 根目錄（與 model/ 同層）
pip install yfinance matplotlib          # 這兩支腳本額外需要的

python run_tw_stock.py --ticker 2330.TW                                  # 單次回測
python run_tw_stock.py --ticker 2330.TW --mode forecast --pred_len 20    # 預測未來
python rolling_backtest.py --tickers 2330.TW 2317.TW 2454.TW             # 滾動回測（可多檔）
python rolling_backtest.py --tickers 2330.TW --windows 5 --samples 5     # 先跑少一點確認流程
```

上櫃股票用 `.TWO`，例如 `6488.TWO`。第一次執行會從 Hugging Face 下載模型。

滾動回測的輸出怎麼看：
- `dir_hit` 要明顯高於 `naive_hit`（永遠猜多數方向）才算有用
- `ic_spearman` > 0.05 而且在多檔、多段期間都穩定，才算有資訊
- `avg_pred_ret` 和 `avg_true_ret` 差很多 → 模型有系統性偏多或偏空

本專案為個人研究，不構成投資建議。
