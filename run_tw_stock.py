"""
用 Kronos 預測台股日 K
------------------------------------------------
放在 Kronos repo 根目錄執行（跟 model/ 資料夾同一層）。

安裝：
    pip install -r requirements.txt
    pip install yfinance

用法：
    # 回測模式（預設）：拿最後 pred_len 天當答案，看模型預測得準不準
    python run_tw_stock.py --ticker 2330.TW

    # 預測模式：用最新資料往未來預測
    python run_tw_stock.py --ticker 2330.TW --mode forecast --pred_len 20

    # 上櫃股票用 .TWO，例如 6488.TWO
    # 想多跑幾條路徑取平均：--samples 5
    # 想用較大模型：--model base
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

MODELS = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def load_tw_daily(ticker: str, period: str) -> pd.DataFrame:
    """用 yfinance 抓日 K，整理成 Kronos 要的欄位。"""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        raise SystemExit(f"抓不到 {ticker} 的資料，確認代號（上市 .TW、上櫃 .TWO）")

    df = pd.DataFrame({
        "timestamps": pd.to_datetime(raw.index).tz_localize(None),
        "open": raw["Open"].values,
        "high": raw["High"].values,
        "low": raw["Low"].values,
        "close": raw["Close"].values,
        "volume": raw["Volume"].values.astype(float),
    })
    # yfinance 沒有成交金額，用 收盤價 × 成交量 近似
    df["amount"] = df["close"] * df["volume"]
    # 去掉停牌或資料缺漏的日子
    df = df[(df["volume"] > 0) & df[["open", "high", "low", "close"]].notna().all(axis=1)]
    return df.reset_index(drop=True)


def future_trading_days(last_day: pd.Timestamp, n: int) -> pd.Series:
    """未來 n 個平日（不含台灣國定假日，只是給模型當時間特徵用，影響不大）。"""
    days = pd.bdate_range(last_day + pd.Timedelta(days=1), periods=n)
    return pd.Series(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="2330.TW")
    ap.add_argument("--mode", choices=["backtest", "forecast"], default="backtest")
    ap.add_argument("--model", choices=MODELS.keys(), default="small")
    ap.add_argument("--lookback", type=int, default=400, help="輸入幾根 K 線（small/base 上限 512）")
    ap.add_argument("--pred_len", type=int, default=20, help="預測幾根")
    ap.add_argument("--samples", type=int, default=1, help="抽樣路徑數，取平均")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--period", default="5y", help="yfinance 抓多久的歷史")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_id, tok_id, max_ctx = MODELS[args.model]
    if args.lookback > max_ctx:
        print(f"lookback {args.lookback} 超過 {args.model} 上限 {max_ctx}，改成 {max_ctx}")
        args.lookback = max_ctx

    # 1. 資料
    df = load_tw_daily(args.ticker, args.period)
    need = args.lookback + (args.pred_len if args.mode == "backtest" else 0)
    if len(df) < need:
        raise SystemExit(f"資料只有 {len(df)} 根，不夠 {need} 根；調小 --lookback 或加大 --period")
    print(f"{args.ticker}: {len(df)} 根日 K，{df.timestamps.iloc[0].date()} ~ {df.timestamps.iloc[-1].date()}")

    if args.mode == "backtest":
        end = len(df) - args.pred_len
        hist = df.iloc[end - args.lookback:end].reset_index(drop=True)
        truth = df.iloc[end:].reset_index(drop=True)
        y_ts = truth["timestamps"]
    else:
        hist = df.iloc[-args.lookback:].reset_index(drop=True)
        truth = None
        y_ts = future_trading_days(hist["timestamps"].iloc[-1], args.pred_len)

    # 2. 模型
    device = "cuda:0" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"載入 {model_id}（device={device}）...")
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)

    # 3. 預測
    cols = ["open", "high", "low", "close", "volume", "amount"]
    pred = predictor.predict(
        df=hist[cols],
        x_timestamp=hist["timestamps"],
        y_timestamp=y_ts,
        pred_len=args.pred_len,
        T=args.T,
        top_p=args.top_p,
        sample_count=args.samples,
        verbose=True,
    )
    pred = pred.reset_index(drop=True)
    pred.insert(0, "timestamps", y_ts.values)

    # 4. 輸出
    tag = f"{args.ticker.replace('.', '_')}_{args.mode}_{args.model}"
    pred.to_csv(f"{tag}_pred.csv", index=False)
    print("\n預測結果（前幾筆）：")
    show = pred[["timestamps", "open", "high", "low", "close"]].head(10).copy()
    show["timestamps"] = show["timestamps"].dt.date
    print(show.round(2).to_string(index=False))

    last_close = hist["close"].iloc[-1]
    pred_ret = pred["close"].iloc[-1] / last_close - 1
    print(f"\n起點收盤 {last_close:.2f} → 預測第 {args.pred_len} 根收盤 {pred['close'].iloc[-1]:.2f}（{pred_ret:+.2%}）")

    if truth is not None:
        true_ret = truth["close"].iloc[-1] / last_close - 1
        mae = (pred["close"] - truth["close"]).abs().mean()
        mape = ((pred["close"] - truth["close"]).abs() / truth["close"]).mean()
        # 每天漲跌方向是否猜對
        p_dir = np.sign(np.diff(np.r_[last_close, pred["close"].values]))
        t_dir = np.sign(np.diff(np.r_[last_close, truth["close"].values]))
        hit = (p_dir == t_dir).mean()
        print(f"實際第 {args.pred_len} 根收盤 {truth['close'].iloc[-1]:.2f}（{true_ret:+.2%}）")
        print(f"收盤 MAE {mae:.2f}｜MAPE {mape:.2%}｜逐日方向命中率 {hit:.0%}｜"
              f"整段方向{'對' if np.sign(pred_ret) == np.sign(true_ret) else '錯'}")

    # 5. 畫圖
    show_hist = hist.iloc[-120:]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(show_hist["timestamps"], show_hist["close"], color="#333", lw=1.4, label="History")
    if truth is not None:
        ax1.plot(truth["timestamps"], truth["close"], color="#1f77b4", lw=1.6, label="Actual")
    ax1.plot(pred["timestamps"], pred["close"], color="#d62728", lw=1.6, ls="--", label="Kronos forecast")
    ax1.axvline(hist["timestamps"].iloc[-1], color="#999", lw=0.8, ls=":")
    ax1.set_title(f"{args.ticker}  Kronos-{args.model}  ({args.mode}, lookback={args.lookback}, pred={args.pred_len})")
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.bar(show_hist["timestamps"], show_hist["volume"], color="#999", width=0.8)
    if truth is not None:
        ax2.bar(truth["timestamps"], truth["volume"], color="#1f77b4", width=0.8)
    ax2.plot(pred["timestamps"], pred["volume"], color="#d62728", lw=1.4, ls="--")
    ax2.set_ylabel("Volume")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=130)
    print(f"\n已存檔：{tag}.png、{tag}_pred.csv")


if __name__ == "__main__":
    main()
