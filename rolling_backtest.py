"""
Kronos 台股滾動回測
------------------------------------------------
在過去一段時間，每隔 step 天取一個起點，用前 lookback 根預測後 pred_len 根，
再跟真實走勢比對，統計模型到底有沒有預測力。

放在 Kronos 資料夾（跟 run_tw_stock.py、model/ 同一層）執行。

用法：
    # 台積電，最近 30 個窗口，每窗 20 天、不重疊
    python rolling_backtest.py --tickers 2330.TW

    # 多檔一起跑
    python rolling_backtest.py --tickers 2330.TW 2317.TW 2454.TW 2308.TW

    # 先跑少一點確認流程（比較快）
    python rolling_backtest.py --tickers 2330.TW --windows 5 --samples 5

輸出：
    rolling_<名稱>_windows.csv   每個窗口的明細
    rolling_<名稱>_summary.csv   每檔 + 全部的統計
    rolling_<名稱>.png           散佈圖 + 策略淨值
"""
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_tw_stock import MODELS, load_tw_daily
from model import Kronos, KronosTokenizer, KronosPredictor

COLS = ["open", "high", "low", "close", "volume", "amount"]


def build_windows(df, ticker, lookback, pred_len, step, n_windows):
    """從最新往回切 n_windows 個窗口，回傳由舊到新的清單。"""
    last_end = len(df) - pred_len          # 最後一個窗口的預測起點
    wins = []
    for k in range(n_windows):
        end = last_end - k * step
        if end - lookback < 0:
            break
        hist = df.iloc[end - lookback:end].reset_index(drop=True)
        truth = df.iloc[end:end + pred_len].reset_index(drop=True)
        wins.append({"ticker": ticker, "hist": hist, "truth": truth})
    return wins[::-1]


def score_window(w, pred):
    hist, truth = w["hist"], w["truth"]
    last_close = hist["close"].iloc[-1]
    pc, tc = pred["close"].values, truth["close"].values
    pred_ret = pc[-1] / last_close - 1
    true_ret = tc[-1] / last_close - 1
    p_dir = np.sign(np.diff(np.r_[last_close, pc]))
    t_dir = np.sign(np.diff(np.r_[last_close, tc]))
    return {
        "ticker": w["ticker"],
        "start": truth["timestamps"].iloc[0].date(),
        "end": truth["timestamps"].iloc[-1].date(),
        "last_close": round(last_close, 2),
        "pred_ret": pred_ret,
        "true_ret": true_ret,
        "dir_correct": int(np.sign(pred_ret) == np.sign(true_ret)),
        "daily_hit": (p_dir == t_dir).mean(),
        "mape": (np.abs(pc - tc) / tc).mean(),
    }


def summarize(g: pd.DataFrame, name: str) -> dict:
    up_rate = (g["true_ret"] > 0).mean()
    long_mask = g["pred_ret"] > 0
    return {
        "name": name,
        "n_windows": len(g),
        "dir_hit": g["dir_correct"].mean(),                 # 整段方向命中率
        "naive_hit": max(up_rate, 1 - up_rate),             # 永遠猜多數方向的命中率（基準）
        "daily_hit": g["daily_hit"].mean(),                 # 逐日方向命中率
        "ic_pearson": g["pred_ret"].corr(g["true_ret"]),
        "ic_spearman": g["pred_ret"].corr(g["true_ret"], method="spearman"),
        "mape": g["mape"].mean(),
        "avg_pred_ret": g["pred_ret"].mean(),
        "avg_true_ret": g["true_ret"].mean(),
        "pct_long": long_mask.mean(),
        # 預測漲才持有 vs 一直持有，每窗平均報酬
        "strat_avg_ret": (g["true_ret"] * long_mask).mean(),
        "bh_avg_ret": g["true_ret"].mean(),
        # 模型看多 / 看空時，實際平均報酬
        "ret_when_long": g.loc[long_mask, "true_ret"].mean() if long_mask.any() else np.nan,
        "ret_when_short": g.loc[~long_mask, "true_ret"].mean() if (~long_mask).any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["2330.TW"])
    ap.add_argument("--model", choices=MODELS.keys(), default="small")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--pred_len", type=int, default=20)
    ap.add_argument("--step", type=int, default=20, help="窗口間隔；>= pred_len 才不重疊")
    ap.add_argument("--windows", type=int, default=30, help="每檔幾個窗口")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--T", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=8, help="一次丟幾個窗口給模型；記憶體不夠就調小")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_id, tok_id, max_ctx = MODELS[args.model]
    args.lookback = min(args.lookback, max_ctx)
    if args.step < args.pred_len:
        print(f"注意：step {args.step} < pred_len {args.pred_len}，窗口會重疊，策略淨值會重複計算，只看命中率/IC 就好")

    # 1. 資料與窗口
    windows = []
    for t in args.tickers:
        df = load_tw_daily(t, args.period)
        w = build_windows(df, t, args.lookback, args.pred_len, args.step, args.windows)
        print(f"{t}: {len(df)} 根日 K → {len(w)} 個窗口"
              + (f"（{w[0]['truth']['timestamps'].iloc[0].date()} ~ {w[-1]['truth']['timestamps'].iloc[-1].date()}）" if w else ""))
        windows += w
    if not windows:
        raise SystemExit("沒有足夠資料切出窗口")

    # 2. 模型
    device = "cuda:0" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"載入 {model_id}（device={device}），共 {len(windows)} 個窗口，samples={args.samples}, T={args.T}")
    predictor = KronosPredictor(Kronos.from_pretrained(model_id),
                                KronosTokenizer.from_pretrained(tok_id),
                                device=device, max_context=max_ctx)

    # 3. 分批預測
    rows = []
    t0 = time.time()
    for i in range(0, len(windows), args.batch):
        chunk = windows[i:i + args.batch]
        preds = predictor.predict_batch(
            df_list=[w["hist"][COLS] for w in chunk],
            x_timestamp_list=[w["hist"]["timestamps"] for w in chunk],
            y_timestamp_list=[w["truth"]["timestamps"] for w in chunk],
            pred_len=args.pred_len, T=args.T, top_p=args.top_p,
            sample_count=args.samples, verbose=False,
        )
        for w, p in zip(chunk, preds):
            rows.append(score_window(w, p.reset_index(drop=True)))
        done = min(i + args.batch, len(windows))
        el = time.time() - t0
        print(f"  {done}/{len(windows)} 完成，已花 {el:.0f}s，預估剩 {el / done * (len(windows) - done):.0f}s")

    res = pd.DataFrame(rows)

    # 4. 統計
    summ = [summarize(g, t) for t, g in res.groupby("ticker", sort=False)]
    if len(args.tickers) > 1:
        summ.append(summarize(res, "ALL"))
    summ = pd.DataFrame(summ)

    tag = "_".join(t.split(".")[0] for t in args.tickers[:5]) + ("_etc" if len(args.tickers) > 5 else "")
    tag = f"rolling_{tag}_{args.model}"
    res.to_csv(f"{tag}_windows.csv", index=False)
    summ.to_csv(f"{tag}_summary.csv", index=False)

    pct = ["dir_hit", "naive_hit", "daily_hit", "mape", "avg_pred_ret", "avg_true_ret",
           "pct_long", "strat_avg_ret", "bh_avg_ret", "ret_when_long", "ret_when_short"]
    show = summ.copy()
    for c in pct:
        show[c] = show[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for c in ["ic_pearson", "ic_spearman"]:
        show[c] = show[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
    print("\n===== 統計 =====")
    print(show.T.to_string(header=False))
    print("""
怎麼看：
  dir_hit 要明顯高於 naive_hit（永遠猜多數方向）才算有用
  ic_spearman > 0.05 且穩定，才算預測報酬的排序有資訊
  ret_when_long 應該 > ret_when_short；strat_avg_ret vs bh_avg_ret 看擇時有沒有加分
  avg_pred_ret 跟 avg_true_ret 差很多 → 模型有系統性偏多/偏空""")

    # 5. 圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for t, g in res.groupby("ticker", sort=False):
        ax1.scatter(g["pred_ret"] * 100, g["true_ret"] * 100, s=28, alpha=0.75, label=t)
    lim = max(res["pred_ret"].abs().max(), res["true_ret"].abs().max()) * 105
    ax1.plot([-lim, lim], [-lim, lim], color="#bbb", lw=0.8, ls="--")
    ax1.axhline(0, color="#999", lw=0.6)
    ax1.axvline(0, color="#999", lw=0.6)
    ax1.set_xlabel(f"Predicted {args.pred_len}-day return (%)")
    ax1.set_ylabel(f"Actual {args.pred_len}-day return (%)")
    ic = res["pred_ret"].corr(res["true_ret"], method="spearman")
    ax1.set_title(f"Predicted vs actual  (Spearman IC {ic:+.3f}, n={len(res)})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    if args.step >= args.pred_len:
        for t, g in res.groupby("ticker", sort=False):
            # 從 1.0 起跳：第一個點是第一窗的起點
            x = pd.to_datetime(pd.concat([g["start"].iloc[:1], g["end"]], ignore_index=True))
            bh = np.r_[1.0, (1 + g["true_ret"]).cumprod()]
            st = np.r_[1.0, (1 + g["true_ret"] * (g["pred_ret"] > 0)).cumprod()]
            line, = ax2.plot(x, bh, lw=1.2, alpha=0.6, label=f"{t} buy&hold")
            ax2.plot(x, st, lw=1.8, color=line.get_color(), ls="--", label=f"{t} Kronos long/cash")
        ax2.set_title("Equity: buy & hold vs long only when Kronos predicts up")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "windows overlap: equity curve skipped", ha="center")

    plt.tight_layout()
    plt.savefig(f"{tag}.png", dpi=130)
    print(f"\n已存檔：{tag}.png、{tag}_windows.csv、{tag}_summary.csv")


if __name__ == "__main__":
    main()
