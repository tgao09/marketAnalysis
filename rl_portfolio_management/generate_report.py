"""Generate self-contained research report from frozen experiment artifacts."""
from __future__ import annotations

import ast, base64, io, json, re
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def loadj(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {} if default is None else default

def fig64(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=150,bbox_inches="tight",facecolor="#0b1220"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

def plot(title, fn):
    plt.rcParams.update({"text.color":"#dce7f5","axes.labelcolor":"#aabbd0","axes.edgecolor":"#40536c","xtick.color":"#9fb0c6","ytick.color":"#9fb0c6","axes.facecolor":"#111c2e","figure.facecolor":"#0b1220","grid.color":"#263a52"})
    fig,ax=plt.subplots(figsize=(9,4)); fn(ax); ax.set_title(title,loc="left",weight="bold"); ax.grid(alpha=.35); return fig64(fig)

def pct(x):
    try: return f"{100*float(x):.2f}%"
    except: return "—"
def num(x):
    try: return f"{float(x):.3f}"
    except: return "—"

def bootstrap(df, cols, n=10000):
    rng=np.random.default_rng(20260711); out={}
    for c in cols:
        x=pd.to_numeric(df[c],errors="coerce").dropna().to_numpy()
        if len(x):
            means=np.mean(rng.choice(x,(n,len(x)),replace=True),axis=1)
            out[c]={"mean":float(x.mean()),"ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"n":len(x)}
    return out

def table(rows, cols):
    h="".join(f"<th>{escape(str(c))}</th>" for c in cols)
    b="".join("<tr>"+"".join(f"<td>{escape(str(r.get(c,'—')))}</td>" for c in cols)+"</tr>" for r in rows)
    return f"<div class=scroll><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"

def main():
    report=ROOT/'report'; report.mkdir(exist_ok=True)
    manifests=list((ROOT/'data/snapshots').glob('*/manifest.json')); manifests.sort(key=lambda p:p.stat().st_mtime)
    manifest=loadj(manifests[-1]) if manifests else {}
    fm=pd.read_csv(ROOT/'runs/final_evaluation/final_metrics.csv')
    summary=loadj(ROOT/'runs/final_evaluation/final_summary.json')
    base_dir=next(iter((ROOT/'results/baselines').glob('*')),None)
    baseline=loadj(base_dir/'aggregate.json') if base_dir else {}
    curves=pd.read_csv(base_dir/'equity_curves.csv') if base_dir and (base_dir/'equity_curves.csv').exists() else pd.DataFrame()
    forecast=loadj(ROOT/'results/forecast_replay/summary.json')
    forecast_metrics=pd.read_csv(ROOT/'results/forecast_replay/metrics.csv')
    trials=pd.read_csv(ROOT/'runs/optuna_rough/study_trials.csv')
    stabs=[loadj(ROOT/'runs/stability_trial11/stability_summary.json'),loadj(ROOT/'runs/seed_confirmation_5k/stability_summary.json')]
    attribution=loadj(ROOT/'runs/final_evaluation/ticker_attribution/aggregate.json')
    ci=bootstrap(fm,['cumulative_return','annualized_return','sharpe','sortino','maximum_drawdown','turnover'])

    agg=fm.groupby('seed').agg(cumulative_return=('cumulative_return','mean'),sharpe=('sharpe','mean'),maximum_drawdown=('maximum_drawdown','mean')).reset_index()
    rlrow={"Strategy":"RL PPO (45 seed-fold tests)","Median return":pct(fm.cumulative_return.median()),"Median Sharpe":num(fm.sharpe.median()),"Worst drawdown":pct(fm.maximum_drawdown.max())}
    comp=[rlrow]+[{"Strategy":k,"Median return":pct(v.get('median_return')),"Median Sharpe":num(v.get('median_sharpe')),"Worst drawdown":pct(v.get('worst_drawdown'))} for k,v in baseline.items()]
    if not forecast_metrics.empty:
        for _,r in forecast_metrics.iterrows(): comp.append({"Strategy":f"{r.iloc[0]} (legacy replay)","Median return":pct(r.get('cumulative_return')),"Median Sharpe":num(r.get('sharpe')),"Worst drawdown":pct(r.get('maximum_drawdown'))})

    imgs={}
    def kpi(ax):
        xs=['RL','Eq-weight','Momentum','SPY']; vals=[fm.cumulative_return.mean()]+[baseline.get(x,{}).get('median_return',0) for x in ['equal_weight_long','momentum_20d','spy_buy_hold']]
        ax.bar(xs,np.array(vals)*100,color=['#f05d5e','#5bc0be','#6fffe9','#ffd166']); ax.axhline(0,color='white',lw=.7); ax.set_ylabel('Return (%)')
    imgs['kpi']=plot('Out-of-sample return comparison',kpi)
    imgs['folds']=plot('RL returns by fold and seed',lambda ax: [ax.plot(g.fold,g.cumulative_return*100,marker='o',alpha=.75,label=str(s)) for s,g in fm.groupby('seed')])
    # representative and normalized equity curves
    eqfiles=list((ROOT/'runs/final_evaluation/final').glob('seed_*/fold_*/test_equity.csv'))
    def equity(ax):
        for p in eqfiles[:9]:
            d=pd.read_csv(p); ax.plot(np.arange(len(d)),d.equity/d.equity.iloc[0],alpha=.42)
        ax.set_ylabel('Normalized equity'); ax.set_xlabel('Test bars')
    imgs['equity']=plot('Representative RL test equity paths',equity)
    def dd(ax):
        for p in eqfiles[:9]:
            d=pd.read_csv(p).equity; ax.plot((d/d.cummax()-1)*100,alpha=.42)
        ax.set_ylabel('Drawdown (%)'); ax.set_xlabel('Test bars')
    imgs['dd']=plot('Representative RL drawdowns',dd)
    def opt(ax):
        t=trials[pd.to_numeric(trials.value,errors='coerce').notna()].copy(); t.value=pd.to_numeric(t.value); ax.scatter(t.number,t.value,c=np.where(t.state.eq('COMPLETE'),'#6fffe9','#697a91')); ax.plot(t.number,t.value.cummax(),color='#ffd166',label='best so far'); ax.legend(); ax.set_xlabel('Trial'); ax.set_ylabel('Validation score')
    imgs['optuna']=plot('Optuna rough-search history',opt)
    def importance(ax):
        t=trials[trials.state.eq('COMPLETE')]; vals={c.replace('params_',''):abs(pd.to_numeric(t[c],errors='coerce').corr(pd.to_numeric(t.value,errors='coerce'),method='spearman')) for c in t if c.startswith('params_')}; vals={k:v for k,v in vals.items() if np.isfinite(v)}; s=pd.Series(vals).sort_values().tail(8); ax.barh(s.index,s.values,color='#5bc0be'); ax.set_xlabel('|Spearman correlation| (descriptive)')
    imgs['importance']=plot('Parameter importance proxy',importance)
    stabrows=[]
    for group,s in zip(['neighborhood','seed confirmation'],stabs):
        for r in s.get('runs',[]): stabrows.append({"group":group,"name":r.get('name',r.get('label','run')),"score":r.get('score',r.get('robust_score',r.get('mean_score')))})
    imgs['stability']=plot('Neighborhood and seed stability',lambda ax: ax.scatter(range(len(stabrows)),[r['score'] if r['score'] is not None else np.nan for r in stabrows],c=['#6fffe9' if r['group']=='neighborhood' else '#ffd166' for r in stabrows]))
    holds=[]
    for x in fm.holding_days_distribution.dropna():
        try: holds.extend(ast.literal_eval(x))
        except: pass
    imgs['holding']=plot('Completed-position holding time (capped display)',lambda ax: ax.hist(np.clip(holds,0,30),bins=30,color='#5bc0be'))
    imgs['exposure']=plot('RL exposure and turnover by fold',lambda ax: (ax.scatter(fm.average_gross_exposure,fm.turnover,c=fm.cumulative_return,cmap='coolwarm'),ax.set_xlabel('Average gross exposure'),ax.set_ylabel('Turnover')))
    imgs['contrib']=plot('Long versus short P&L contribution',lambda ax: ax.scatter(fm.long_contribution,fm.short_contribution,c=fm.cumulative_return,cmap='coolwarm'))
    ticker_rows=[]
    for r in attribution.get('by_symbol',[]):
        ticker_rows.append({'Ticker':r['symbol'],'Total P&L':f"${r['total_contribution']:,.2f}",
                            'Realized':f"${r['realized_pnl']:,.2f}",
                            'Ending unrealized':f"${r['ending_unrealized_pnl']:,.2f}",
                            'Orders':r['filled_order_count'],'FIFO trades':r['completed_trade_count'],
                            'Win rate':pct(r['completed_trade_win_rate_weighted'])})
    def ticker_plot(ax):
        values=sorted(attribution.get('by_symbol',[]),key=lambda r:r['total_contribution'])
        ax.barh([r['symbol'] for r in values],[r['total_contribution'] for r in values],
                color=['#5bc0be' if r['total_contribution'] >= 0 else '#f05d5e' for r in values])
        ax.axvline(0,color='white',lw=.7); ax.set_xlabel('Aggregated seed-fold P&L ($)')
    imgs['ticker']=plot('Frozen PPO per-ticker contribution',ticker_plot)

    # descriptive regimes from matching SPY fold return sign and cross-fold volatility median
    spy=baseline.get('spy_buy_hold',{}); fold_spy=spy.get('fold_metrics',[]) if isinstance(spy,dict) else []
    spy_from_curves={}
    if not curves.empty:
        for f,g in curves[curves.strategy.eq('spy_buy_hold')].groupby('fold'):
            e=pd.to_numeric(g.equity,errors='coerce').dropna()
            if len(e)>1: spy_from_curves[int(f)]={'cumulative_return':float(e.iloc[-1]/e.iloc[0]-1),'volatility':float(e.pct_change().std()*np.sqrt(252))}
    spy_vol_median=np.median([x['volatility'] for x in spy_from_curves.values()]) if spy_from_curves else np.nan
    regimes=[]
    for fold,g in fm.groupby('fold'):
        sr=None
        sv=None
        if fold < len(fold_spy) and isinstance(fold_spy[fold],dict): sr=fold_spy[fold].get('cumulative_return'); sv=fold_spy[fold].get('volatility')
        if int(fold) in spy_from_curves: sr=spy_from_curves[int(fold)]['cumulative_return']; sv=spy_from_curves[int(fold)]['volatility']
        regimes.append({'fold':int(fold),'regime':('Bull' if (sr or 0)>=0 else 'Bear')+' / '+('high vol' if sv is not None and sv>spy_vol_median else 'low vol'),'rl_return':pct(g.cumulative_return.mean()),'rl_sharpe':num(g.sharpe.mean()),'spy_return':pct(sr)})

    foldrows=[]
    for f,g in fm.groupby('fold'): foldrows.append({'Fold':int(f),'Seeds':len(g),'Return mean':pct(g.cumulative_return.mean()),'Sharpe mean':num(g.sharpe.mean()),'Max DD mean':pct(g.maximum_drawdown.mean()),'Holding days':num(g.average_holding_days.mean()),'Turnover':num(g.turnover.mean())})
    failures=[]
    try:
        failures=[json.loads(x) for x in (ROOT/'failures.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    except: pass
    data={"generated_from":{"snapshot_id":manifest.get('snapshot_id'),"snapshot_hash":manifest.get('content_sha256')},"rl_bootstrap_ci":ci,"rl_seed_fold_metrics":fm.drop(columns=['holding_days_distribution']).to_dict('records'),"comparisons":comp,"regimes":regimes,"optuna_trials":len(trials),"stability":stabs,"ticker_attribution":attribution,"forecast_provenance":forecast.get('provenance_limitation'),"failures":failures}
    (report/'report_data.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

    card=lambda k,label,fmt: f"<div class=card><small>{label}</small><strong>{fmt(ci[k]['mean'])}</strong><span>95% CI {fmt(ci[k]['ci95'][0])} to {fmt(ci[k]['ci95'][1])}</span></div>"
    html=f'''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width"><title>RL Portfolio Research</title><style>
    :root{{--bg:#07101d;--panel:#101c2d;--ink:#e8f0fa;--muted:#9eb0c7;--cyan:#5bc0be;--red:#f05d5e;--gold:#ffd166}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,Segoe UI,sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 24px}}h1{{font-size:46px;line-height:1.05;margin:.2em 0}}h2{{margin-top:55px;border-top:1px solid #273951;padding-top:28px}}h3{{color:#bde7e5}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.14em}}.verdict{{border-left:5px solid var(--red);background:#211824;padding:20px 25px;font-size:19px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:13px}}.card,.note{{background:var(--panel);border:1px solid #223650;border-radius:12px;padding:17px}}.card strong{{display:block;font-size:26px}}.card span,small,.muted{{color:var(--muted)}}img{{width:100%;border:1px solid #263b55;border-radius:10px;background:#0b1220}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:9px 12px;border-bottom:1px solid #263950;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:#8de0dc}}.scroll{{overflow:auto}}code,pre{{background:#091321;color:#bce9e6}}pre{{padding:16px;overflow:auto;border-radius:9px}}a{{color:#79d7d3}}@media(max-width:600px){{h1{{font-size:34px}}.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><div class=eyebrow>Universal Backtester · walk-forward study</div><h1>Active equity RL:<br>negative result, useful evidence</h1>
    <p class=verdict><b>Executive conclusion.</b> Frozen PPO policy did not outperform simple long-only baselines or SPY on risk-adjusted out-of-sample evidence. Mean seed-fold return was {pct(fm.cumulative_return.mean())}; median Sharpe {num(fm.sharpe.median())}. Evidence does not support deployment or more broad tuning. Failure is reported without test-set retuning.</p>
    <div class=cards>{card('cumulative_return','RL mean test return',pct)}{card('sharpe','RL mean Sharpe',num)}{card('maximum_drawdown','RL mean max drawdown',pct)}{card('turnover','RL mean turnover',num)}</div>
    <h2>Comparison</h2><p class=note><b>Comparability warning.</b> Simple policies and SPY use matching Universal Backtester folds. GP-return and GBM-return forecast replay uses next-bar execution on frozen forecast CSVs, but strict OOS provenance cannot be independently proven; model-fit cutoffs and prediction-created timestamps are absent. They are contextual, not strict head-to-head OOS evidence.</p><img src="data:image/png;base64,{imgs['kpi']}">{table(comp,['Strategy','Median return','Median Sharpe','Worst drawdown'])}
    <h2>Walk-forward evidence</h2><p>Rolling folds: 2 years train, 6 months validation, 6 months untouched test; fold-local normalization; multiple frozen seeds. Bootstrap intervals resample 45 seed-fold observations with fixed seed. Fold dependence means intervals are descriptive, not iid guarantees.</p><div class=grid><img src="data:image/png;base64,{imgs['folds']}"><img src="data:image/png;base64,{imgs['equity']}"><img src="data:image/png;base64,{imgs['dd']}"></div>{table(foldrows,['Fold','Seeds','Return mean','Sharpe mean','Max DD mean','Holding days','Turnover'])}
    <h2>Data and methodology</h2><p>Snapshot <code>{escape(str(manifest.get('snapshot_id')))}</code>, content hash <code>{escape(str(manifest.get('content_sha256')))}</code>. Daily adjusted OHLCV used for full history because Yahoo hourly retention cannot sustain requested 3-year fold. Universe is deterministic but based on surviving current large caps; survivorship bias remains. Corporate actions follow snapshot adjustment semantics. Ideal fills are modeled because reproducible historical bid/ask data were unavailable.</p>
    <h3>Leakage and execution safeguards</h3><ul><li>Observation contains current/past bars only; decisions cannot fill on observed close.</li><li>Market orders execute no earlier than next bar open. Limits cross only later OHLC ranges under deterministic conservative rules.</li><li>Preprocessing fits training slice only. Purge/embargo separates lifecycle slices.</li><li>Test access audit records one frozen evaluation; test metrics never drive reward, early stopping, or Optuna.</li><li>Accounting asserts cash, positions, equity, gross exposure ≤ equity, pending orders, realized/unrealized P&amp;L.</li></ul>
    <h2>Architecture and reward</h2><p>One shared cross-asset PPO policy observes causal per-asset OHLCV, returns, RSI, Bollinger, momentum, volatility and volume features plus portfolio state. Assets share one simultaneous observation, permitting implicit cross-asset comparison; this price-only baseline does not add an explicit SPY-relative feature. Continuous target weights express long/short/cash without leverage. Feed-forward PPO was chosen for maintained CPU-friendly implementation and continuous control; recurrent PPO adds cost and leakage-state complexity, while off-policy SAC/TD3 were deferred until environment correctness and baseline value were proven.</p><p>Compact reward starts with log equity change, then small drawdown, turnover and holding-target terms. Holding objective targets 3–5 trading days softly; profitability and risk remain dominant. Validation objective combines return, drawdown and fold stability.</p>
    <h2>Optimization and sensitivity</h2><div class=grid><img src="data:image/png;base64,{imgs['optuna']}"><img src="data:image/png;base64,{imgs['importance']}"><img src="data:image/png;base64,{imgs['stability']}"></div><p>{len(trials)} rough trials were recorded with pruning. Search stopped below 150-trial cap because full OOS evidence was weak and further compute lacked credible validation payoff. Correlation importance is descriptive, not causal. Neighborhood result isolated-spike flag: <b>{escape(str(stabs[0].get('isolated_spike')))}</b>.</p>
    <h2>Trading behavior</h2><div class=grid><img src="data:image/png;base64,{imgs['holding']}"><img src="data:image/png;base64,{imgs['exposure']}"><img src="data:image/png;base64,{imgs['contrib']}"></div><p>Mean holding time {num(fm.average_holding_days.mean())} days; median {num(fm.median_holding_days.median())}. Display caps histogram at 30 days; machine-readable report intentionally excludes huge raw holding arrays. Mean gross exposure {pct(fm.average_gross_exposure.mean())}, mean cash ratio {pct(fm.average_cash_ratio.mean())}, mean turnover {num(fm.turnover.mean())}.</p>
    <h2>Per-ticker attribution</h2><p>Reporting-only replay loaded every frozen <code>best_model.zip</code>; no model was retrained and no test result changed selection. Contribution equals engine average-cost realized P&amp;L plus ending unrealized P&amp;L marked at test-end close. FIFO matching supplies trade count, win rate, holding bars, and long/short matched diagnostics. Across {attribution.get('seed_fold_count',0)} seed-folds, maximum conservation residual was ${attribution.get('max_abs_reconciliation_residual',float('nan')):.2e}.</p><img src="data:image/png;base64,{imgs['ticker']}">{table(ticker_rows,['Ticker','Total P&L','Realized','Ending unrealized','Orders','FIFO trades','Win rate'])}
    <h2>Regime analysis</h2><p>Labels are descriptive: SPY fold return sign gives Bull/Bear; RL realized volatility above/below cross-fold median gives high/low volatility. No regime label enters training.</p>{table(regimes,['fold','regime','rl_return','rl_sharpe','spy_return'])}
    <h2>Failed experiments and limitations</h2><ul><li>Hourly full protocol rejected: Yahoo retention insufficient; only feasibility use is credible.</li><li>RL failed to beat equal-weight, momentum, or SPY consistently; seed/fold dispersion remains material.</li><li>GP/GBM frozen forecasts lack independently verifiable strict OOS provenance.</li><li>Current deterministic universe creates survivorship bias; historical membership was unavailable.</li><li>No fees, borrow cost, margin interest, spread, slippage, partial fills, liquidity or market impact. Ideal fills overstate deployability.</li><li>Only 45 seed-fold outcomes; overlapping market regimes reduce effective sample size.</li><li>Price-only baseline; news rejected pending reproducible timestamped historical coverage.</li></ul><p>Recorded failure events: {len(failures)}. Full structured records remain in <code>failures.jsonl</code>.</p>
    <h2>Reproduction</h2><pre>python -m rl_portfolio_management.data_pipeline
python -m rl_portfolio_management.run_baselines
python -m rl_portfolio_management.optimize_ppo
python -m rl_portfolio_management.stability_check
python -m rl_portfolio_management.final_evaluate
python -m rl_portfolio_management.replay_forecast_baselines
python -m rl_portfolio_management.extract_ticker_attribution
python -m rl_portfolio_management.generate_report
python -m unittest rl_portfolio_management.tests.test_generate_report</pre><p class=muted>Machine-readable companion: report/report_data.json. Report is self-contained; every chart is embedded as base64 PNG. Generated from local frozen artifacts.</p>
    </main></body></html>'''
    (report/'report.html').write_text(html,encoding='utf-8')
    print(report/'report.html')

if __name__=='__main__': main()
