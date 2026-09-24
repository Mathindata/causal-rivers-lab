import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 01 · Getting to know CausalRivers through a causal lens

**Project:** causal discovery on real river networks (baselines → new methods → paper → course notes)
**Notebook goal:** before running any causal discovery method, understand what the data *is*, what the ground truth *means*, and which properties of the problem will make a method succeed, fail, or look good for the wrong reasons.

Every section ends with a short **"Causal takeaway"**. Section 8 collects them into a checklist that later notebooks (baselines, new methods) must respect.

| Section | Question |
|---|---|
| 1 | What is the causal question here? |
| 2 | What does the ground-truth graph look like? |
| 3 | Where are the stations? |
| 4 | What do the time series look like (gaps, scale, oddities)? |
| 5 | Can we *see* causation in one river chain? |
| 6 | Which causal assumptions hold, and which break? |
| 7 | Are there shortcuts that make the benchmark easy to game? |
| 8 | Summary: what makes a method acceptable or meaningless here |

Runs on **Google Colab** or **Kaggle** (turn Internet on). Needs about 3 GB of RAM.
""")

md("## 0 · Setup")
code(r"""
import os, sys, subprocess, json, pickle, random, warnings
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if IN_KAGGLE else ("/content" if IN_COLAB else os.getcwd())

REPO = os.path.join(WORK, "causalrivers")          # official benchmark package
DATA = os.path.join(REPO, "product")               # graphs, metadata, time series
CACHE = os.path.join(WORK, "cache")                # our resampled copies
FIG = os.path.join(WORK, "figures", "01_eda")
RES = os.path.join(WORK, "results")
for d in (CACHE, FIG, RES):
    os.makedirs(d, exist_ok=True)

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if not os.path.exists(REPO):
    sh(f"git clone -q --depth 1 https://github.com/CausalRivers/causalrivers.git {REPO}")
if not os.path.exists(os.path.join(DATA, "rivers_ts_east_germany.csv")):
    # ~130 MB download, ~1.1 GB unzipped
    sh(f"cd {REPO} && wget -q https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip "
       f"&& unzip -q product.zip && rm product.zip")
sys.path.append(REPO)
print(sorted(os.listdir(DATA)))
""")

code(r"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

SEED = 42
random.seed(SEED); np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
REGIONS = ["east_germany", "bavaria"]
NICE = {"east_germany": "East Germany", "bavaria": "Bavaria", "flood": "Flood (Elbe 2024)"}
COL = {"east_germany": "#1f6f8b", "bavaria": "#c05a2b"}

def savefig(name):
    plt.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")

SUMMARY = {}   # key numbers, saved to results/01_eda_summary.json at the end
""")
code(r"""
# Lagged cross-correlation helpers (explained in Section 6)
def _prep(s):
    d = s.diff()
    mask = d.notna().values.astype(np.float64)
    z = ((d - d.mean()) / d.std()).fillna(0).values.astype(np.float64)
    return z, mask

def xcorr(x, y, max_lag):
    # corr(x_t, y_{t+L}) for L = -max_lag..max_lag.  Positive best lag  => x leads y.
    zx, mx = _prep(x); zy, my = _prep(y)
    n = len(zx); nfft = 1 << (2 * n - 1).bit_length()
    num = np.fft.irfft(np.conj(np.fft.rfft(zx, nfft)) * np.fft.rfft(zy, nfft), nfft)
    cnt = np.fft.irfft(np.conj(np.fft.rfft(mx, nfft)) * np.fft.rfft(my, nfft), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, num[lags % nfft] / np.maximum(np.round(cnt[lags % nfft]), 1)

def peak(x, y, max_lag):
    lags, c = xcorr(x, y, max_lag)
    i = np.argmax(c)
    return lags[i], c[i]
""")

md(r"""
## 1 · The causal question

- **Variables:** each variable is one gauging station, measuring river **discharge** (m³/s, water volume per second) every 15 minutes, 2019 to 2023.
- **Ground truth:** station A → station B if water measured at A flows *directly* to B, with no other *measured* station in between. So the true graph is built from river geography, not from the data. That is what makes this benchmark valuable: the labels are independent of any statistical assumption.
- **Task (summary graph discovery):** given only the time series of a set of stations, recover the directed graph. The benchmark scores the *summary graph* (is there an edge A → B at any lag?), not the lag itself, and it ignores self-loops (every river is autocorrelated with itself).
- **Why this is causal and not just correlation:** an intervention on A (open a dam upstream) changes B later. An intervention on B does nothing to A. Two tributaries that join at C are correlated (same rain) but neither causes the other.

The last point is the heart of the problem, and we will measure it directly in Section 6.
""")

md("## 2 · The ground-truth graphs")
code(r"""
G = {r: pickle.load(open(os.path.join(DATA, f"rivers_{r}.p"), "rb")) for r in REGIONS + ["flood"]}
META = {r: pd.read_csv(os.path.join(DATA, f"rivers_meta_{r}.csv"), index_col=0) for r in REGIONS + ["flood"]}

def graph_stats(g):
    od = pd.Series(dict(g.out_degree())); idg = pd.Series(dict(g.in_degree()))
    n = g.number_of_nodes()
    return {
        "nodes": n, "edges": g.number_of_edges(),
        "density": round(g.number_of_edges() / (n * (n - 1)), 5),
        "is_DAG": nx.is_directed_acyclic_graph(g),
        "components": nx.number_weakly_connected_components(g),
        "largest_component": max(len(c) for c in nx.weakly_connected_components(g)),
        "longest_path": nx.dag_longest_path_length(g),
        "sinks (out=0)": int((od == 0).sum()),
        "out-degree = 1": f"{(od == 1).mean():.1%}",
        "splits (out>=2)": int((od >= 2).sum()),
        "sources (in=0)": int((idg == 0).sum()),
        "max in-degree": int(idg.max()),
    }

stats = pd.DataFrame({NICE[r]: graph_stats(G[r]) for r in G})
SUMMARY["graph_stats"] = stats.astype(str).to_dict()
stats
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, kind in zip(axes, ["in", "out"]):
    width = 0.4
    for i, r in enumerate(REGIONS):
        deg = pd.Series(dict(G[r].in_degree() if kind == "in" else G[r].out_degree()))
        vc = deg.value_counts(normalize=True).sort_index()
        ax.bar(vc.index + (i - 0.5) * width, vc.values, width, label=NICE[r], color=COL[r])
    ax.set_title(f"{kind}-degree distribution"); ax.set_xlabel(f"{kind}-degree"); ax.set_ylabel("share of stations")
    ax.legend()
plt.tight_layout(); savefig("degree_distributions"); plt.show()
""")

code(r"""
# How were edges built? origin = construction step; quality > 0 means estimated by hand
for r in REGIONS:
    e = pd.DataFrame([d for *_, d in G[r].edges(data=True)])
    print(f"{NICE[r]:13s} edge origin counts:", e["origin"].value_counts().sort_index().to_dict(),
          "| edges with estimated geometry:", f"{(e['quality_geo'] > 0).mean():.0%}")
""")

md(r"""
**Causal takeaway (graph structure)**

1. **Very sparse.** Density is well under 1%. In a benchmark subgraph of 5 stations, most of the 20 possible directed edges are absent, so class imbalance matters for the choice of metric.
2. **Almost an in-forest.** About 90% of stations have exactly one downstream neighbour. Water does not split, except at a few man-made channels or river branches (the "splits" row). This is a very strong structural prior that generic methods do not use. It is our first candidate for a new method (Phase 2).
3. **Splits create true confounders.** A station with out-degree ≥ 2 is a common cause of its children. The benchmark builds a special "confounder" subset from exactly these nodes.
4. **Many sources and a few big hubs** (in-degree up to 8 to 10): large rivers collect many measured tributaries. Methods that cap the number of parents will struggle at hubs.
5. **Deep chains** (longest path 16 to 24 stations): direct vs indirect effects will be a real issue. A → B → C makes A and C dependent, but A → C is *not* a true edge.
""")

md("## 3 · Where are the stations?")
md(r"""
Note: the README says metadata column `X` is longitude, but the values (≈ 47 to 54) are clearly **latitude**, and `Y` (≈ 9 to 15) is longitude. The node attribute `p` is also `(lat, lon)`. We plot with that in mind.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, r in zip(axes, REGIONS):
    g = G[r]
    pos = {n: (d["p"][1], d["p"][0]) for n, d in g.nodes(data=True)}   # (lon, lat)
    segs = [(pos[u], pos[v]) for u, v in g.edges]
    ax.add_collection(LineCollection(segs, colors="#5b8fb9", linewidths=0.8, alpha=0.8))
    origins = pd.Series({n: d["origin"] for n, d in g.nodes(data=True)})
    for o, cmap_i in zip(sorted(origins.unique()), range(20)):
        idx = origins[origins == o].index
        ax.scatter([pos[n][0] for n in idx], [pos[n][1] for n in idx], s=7, label=o, color=plt.cm.tab10(cmap_i % 10), zorder=3)
    splits = [n for n in g if g.out_degree(n) >= 2]
    ax.scatter([pos[n][0] for n in splits], [pos[n][1] for n in splits], s=45, facecolors="none",
               edgecolors="red", linewidths=1.2, zorder=4, label="split (confounder)")
    ax.set_title(f"{NICE[r]}: {g.number_of_nodes()} stations, {g.number_of_edges()} edges")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect(1.5); ax.autoscale()
    ax.legend(fontsize=7, title="data source", loc="best", markerscale=1.5)
plt.tight_layout(); savefig("station_maps"); plt.show()
""")
md(r"""
**Causal takeaway (space):** stations close together share weather. Geographic closeness is *not* causal closeness: two nearby stations on different rivers are neighbours in space but have no edge. Any method or prior that uses distance must use *distance along the river*, not straight-line distance.
""")

md("## 4 · The time series")
md(r"""
The raw files are large (0.5 to 0.6 GB each). We read them in chunks as `float32`, profile the raw 15-minute data, then cache an hourly version. The East Germany raw data is kept in memory for the resolution study in Section 6.
""")
code(r"""
def load_raw(region):
    parts = []
    for ch in pd.read_csv(os.path.join(DATA, f"rivers_ts_{region}.csv"), index_col=0, chunksize=20000):
        ch.index = pd.to_datetime(ch.index)
        parts.append(ch.astype("float32"))
    return pd.concat(parts)

def profile(df):
    step = pd.Series(df.index).diff().value_counts()
    nan_rate = df.isna().mean()
    return {
        "rows": len(df), "stations": df.shape[1],
        "start": str(df.index.min()), "end": str(df.index.max()),
        "regular 15-min grid": bool(len(step) == 1),
        "overall NaN rate": f"{df.isna().values.mean():.1%}",
        "median station NaN rate": f"{nan_rate.median():.1%}",
        "stations with >20% NaN": int((nan_rate > 0.2).sum()),
        "negative values": int((df < 0).values.sum()),
        "stations with negatives": int((df < 0).any().sum()),
        "exact zeros": int((df == 0).values.sum()),
        "median share of 15-min changes that are exactly 0": f"{((df.diff() == 0).sum() / df.diff().notna().sum()).median():.0%}",
    }

RAW, H = {}, {}
prof = {}
for r in REGIONS:
    raw = load_raw(r)
    prof[NICE[r]] = profile(raw)
    H[r] = raw.resample("1h").mean()
    H[r].to_parquet(os.path.join(CACHE, f"{r}_1h.parquet"))
    if r == "east_germany":
        RAW[r] = raw            # keep for the resolution study
    del raw
SUMMARY["ts_profile"] = prof
pd.DataFrame(prof)
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
for r in REGIONS:
    h = H[r]
    axes[0].hist(h.isna().mean() * 100, bins=40, alpha=0.6, color=COL[r], label=NICE[r])
    daily_missing = h.isna().resample("1D").mean().mean(axis=1) * 100
    axes[1].plot(daily_missing.index, daily_missing.values, color=COL[r], lw=0.8, label=NICE[r])
    med = h.median().clip(lower=1e-3)
    axes[2].hist(np.log10(med), bins=40, alpha=0.6, color=COL[r], label=NICE[r])
axes[0].set(title="missing data per station", xlabel="% of hours missing", ylabel="stations")
axes[1].set(title="missing data over time", ylabel="% of stations missing")
axes[2].set(title="typical flow size (median)", xlabel="log10 median discharge (m³/s)", ylabel="stations")
for a in axes: a.legend(fontsize=8)
plt.tight_layout(); savefig("missingness_and_scale"); plt.show()
""")

code(r"""
# Where do negative discharges come from?
neg = (H["east_germany"] < 0).mean()
worst = neg[neg > 0].sort_values(ascending=False).head(8)
m = META["east_germany"]
print(pd.DataFrame({"river": m.loc[worst.index.astype(int), "R"].values,
                    "source": m.loc[worst.index.astype(int), "O"].values,
                    "share of hours < 0": worst.round(3).values}, index=worst.index))
""")
code(r"""
# Step-like series: how often does the flow not change at all from one 15-min reading to the next?
zero_share = {r: (H[r].diff() == 0).sum() / H[r].diff().notna().sum() for r in REGIONS}
raw = RAW["east_germany"]
zs = (raw.diff() == 0).sum() / raw.diff().notna().sum()
flat = (zs - zs.quantile(0.9)).abs().idxmin(); smooth = (zs - zs.quantile(0.1)).abs().idxmin()   # typical, not extreme
fig, axes = plt.subplots(1, 2, figsize=(13, 3.2))
for r in REGIONS:
    axes[0].hist(zero_share[r] * 100, bins=40, alpha=0.6, color=COL[r], label=NICE[r])
axes[0].set(title="share of hourly changes that are exactly zero", xlabel="% of hours", ylabel="stations"); axes[0].legend(fontsize=8)
rng_ = raw[flat].rolling(4 * 24 * 3).max() - raw[flat].rolling(4 * 24 * 3).min()   # most active 3-day window
t0 = rng_.idxmax() - pd.Timedelta("3D")
for sid, c in [(flat, "#c0392b"), (smooth, "#2e8b57")]:
    s = raw.loc[t0: t0 + pd.Timedelta("3D"), sid]
    axes[1].plot(s.index, (s - s.min()) / (s.max() - s.min() + 1e-9), color=c, lw=1, drawstyle="steps-post",
                 label=f"{sid} ({META['east_germany'].loc[int(sid), 'R']}), {zs[sid]:.0%} zero steps")
axes[1].set(title="3 days at 15-min resolution (min-max scaled)"); axes[1].legend(fontsize=8); axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); savefig("step_like_series"); plt.show()
SUMMARY["median_share_zero_hourly_changes"] = {NICE[r]: round(float(zero_share[r].median()), 3) for r in REGIONS}
""")

md(r"""
**Causal takeaway (data quality)**

- **Missing data is uneven.** East Germany has far more gaps than Bavaria, and gaps are clustered in time. Methods that need complete windows will silently drop data; interpolation creates fake smooth segments that look like perfect predictability. Keep the missingness pattern in mind when comparing regions.
- **Scale spans about five orders of magnitude** (small creeks to the Elbe). Any score based on raw covariance is dominated by big rivers. Whether we normalize is a modelling decision with consequences (Section 7).
- **Many series move in steps.** In East Germany the median station has no change at all in most hours (about half the hours in Bavaria): discharge is derived from water level readings with limited precision, and some rivers are regulated by weirs and reservoirs. So the *effective* resolution is coarser than 15 minutes, differenced series are sparse and full of ties (a problem for rank-based tests), and small causal effects can be hidden below the measurement step.
- **Negative discharges** appear mostly in lowland rivers near the Baltic coast (e.g. Peene, Uecker) where backwater and wind-driven flow reversal happen. Physically, the causal direction can locally flip there. These stations are a known source of label noise for "downstream" reasoning.
""")

md("## 5 · Seeing causation in one river chain")
code(r"""
def find_chain(g, h, length=4, max_nan=0.02):
    # pick the chain whose weakest edge still shows a clear lagged link (a good teaching example)
    # small and medium rivers (median < 100 m3/s) show sharper flood waves than the Elbe
    ok = {n for n in g if str(n) in h and h[str(n)].isna().mean() < max_nan and h[str(n)].median() < 100}
    best = None
    for s in sorted(ok):
        chain = [s]
        while len(chain) < length:
            nxt = [v for v in g.successors(chain[-1]) if v in ok]
            if not nxt: break
            chain.append(nxt[0])
        if len(chain) == length:
            sc = []
            for a, b in zip(chain[:-1], chain[1:]):
                L, c = peak(h[str(a)], h[str(b)], 72)
                sc.append(c if L > 0 else 0)
            if best is None or min(sc) > best[1]:
                best = (chain, min(sc))
    return best[0]

g, h = G["east_germany"], H["east_germany"]
chain = find_chain(g, h)
print("chain:", " -> ".join(f"{n} ({g.nodes[n]['R']})" for n in chain))

t_peak = h[str(chain[0])].iloc[24 * 7: -24 * 7].idxmax()        # biggest upstream event, away from the edges
win = h.loc[t_peak - pd.Timedelta("3D"): t_peak + pd.Timedelta("4D"), [str(n) for n in chain]]
print("peak times:", {n: str(win[str(n)].idxmax()) for n in chain})
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for i, n in enumerate(chain):
    s = win[str(n)]
    axes[0].plot(s.index, s.values, lw=1.4, label=f"{n} {g.nodes[n]['R']}", color=plt.cm.viridis(i / len(chain)))
    z = (s - s.min()) / (s.max() - s.min())
    axes[1].plot(s.index, z.values, lw=1.4, color=plt.cm.viridis(i / len(chain)))
    axes[1].axvline(s.idxmax(), color=plt.cm.viridis(i / len(chain)), ls="--", lw=1)
axes[0].set(title="raw discharge (upstream → downstream)", ylabel="m³/s"); axes[0].legend(fontsize=8)
axes[1].set(title="min-max scaled: peaks (dashed) arrive later downstream", ylabel="scaled")
for a in axes: a.tick_params(axis="x", rotation=30)
plt.tight_layout(); savefig("chain_flood_event"); plt.show()
""")
md(r"""
**Causal takeaway (the signal):** the causal signal is visible *in events*: a flood wave travels downstream, arrives later, and grows as tributaries join. Between events, flow is slow and smooth, and carries little information about direction. So most of the causal evidence sits in a small fraction of the time points. Methods that weight all time points equally dilute it.
""")

md("## 6 · Which causal assumptions hold?")
md(r"""
Most time-series causal discovery methods assume some mix of: **temporal precedence** (causes come first), **faithfulness** (true edges show up as dependence), **causal sufficiency** (no hidden common causes), **stationarity**, and a model class (**linear / Gaussian / additive noise**). We test each one with the ground truth in hand.

The main tool is **lagged cross-correlation of first differences** (helpers defined in Setup). We difference the series because raw discharge is dominated by slow seasonal level changes that make almost everything correlated. For speed we compute it with FFT; missing values are masked and each lag is normalized by the number of overlapping points.
""")

md(r"""
### 6.1 Faithfulness vs confounding: how strong is dependence for each kind of pair?

We compare pair types that a method must tell apart:

| pair type | true edge? | why they could be dependent |
|---|---|---|
| **direct edge** A → B | yes | water flows A → B |
| **2-hop** A → B → C (A, C) | no | indirect effect through B |
| **co-parents** A → C ← B (A, B) | no | same rain (hidden confounder) |
| **split siblings** C ← A → B (C, B) | no | common upstream cause A |
| **unconnected, same basin** | no | shared regional weather |
| **unconnected, different basins** | no | shared large-scale weather |
""")
code(r"""
def pair_families(g, h, n=150, max_nan=0.3):
    ok = {v for v in g if str(v) in h and h[str(v)].isna().mean() < max_nan}
    comp = {v: i for i, c in enumerate(nx.weakly_connected_components(g)) for v in c}
    anc = {v: nx.ancestors(g, v) for v in g}
    fam = {"direct edge": [(u, v) for u, v in g.edges if u in ok and v in ok]}
    fam["2-hop (indirect)"] = [(u, w) for u, v in g.edges for w in g.successors(v) if u in ok and w in ok]
    fam["co-parents"] = [(a, b) for c in g for i, a in enumerate(sorted(g.predecessors(c)))
                         for b in sorted(g.predecessors(c))[i + 1:] if a in ok and b in ok]
    fam["split siblings"] = [(a, b) for c in g for i, a in enumerate(sorted(g.successors(c)))
                             for b in sorted(g.successors(c))[i + 1:] if a in ok and b in ok]
    nodes = sorted(ok); same, diff = [], []
    rng = random.Random(SEED)
    while len(same) < n or len(diff) < n:
        a, b = rng.sample(nodes, 2)
        if a in anc[b] or b in anc[a]: continue
        (same if comp[a] == comp[b] else diff).append((a, b))
    fam["unconnected, same basin"] = same[:n]; fam["unconnected, diff. basin"] = diff[:n]
    return {k: random.Random(SEED).sample(v, min(n, len(v))) for k, v in fam.items()}

MAXLAG_H = 72   # hours
dep = []
for r in REGIONS:
    fams = pair_families(G[r], H[r])
    for fname, pairs in fams.items():
        for a, b in pairs:
            L, c = peak(H[r][str(a)], H[r][str(b)], MAXLAG_H)
            dep.append({"region": NICE[r], "family": fname, "a": a, "b": b, "lag_h": L, "abs_lag": abs(L), "corr": c})
dep = pd.DataFrame(dep)
tab = dep.groupby(["family", "region"])["corr"].median().unstack().round(3)
order = ["direct edge", "2-hop (indirect)", "co-parents", "split siblings", "unconnected, same basin", "unconnected, diff. basin"]
tab = tab.loc[order]; tab["n pairs (EG)"] = dep[dep.region == "East Germany"].groupby("family").size().loc[order]
SUMMARY["dependence_by_family_median_corr"] = tab.to_dict()
tab
""")
code(r"""
fig, ax = plt.subplots(figsize=(11, 4))
pos = np.arange(len(order))
for i, r in enumerate(REGIONS):
    data = [dep[(dep.region == NICE[r]) & (dep.family == f)]["corr"].values for f in order]
    bp = ax.boxplot(data, positions=pos + (i - 0.5) * 0.35, widths=0.3, patch_artist=True, showfliers=False)
    for b in bp["boxes"]: b.set_facecolor(COL[r]); b.set_alpha(0.6)
ax.set_xticks(pos); ax.set_xticklabels(order, rotation=15)
ax.set_ylabel("peak lagged corr. of differences")
ax.set_title("Only the first column is a true edge. 2-hop pairs and co-parents overlap heavily with it.")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=COL[r], alpha=0.6) for r in REGIONS], [NICE[r] for r in REGIONS])
plt.tight_layout(); savefig("dependence_by_pair_family"); plt.show()
""")
md(r"""
**Causal takeaway (faithfulness and confounding):**

- True edges do produce dependence, but it is **weak and very spread out**: a quarter of true edges sit below about 0.1, no better than unconnected pairs. Faithfulness holds on average, not reliably per edge.
- **Co-parents are the main trap.** Two tributaries that join are *not* causally linked, yet their dependence distribution overlaps heavily with that of true edges, because the same storm hits both. Rainfall is an **unobserved confounder**, so causal sufficiency is violated everywhere. A method that thresholds on pairwise dependence will produce many false edges between co-parents.
- **Indirect (2-hop) pairs** are dependent too. Methods that do not condition on intermediate stations (pairwise Granger, cross-correlation) will add false shortcut edges A → C.
- **Split siblings are surprisingly weakly dependent**, even though they share an upstream cause. Many splits are managed channels (flood canals, diversions) that only carry water above a threshold, so the confounding is switched on and off by human control. There are few such pairs, so they matter mostly in the benchmark's confounder subset.
""")

md("### 6.2 Temporal precedence and the role of time resolution")
code(r"""
RES_LIST = [("15min", 0.25), ("1h", 1), ("6h", 6), ("1D", 24)]
raw = RAW["east_germany"]; g = G["east_germany"]
edges = dep[(dep.region == "East Germany") & (dep.family == "direct edge")][["a", "b"]].values.tolist()

lag_rows = []
for res, hours in RES_LIST:
    d = raw if res == "15min" else raw.resample(res).mean()
    K = int(np.ceil(96 / hours))                  # look 4 days back and forward
    for a, b in edges:
        L, c = peak(d[str(a)], d[str(b)], K)
        lag_rows.append({"res": res, "a": a, "b": b, "lag_steps": L, "lag_hours": L * hours, "corr": c,
                         "dist": g.edges[a, b]["geo_distance"]})
lags = pd.DataFrame(lag_rows)
direction = lags.assign(sign=np.sign(lags.lag_steps)).groupby("res")["sign"].value_counts(normalize=True).unstack().fillna(0)
direction = direction.rename(columns={1: "correct (A leads B)", 0: "lag 0 (no direction)", -1: "wrong (B leads A)"})
direction = direction.loc[[r for r, _ in RES_LIST]].round(3)
direction["median corr"] = lags.groupby("res")["corr"].median().loc[direction.index].round(3)
direction["median lag (h), correct only"] = lags[lags.lag_steps > 0].groupby("res")["lag_hours"].median().loc[direction.index]
SUMMARY["direction_by_resolution"] = direction.to_dict()
direction
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
direction[["correct (A leads B)", "lag 0 (no direction)", "wrong (B leads A)"]].plot.bar(
    stacked=True, ax=axes[0], color=["#2e8b57", "#bbbbbb", "#c0392b"], rot=0)
axes[0].set(title="Where is the peak cross-correlation of a true edge?", ylabel="share of true edges", xlabel="resolution")
axes[0].legend(fontsize=8)
sub = lags[(lags.res == "1h") & (lags.lag_steps > 0)]
axes[1].scatter(sub["dist"], sub["lag_hours"], s=12, alpha=0.6, color=COL["east_germany"])
rho = sub[["dist", "lag_hours"]].corr("spearman").iloc[0, 1]
axes[1].set(title=f"Travel time vs straight-line distance (1h, Spearman ρ = {rho:.2f})",
            xlabel="geo distance (degrees)", ylabel="peak lag (hours)")
SUMMARY["lag_vs_distance_spearman_1h"] = round(float(rho), 3)
plt.tight_layout(); savefig("lags_and_resolution"); plt.show()
""")
md(r"""
**Causal takeaway (time):**

- **Resolution changes the problem.** At fine resolution the lag is visible but noisy. At coarse resolution many causal effects become **contemporaneous** (lag 0), so the direction can no longer come from "cause before effect". Methods that only allow lagged edges (plain VAR, Granger) lose these edges; methods that allow contemporaneous edges (PCMCI+, VARLiNGAM, DYNOTEARS) need another source of direction.
- **Lags are heterogeneous**, from minutes to days. A single global `max_lag` is a poor fit: too small misses slow rivers, too large adds noise and parameters.
- **Straight-line distance only partly predicts travel time** (moderate rank correlation, see plot title). River length, slope, and reservoirs matter too. A physics-informed lag prior needs better inputs than coordinates; metadata column `D` (distance along the river) is a candidate for Phase 2.
""")

md("### 6.3 Model class: Gaussian? linear? stationary?")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for r in REGIONS:
    dd = H[r].diff()
    axes[0].hist(np.log10(dd.kurt().clip(lower=0.1)), bins=40, alpha=0.6, color=COL[r], label=NICE[r])
axes[0].axvline(np.log10(3), color="k", ls="--", lw=1); axes[0].text(np.log10(3), axes[0].get_ylim()[1] * 0.9, " Laplace", fontsize=8)
axes[0].set(title="tails of hourly changes (Gaussian = 0)", xlabel="log10 excess kurtosis", ylabel="stations"); axes[0].legend(fontsize=8)
SUMMARY["median_excess_kurtosis_of_diffs"] = {NICE[r]: round(float(H[r].diff().kurt().median()), 1) for r in REGIONS}

# nonlinearity: upstream at best lag vs downstream, log-log, for the chain's first edge
a, b = chain[0], chain[1]
L, _ = peak(H["east_germany"][str(a)], H["east_germany"][str(b)], MAXLAG_H)
x = H["east_germany"][str(a)]; y = H["east_germany"][str(b)].shift(-L)
ok = (x > 0) & (y > 0)
axes[1].hexbin(np.log10(x[ok]), np.log10(y[ok]), gridsize=45, bins="log", cmap="Blues")
axes[1].set(title=f"edge {a}→{b} at lag {L}h (log-log)", xlabel="log10 upstream", ylabel="log10 downstream")

# nonstationarity: rolling 90-day dependence of the same edge
dx, dy = x.diff(), y.diff()
roll = dx.rolling(24 * 90, min_periods=24 * 30).corr(dy)
axes[2].plot(roll.index, roll.values, color=COL["east_germany"], lw=1)
axes[2].set(title=f"edge {a}→{b}: 90-day rolling corr. of changes", ylabel="corr")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); savefig("model_class_checks"); plt.show()
""")
code(r"""
# seasonality: each station's monthly mean relative to its own overall mean
fig, ax = plt.subplots(figsize=(7, 3.2))
for r in REGIONS:
    h = H[r]
    rel = h.groupby(h.index.month).mean() / h.mean()
    ax.plot(rel.index, rel.median(axis=1), marker="o", color=COL[r], label=NICE[r])
    ax.fill_between(rel.index, rel.quantile(0.25, axis=1), rel.quantile(0.75, axis=1), color=COL[r], alpha=0.15)
ax.set(title="Seasonal cycle (median station, IQR band)", xlabel="month", ylabel="monthly mean / overall mean")
ax.legend(); plt.tight_layout(); savefig("seasonality"); plt.show()
""")
md(r"""
**Causal takeaway (model class):**

- **Extremely heavy tails.** Hourly changes have excess kurtosis in the hundreds (a Gaussian has 0, a Laplace distribution 3). Pearson correlation and least squares are dominated by a few flood spikes. On the good side, strong non-Gaussianity is exactly what LiNGAM-style methods need to find direction; on the bad side, Gaussian conditional-independence tests (ParCorr) are mis-specified.
- **Nonlinear, state-dependent relations.** Upstream vs downstream is closer to linear in log space, and the relation changes between low flow and floods (routing depends on water level).
- **Nonstationary.** The strength of the same true edge switches on and off over time (high in wet winters, near zero in dry summers when the signal is below the measurement step), and there is a strong seasonal cycle driven by snowmelt and summer drought (Bavaria peaks later, with Alpine snowmelt). Any method assuming one fixed dependence over five years averages over different regimes. Selecting windows (e.g. only wet periods, which the benchmark supports via weather sub-selection) is a legitimate design choice to test.
""")

md("## 7 · Shortcuts: could a method score well for the wrong reason?")
md(r"""
Reisach et al. (NeurIPS 2021, *"Beware of the Simulated DAG!"*) showed that on many synthetic benchmarks, variance increases along the causal order, so simply sorting variables by variance recovers the direction. Rivers have a *physical* version of this: **water accumulates downstream**. Let us measure how much direction information is available *without any causal reasoning at all*.
""")
code(r"""
def shortcut_table(r):
    g, h = G[r], H[r]
    E = [(u, v) for u, v in g.edges if str(u) in h and str(v) in h]
    med, dvar = h.median(), h.diff().var()
    z = (h - h.mean()) / h.std()
    ac = {c: z[c].diff().autocorr(1) for c in h.columns}    # smoothness survives normalization
    el = [(g.nodes[u]["H"], g.nodes[v]["H"]) for u, v in E]
    el = [(a, b) for a, b in el if not (np.isnan(a) or np.isnan(b))]
    return {
        "downstream has larger median flow": np.mean([med[str(v)] > med[str(u)] for u, v in E]),
        "downstream has larger change variance": np.mean([dvar[str(v)] > dvar[str(u)] for u, v in E]),
        "downstream is smoother (after z-scoring)": np.mean([ac[str(v)] > ac[str(u)] for u, v in E]),
        "upstream is higher (metadata elevation)": np.mean([a > b for a, b in el]),
    }

short = pd.DataFrame({NICE[r]: shortcut_table(r) for r in REGIONS}).round(3)
xc = direction.loc["1h", "correct (A leads B)"]
short.loc["lagged cross-corr. says A leads B (1h, EG only)"] = [xc, np.nan]
SUMMARY["shortcut_orientation_accuracy"] = short.to_dict()
short
""")
md(r"""
**Causal takeaway (shortcuts), probably the most important result of this notebook:**

- Given a true edge, **"the bigger river is downstream" gets the direction right about 9 times out of 10**, and elevation from the metadata does even better. That beats the lagged cross-correlation, which is the naive "causal" signal.
- **Normalizing removes the obvious shortcut.** After z-scoring, the smoothness test above is at chance (about 50%), so at least this simple statistic carries no direction information. Flexible models might still find subtler scale traces, which is why we keep checking.
- **Consequence:** a method evaluated on unnormalized data may orient edges correctly because of flow magnitude, not causal reasoning. This does not make it useless (in hydrology, "bigger is downstream" is real physics), but it has to be reported honestly:
  1. Always include **trivial baselines** (random, magnitude-sort, smoothness-sort) next to real methods.
  2. Report results **with and without normalization**.
  3. Separate **edge detection** (is there a link?) from **orientation** (which way?). The shortcuts only help with orientation; detecting the right pairs among co-parents and 2-hop pairs is the genuinely hard part.
""")

md("## 8 · Summary: what makes a method acceptable or meaningless here")
code(r"""
checklist = pd.DataFrame([
    ["Near in-forest graph (≈90% out-degree 1)", "Sec. 2", "Generic methods ignore it", "Use as structural prior (tree / arborescence search)"],
    ["Rain = hidden common cause", "Sec. 6.1 co-parents", "False edges between tributaries", "Condition on neighbours, model a shared driver, or use events"],
    ["Indirect paths A→B→C", "Sec. 6.1 2-hop", "Pairwise methods add shortcut edges", "Conditioning (PCMCI-type) or transitive reduction"],
    ["Contemporaneous effects at coarse resolution", "Sec. 6.2", "Lagged-only methods lose edges", "Allow lag-0 links, or use fine resolution"],
    ["Heterogeneous lags (minutes to days)", "Sec. 6.2", "One max_lag is wrong for most edges", "Per-edge lag, or physically informed lag prior"],
    ["Heavy tails, nonlinearity", "Sec. 6.3", "Gaussian CI tests mis-specified", "Rank/robust tests, log transform, non-Gaussian methods"],
    ["Nonstationarity, seasonality", "Sec. 6.3", "Averaging across regimes", "Window selection, event-based analysis"],
    ["Missing data, uneven by region", "Sec. 4", "Interpolation artifacts", "Report NaN handling; compare regions fairly"],
    ["Magnitude / smoothness shortcuts", "Sec. 7", "Right direction for the wrong reason", "Trivial baselines + normalized and raw runs"],
], columns=["property", "evidence", "risk for methods", "what we will do"])
checklist.to_csv(os.path.join(RES, "01_causal_checklist.csv"), index=False)
checklist
""")
md(r"""
### Rules for the next notebooks

1. **Tune on Bavaria, report on East Germany.** The official sampling config already names Bavaria the training graph and East Germany the test graph. We never tune hyperparameters on test subgraphs.
2. **Use the official subgraph samples and scoring code** so numbers match the paper and leaderboard.
3. **Mind the metric.** The official `score` function reports F1 at the *best* threshold (chosen using the labels), which is optimistic. We will report AUROC (threshold free) as the main number, and add F1 at a threshold fixed on the training region.
4. **Every table includes trivial baselines.** A method is *meaningless* here if it does not beat magnitude-sorting on orientation or cross-correlation on detection.
5. **Separate detection from orientation** in the analysis, and break results down by pair type (co-parents, 2-hop) using the families built in Section 6.1.
""")
code(r"""
with open(os.path.join(RES, "01_eda_summary.json"), "w") as f:
    json.dump(SUMMARY, f, indent=2, default=str)
dep.to_csv(os.path.join(RES, "01_pair_dependence.csv"), index=False)
lags.to_csv(os.path.join(RES, "01_edge_lags_by_resolution.csv"), index=False)
print("saved:", sorted(os.listdir(RES)), "\nfigures:", sorted(os.listdir(FIG)))
""")
md(r"""
### Teaching notes (for the course)

This notebook maps directly onto four intro lessons:

- **Correlation vs causation:** co-parents (Section 6.1) are a clean real-world example of confounding with a known answer.
- **Direct vs indirect effects:** 2-hop pairs show why conditioning matters.
- **Time and causality:** the resolution study (Section 6.2) shows that "cause precedes effect" depends on how we measure time.
- **Benchmarks can be gamed:** Section 7 is a good discussion exercise: *is "bigger river is downstream" causal knowledge or a cheat?*
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
nbf.write(nb, "/home/claude/01_eda_causal_lens.ipynb")
print("written", len(C), "cells")
