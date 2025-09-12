import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

csv = Path("context/sweeps/20250826/sweep_results.csv")
out = Path("context/sweep_plots"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv)

def col(df, name):
    # case-insensitive column getter
    m = {c.lower(): c for c in df.columns}
    return m.get(name.lower(), name)

dataset = col(df,"dataset")
tf_th   = col(df,"tf_th")
cj_th   = col(df,"cj_th")
sup     = col(df,"support_pct")
clean   = col(df,"cleaning") if "cleaning" in map(str.lower, df.columns) else None
bm25    = col(df,"bm25_mode") if "bm25_mode" in map(str.lower, df.columns) else None
bm25_b  = col(df,"bm25_b") if "bm25_b" in map(str.lower, df.columns) else None
bm25_topk = col(df,"bm25_topk") if "bm25_topk" in map(str.lower, df.columns) else None
canon   = None
for c in ["canon","canon_enabled"]:
    if c in map(str.lower, df.columns): canon = col(df,c); break

# Normalize friendly labels
if canon: df["_canon"] = df[canon].astype(str).str.lower().map(
    {"true":"on","1":"on","on":"on","false":"off","0":"off","off":"off"}).fillna(df[canon].astype(str))
else: df["_canon"] = "n/a"
df["_clean"] = df[clean].astype(str).str.lower() if clean else "n/a"
df["_bm25"]  = df[bm25].astype(str) if bm25 else "n/a"

def grouped_mean(frame, by, y):
    return frame.groupby(by)[y].mean().reset_index().sort_values(by if isinstance(by,list) else [by])

pdf = PdfPages(out/"sweep_plots.pdf")

def savefig(name):
    p = out/f"{name}.png"
    plt.savefig(p, bbox_inches="tight", dpi=160); pdf.savefig(plt.gcf(), bbox_inches="tight"); plt.close()

for ds in sorted(df[dataset].unique()):
    sub = df[df[dataset]==ds]
    if not sub.empty:
        t = grouped_mean(sub, tf_th, sup)
        if not t.empty:
            plt.figure(); plt.plot(t[tf_th], t[sup], marker="o")
            plt.xlabel("TF-IDF threshold"); plt.ylabel("Support % (mean)"); plt.title(f"{ds}: Support vs TF-IDF")
            savefig(f"{ds}_support_vs_tf")

        c = grouped_mean(sub, cj_th, sup)
        if not c.empty:
            plt.figure(); plt.plot(c[cj_th], c[sup], marker="o")
            plt.xlabel("char-3 Jaccard threshold"); plt.ylabel("Support % (mean)"); plt.title(f"{ds}: Support vs char-3")
            savefig(f"{ds}_support_vs_cj")

        for label,colname in [("Cleaning mode","_clean"),("Canon","_canon"),("BM25 mode","_bm25")]:
            if colname in sub.columns and sub[colname].nunique()>1:
                b = grouped_mean(sub, colname, sup)
                plt.figure(); plt.bar(b[colname].astype(str), b[sup])
                plt.xlabel(label); plt.ylabel("Support % (mean)"); plt.title(f"{ds}: {label} vs support")
                savefig(f"{ds}_bar_{colname.strip('_')}")

        if bm25 and (sub["_bm25"].str.lower()=="filter").any():
            subf = sub[sub["_bm25"].str.lower()=="filter"]
            if bm25_b and not grouped_mean(subf, bm25_b, sup).empty:
                g = grouped_mean(subf, bm25_b, sup)
                plt.figure(); plt.plot(g[bm25_b], g[sup], marker="o")
                plt.xlabel("BM25 b"); plt.ylabel("Support % (mean)"); plt.title(f"{ds}: Support vs BM25 b (filter)")
                savefig(f"{ds}_support_vs_bm25b")
            if bm25_topk and not grouped_mean(subf, bm25_topk, sup).empty:
                g = grouped_mean(subf, bm25_topk, sup)
                plt.figure(); plt.plot(g[bm25_topk], g[sup], marker="o")
                plt.xlabel("BM25 topk"); plt.ylabel("Support % (mean)"); plt.title(f"{ds}: Support vs BM25 topk (filter)")
                savefig(f"{ds}_support_vs_bm25topk")

pdf.close()
print("Wrote", out)
