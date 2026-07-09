"""
Análisis estadístico de las re-corridas del paper 1 (responde R2.4/R3.3).

Sobre los JSON de ./results/ (5 seeds x 4 folds por celda optimizador/dataset):
- Media y IC95 (t-Student) del test accuracy por celda.
- Test de Friedman por dataset (ranking global entre optimizadores).
- Wilcoxon signed-rank pareado (mismo seed+fold => mismos datos) QDPSOo vs cada
  baseline, con corrección de Holm por dataset.
- Robustez: mismos tests sobre las medias por seed (n=5), que elimina la
  dependencia entre folds de una misma semilla.
- MCW: Wilcoxon isomap vs pca por nº de componentes, y mejor config vs ORG.

Salida: results/stats_report.md (+ resumen por stdout).

Uso:
    conda activate pytorch_qpso_gpu
    python analyze_stats.py
"""

import json
import glob
from collections import defaultdict

import numpy as np
from scipy import stats

BENCH_DATASETS = ['circle', 'iris', 'wine', 'breast_cancer']
# Optimizadores con 5 seeds (QPSO lento queda fuera: 1 seed, solo timing)
OPTS = ['QPSOo', 'PSO_bound', 'PSO', 'Adam(ep100)', 'Adam(ep1000)']
REFERENCE = 'QPSOo'


def load_runs():
    """{(dataset, opt): {(seed, fold): test_acc}}"""
    runs = defaultdict(dict)
    for f in glob.glob('results/*.json'):
        d = json.load(open(f))
        opt = d['optimizer']
        tag = d['args'].get('tag', '')
        if tag:
            opt = f"{opt}({tag})"
        ds = d['args']['dataset']
        seed = d['args']['seed']
        for fold, acc in enumerate(d['test_acc_per_fold']):
            runs[(ds, opt)][(seed, fold)] = acc
    return runs


def ci95(x):
    x = np.asarray(x)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def holm(pvals):
    """Corrección de Holm; devuelve p ajustados en el orden de entrada."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, p)
        adj[idx] = running_max
    return adj


def paired_matrix(runs, ds, opts, keys=None):
    """Matriz (n_pares, n_opts) alineada por (seed, fold)."""
    common = set(runs[(ds, opts[0])].keys())
    for o in opts[1:]:
        common &= set(runs[(ds, o)].keys())
    keys = sorted(common) if keys is None else keys
    M = np.array([[runs[(ds, o)][k] for o in opts] for k in keys])
    return M, keys


def per_seed_means(runs, ds, opts):
    """Matriz (n_seeds, n_opts) con la media de folds por seed."""
    seeds = sorted({s for (s, _) in runs[(ds, opts[0])].keys()})
    M = np.zeros((len(seeds), len(opts)))
    for j, o in enumerate(opts):
        for i, s in enumerate(seeds):
            vals = [v for (sd, _), v in runs[(ds, o)].items() if sd == s]
            M[i, j] = np.mean(vals)
    return M


def wilcoxon_block(M, opts, ref_idx):
    """Wilcoxon pareado ref vs cada otro optimizador + Holm."""
    rows = []
    pvals = []
    others = [j for j in range(len(opts)) if j != ref_idx]
    for j in others:
        diff = M[:, ref_idx] - M[:, j]
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(M[:, ref_idx], M[:, j])
        pvals.append(p)
        rows.append([opts[j], np.mean(diff), stat, p])
    for row, p_adj in zip(rows, holm(np.array(pvals))):
        row.append(p_adj)
    return rows


def main():
    runs = load_runs()
    lines = []
    w = lines.append

    w("# Statistical Validation Report — Paper 1 resubmission")
    w("")
    w("Data: 5 seeds x 4 CV folds per optimizer/dataset cell (paired by "
      "identical seed+fold, i.e., identical data partitions). "
      "QDPSO (slow reference) excluded: single seed, timing study only.")
    w("")

    # ============ Benchmark: medias + IC95 ============
    w("## 1. Test accuracy: mean ± CI95 (n = 20 seed-fold measurements)")
    w("")
    w("| Dataset | " + " | ".join(OPTS) + " |")
    w("|---|" + "---|" * len(OPTS))
    for ds in BENCH_DATASETS:
        cells = []
        for o in OPTS:
            v = list(runs[(ds, o)].values())
            cells.append(f"{np.mean(v):.4f} ± {ci95(v):.4f}")
        w(f"| {ds} | " + " | ".join(cells) + " |")
    w("")

    # ============ Friedman ============
    w("## 2. Friedman test per dataset (H0: all optimizers rank equally)")
    w("")
    w("| Dataset | chi2 | p-value | Mean ranks (" + ", ".join(OPTS) + ") |")
    w("|---|---|---|---|")
    for ds in BENCH_DATASETS:
        M, _ = paired_matrix(runs, ds, OPTS)
        chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        # ranks: mayor accuracy = mejor rank (1)
        ranks = stats.rankdata(-M, axis=1).mean(axis=0)
        rk = ", ".join(f"{r:.2f}" for r in ranks)
        w(f"| {ds} | {chi2:.2f} | {p:.2e} | {rk} |")
    w("")

    # ============ Wilcoxon QDPSOo vs baselines ============
    w("## 3. Pairwise Wilcoxon signed-rank: QDPSOo vs each baseline")
    w("")
    w("Positive mean diff = QDPSOo better. Holm-corrected p-values per dataset.")
    w("")
    ref_idx = OPTS.index(REFERENCE)
    for ds in BENCH_DATASETS:
        M, keys = paired_matrix(runs, ds, OPTS)
        w(f"### {ds} (n = {len(keys)} pairs)")
        w("")
        w("| vs | mean diff | W | p | p (Holm) | significant (α=0.05) |")
        w("|---|---|---|---|---|---|")
        for opt, md, stat, p, p_adj in wilcoxon_block(M, OPTS, ref_idx):
            sig = "**yes**" if p_adj < 0.05 else "no"
            w(f"| {opt} | {md:+.4f} | {stat:.1f} | {p:.2e} | {p_adj:.2e} | {sig} |")
        w("")

    # ============ Robustez: por seed (consistencia direccional) ============
    w("## 4. Robustness check: per-seed means (n = 5 seeds)")
    w("")
    w("With n = 5 the Wilcoxon test cannot reach significance (minimum "
      "attainable p = 0.0625), so this check reports *directional "
      "consistency*: in how many of the 5 independent seeds the mean "
      "difference has the same sign as the main analysis.")
    w("")
    w("| Dataset | vs | mean diff | seeds agreeing | consistent |")
    w("|---|---|---|---|---|")
    for ds in BENCH_DATASETS:
        M = per_seed_means(runs, ds, OPTS)
        others = [j for j in range(len(OPTS)) if j != ref_idx]
        for j in others:
            diff = M[:, ref_idx] - M[:, j]
            md = diff.mean()
            agree = int(np.sum(np.sign(diff) == np.sign(md)))
            w(f"| {ds} | {OPTS[j]} | {md:+.4f} | {agree}/5 | "
              f"{'**yes**' if agree >= 4 else 'no'} |")
    w("")

    # ============ MCW ============
    w("## 5. MCW: reduction methods (QDPSOo, n = 20 seed-fold measurements)")
    w("")
    mcw = defaultdict(dict)
    for f in glob.glob('results/mcw_QPSOo_*.json'):
        d = json.load(open(f))
        red = d['args'].get('reduction') or 'ORG'
        comp = d['args'].get('components') or 84
        seed = d['args']['seed']
        for fold, acc in enumerate(d['test_acc_per_fold']):
            mcw[(red, comp)][(seed, fold)] = acc

    w("| Config | test acc mean ± CI95 |")
    w("|---|---|")
    order = [('ORG', 84)] + [(r, c) for c in [42, 21, 14, 7]
                             for r in ['isomap', 'pca', 'mds']]
    for key in order:
        v = list(mcw[key].values())
        w(f"| {key[0]}/{key[1]} | {np.mean(v):.4f} ± {ci95(v):.4f} |")
    w("")

    w("### Isomap vs PCA per component count (Wilcoxon, paired)")
    w("")
    w("| Components | mean diff (iso−pca) | p | significant |")
    w("|---|---|---|---|")
    for c in [42, 21, 14, 7]:
        keys = sorted(set(mcw[('isomap', c)]) & set(mcw[('pca', c)]))
        a = np.array([mcw[('isomap', c)][k] for k in keys])
        b = np.array([mcw[('pca', c)][k] for k in keys])
        stat, p = stats.wilcoxon(a, b)
        w(f"| {c} | {np.mean(a-b):+.4f} | {p:.2e} | "
          f"{'**yes**' if p < 0.05 else 'no'} |")
    w("")

    w("### Best config (isomap/7) vs original representation (ORG/84)")
    w("")
    keys = sorted(set(mcw[('isomap', 7)]) & set(mcw[('ORG', 84)]))
    a = np.array([mcw[('isomap', 7)][k] for k in keys])
    b = np.array([mcw[('ORG', 84)][k] for k in keys])
    stat, p = stats.wilcoxon(a, b)
    w(f"Mean diff = {np.mean(a-b):+.4f}, W = {stat:.1f}, p = {p:.2e} "
      f"({'significant' if p < 0.05 else 'not significant'} at α=0.05)")
    w("")

    report = "\n".join(lines)
    with open('results/stats_report.md', 'w') as f:
        f.write(report)
    print(report)


if __name__ == '__main__':
    main()
