"""Relevance testing of TCAV results (Sec. III-E): Mann-Whitney U against random concepts, Benjamini-Hochberg FDR,
and the linear-probe separability criterion."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .tcav_utils import calculate_confidence_intervals


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values). NaNs are preserved."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan)
    mask = np.isfinite(pvals)
    p = pvals[mask]
    n = p.size
    if n == 0:
        return qvals
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    qvals[mask] = out
    return qvals


def summarize_concept(result, score_type='sign_count', min_accuracy=0.6):
    """Per-layer summary of one `run_tcav.py` output: one row per (concept, layer).

    `result['scores']` maps "<concept_id>-<random_id>" to {layer: {score_type: [positive, random]}}, and
    `result['stats']` maps the same keys to {layer: {'test_acc': ...}}.
    """
    rows = []
    keys = list(result['scores'].keys())
    for layer in result['layers']:
        pos_scores = [float(result['scores'][k][layer][score_type][0]) for k in keys]
        neg_scores = [float(result['scores'][k][layer][score_type][1]) for k in keys]
        _, pval = mannwhitneyu(pos_scores, neg_scores, alternative='two-sided')
        mean_score, ci_score = calculate_confidence_intervals(pos_scores)
        median_stat = np.median([result['stats'][k][layer]['test_acc'] for k in keys], axis=0)
        rows.append({'concept': result['concept'], 'layer': layer, 'mean_score': mean_score, 'ci_score': ci_score,
                     'median_stat': median_stat, 'p_value': pval, 'accurate': median_stat > min_accuracy})
    return pd.DataFrame(rows)


def apply_fdr(tables, q=0.05):
    """BH-FDR jointly over all tests in `tables` (one DataFrame per seed), as in the paper's analysis.

    Returns copies with `p_value` replaced by the adjusted value and `significant = p_value < q`.
    """
    sizes = [len(t) for t in tables]
    qvals = bh_fdr(np.concatenate([t['p_value'].to_numpy() for t in tables]))
    out, start = [], 0
    for t, n in zip(tables, sizes):
        t = t.copy()
        t['p_value'] = qvals[start:start + n]
        t['significant'] = t['p_value'] < q
        out.append(t)
        start += n
    return out
