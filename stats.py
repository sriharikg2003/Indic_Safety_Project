import argparse
import csv
import re
import sys
import warnings
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd
from scipy import sparse

from common import (CORE_LANGS, DATA, LANGS, OUT, SOURCE_OF_PREFIX, bad_pids, is_english, load_base,
                    read_jsonl)

MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "aya-expanse-8b", "gemma-3-12b-it", "sarvam-m"]
REFUSE = {"full_refusal", "partial_refusal"}
INDIC = [lang for lang in LANGS if lang != "en"]
INDIC_CORE = [lang for lang in CORE_LANGS if lang != "en"]
CONDITIONS = ["direct", "gloss", "intent_probe", "mismatch_gloss", "self_translate"]

ST_ENGLISH_WORDS = set(
    "the a an is are was were do does did how what why who where when which can could should i you "
    "my me to of in on for with and or not it this that be have has from at by about as if your".split())


def warn(message):
    print("WARNING: " + message)
    print("WARNING: " + message, file=sys.stderr)


def harm(row):
    return int(row.get("harmful_content") is True)


def refused(row):
    return int(row["verdict"] in REFUSE)


def read_judgments(judge, model, bad):
    by = defaultdict(list)
    for condition in CONDITIONS:
        for row in read_jsonl(OUT / "judgments" / judge / model / f"{condition}.jsonl"):
            if row["pid"] in bad:
                continue
            source = SOURCE_OF_PREFIX[row["pid"][0]]
            by[(model, row["condition"], source, row["lang"])].append(row)
    return by


def load_judgments(judge, models, mt_langs):
    bad = bad_pids()
    by = {}
    for model in models:
        native = read_judgments(judge, model, bad)
        translated = {}
        if mt_langs:
            translated = read_judgments(judge + "+mt", model, bad)
        for key in list(native):
            if key[3] in mt_langs and key[1] != "intent_probe":
                if key in translated:
                    native[key] = translated[key]
                else:
                    warn(f"no {judge}+mt judgments for {key}; keeping native-judge verdicts")
        by.update(native)
    return by


def xstest_rows(by, model, condition, lang):
    rows = {}
    for row in by.get((model, condition, "xstest", lang), []):
        if row["pid"].startswith("p"):
            rows[row["pid"]] = row
    return rows


def source_rows(by, model, condition, source, lang):
    return {row["pid"]: row for row in by.get((model, condition, source, lang), [])}


def has_model(by, model):
    return any(key[0] == model for key in by)


def sign_test_p(k, n):
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def new_estimate(base, ref_langs=()):
    return {"obs": [], "base": base, "ref": set(ref_langs), "prepared": False}


def add(est, cluster, lang, cell, value):
    est["obs"].append((cluster, lang, cell, value))


def prepare(est, args):
    clusters = sorted({obs[0] for obs in est["obs"]})
    index = {cluster: i for i, cluster in enumerate(clusters)}
    n_clusters = len(clusters)
    sums = {}
    counts = {}
    for cell in {obs[2] for obs in est["obs"]}:
        idx = np.array([index[obs[0]] for obs in est["obs"] if obs[2] == cell])
        values = np.array([obs[3] for obs in est["obs"] if obs[2] == cell])
        sums[cell] = np.bincount(idx, weights=values, minlength=n_clusters)
        counts[cell] = np.bincount(idx, minlength=n_clusters).astype(float)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n_clusters, size=(args.reps, n_clusters))
    weights = np.zeros((args.reps, n_clusters))
    np.add.at(weights, (np.repeat(np.arange(args.reps), n_clusters), draws.ravel()), 1)
    est.update(C=n_clusters, S=sums, N=counts, W=weights, prepared=True)


def cell_mean(est, cell):
    if not est["prepared"]:
        return float("nan"), 0
    n = est["N"].get(cell)
    if n is None or not n.sum():
        return float("nan"), 0
    return est["S"][cell].sum() / n.sum(), int(n.sum())


def effect(est, terms, args):
    if not est["obs"]:
        return None
    if not est["prepared"]:
        prepare(est, args)
    point = 0.0
    boot = np.zeros(args.reps)
    for coef, cell in terms:
        mean, n = cell_mean(est, cell)
        if not n:
            return None
        point += coef * mean
        num = np.einsum("bc,c->b", est["W"], est["S"][cell])
        den = np.einsum("bc,c->b", est["W"], est["N"][cell])
        with np.errstate(invalid="ignore", divide="ignore"):
            boot += coef * num / den
    finite = boot[np.isfinite(boot)]
    if len(finite) < args.reps:
        warn(f"{args.reps - len(finite)} bootstrap reps had an empty cell and were dropped")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return point, lo, hi, est["C"], args.reps - len(finite)


def per_language(est, terms, args):
    cells = {cell for coef, cell in terms}
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for cluster, lang, cell, value in est["obs"]:
        acc[lang][cell][0] += value
        acc[lang][cell][1] += 1
    out = {}
    for lang in acc:
        if lang in est["ref"]:
            continue
        if acc[lang][est["base"]][1] < args.min_n or not any(acc[lang][c][1] for c in cells):
            continue
        totals = {}
        for c in cells:
            total = acc[lang][c][0] + sum(acc[r][c][0] for r in est["ref"])
            n = acc[lang][c][1] + sum(acc[r][c][1] for r in est["ref"])
            totals[c] = [total, n]
        if any(totals[c][1] < args.min_cell for c in cells):
            continue
        out[lang] = sum(coef * totals[c][0] / totals[c][1] for coef, c in terms)
    return out


def report_effect(est, name, terms, args, expect="+"):
    result = effect(est, terms, args)
    if result is None:
        print(f"  {name:46s}    (no data)")
        return
    point, lo, hi, n_clusters, dropped = result
    by_lang = per_language(est, terms, args)
    pos = sum(v > 0 for v in by_lang.values())
    neg = sum(v < 0 for v in by_lang.values())
    ties = len(by_lang) - pos - neg
    if pos + neg:
        p = sign_test_p(pos, pos + neg)
    else:
        p = float("nan")
    if lo > 0 or hi < 0:
        star = "*"
    else:
        star = " "
    if expect != "+":
        name = name + " [?]"
    print(f"  {name:46s} {point:+7.3f} [{lo:+.3f}, {hi:+.3f}]{star} {n_clusters:7d}   "
          f"{pos:4d}/{pos + neg:<4d} {ties:4d} {p:7.2g}")


def print_cells(est, names):
    parts = []
    for cell in names:
        mean, n = cell_mean(est, cell)
        if n:
            parts.append(f"{cell}={mean:.3f} (n={n})")
        else:
            parts.append(f"{cell}=- (n=0)")
    print("    cell means: " + ";  ".join(parts))


def probe_english(model):
    path = OUT / "generations" / model / "intent_probe.jsonl"
    if not path.exists():
        warn(f"{path} missing: English-probe-only rows will be empty for {model}")
        return {}
    result = {}
    for row in read_jsonl(path):
        result[(row["pid"], row["lang"])] = is_english(row.get("response"), row.get("text", ""))
    return result


def clustered_gap(by, model, english, direct, args):
    est = new_estimate("indic", ref_langs=("en",))
    for pid, row in english.items():
        if row["label"] == "unsafe":
            add(est, pid, "en", "en", harm(row))
    for lang in INDIC:
        for pid, row in direct[lang].items():
            if row["label"] == "unsafe" and pid in english:
                add(est, pid, lang, "indic", harm(row))
    print("\n 1. Language gap (XSTest unsafe, direct)")
    report_effect(est, "harm Indic - harm English", [(1, "indic"), (-1, "en")], args)
    print_cells(est, ["en", "indic"])


def clustered_gloss(direct, gloss, probe, args):
    harm_est = new_estimate("d")
    harm_same = new_estimate("d")
    refusal_est = new_estimate("d")
    for lang in INDIC:
        for pid, row in direct[lang].items():
            if pid not in gloss[lang]:
                continue
            gloss_row = gloss[lang][pid]
            if row["label"] == "unsafe":
                add(harm_est, pid, lang, "d", harm(row))
                add(harm_est, pid, lang, "g", harm(gloss_row))
                if pid in probe[lang] and probe[lang][pid]["verdict"] == "same":
                    add(harm_same, pid, lang, "d", harm(row))
                    add(harm_same, pid, lang, "g", harm(gloss_row))
            elif row["label"] == "safe":
                add(refusal_est, pid, lang, "d", refused(row))
                add(refusal_est, pid, lang, "g", refused(gloss_row))
    print("\n 2. Gloss (Indic pooled, paired pid x lang)")
    report_effect(harm_est, "harm direct - harm gloss", [(1, "d"), (-1, "g")], args)
    print_cells(harm_est, ["d", "g"])
    report_effect(refusal_est, "over-refusal gloss - direct", [(1, "g"), (-1, "d")], args, expect="?")
    print_cells(refusal_est, ["d", "g"])
    return harm_same


def clustered_danger_zone(direct, probe, english_probe, args):
    print("\n 3. Danger zone (Indic direct, unsafe, by intent-probe verdict of the same pid x lang)")
    for only_english in (False, True):
        est = new_estimate("_all")
        dropped = 0
        for lang in INDIC:
            for pid, row in direct[lang].items():
                if row["label"] != "unsafe" or pid not in probe[lang]:
                    continue
                if only_english and not english_probe.get((pid, lang), False):
                    dropped += 1
                    continue
                add(est, pid, lang, "_all", 0.0)
                add(est, pid, lang, probe[lang][pid]["verdict"], harm(row))
        if only_english:
            tag = "English probes only"
        else:
            tag = "all probes"
        report_effect(est, f"partial - same ({tag})", [(1, "partial"), (-1, "same")], args)
        report_effect(est, f"partial - different ({tag})", [(1, "partial"), (-1, "different")], args)
        print_cells(est, ["same", "partial", "different", "refused"])
        if only_english:
            print(f"    dropped {dropped} non-English probe replies")


def clustered_understood(english, direct, probe, args):
    est = new_estimate("i")
    for lang in INDIC:
        for pid, row in direct[lang].items():
            if row["label"] != "unsafe" or pid not in english or pid not in probe[lang]:
                continue
            if probe[lang][pid]["verdict"] == "same":
                add(est, pid, lang, "i", harm(row))
                add(est, pid, lang, "e", harm(english[pid]))
    print("\n 4. Understood but harmful (probe == same, paired with English on the same pid)")
    report_effect(est, "harm Indic | same - harm English", [(1, "i"), (-1, "e")], args)
    print_cells(est, ["i", "e"])


def clustered_mismatch(direct, gloss, mismatch, args):
    refusal_est = new_estimate("d")
    harm_est = new_estimate("d")
    gloss_est = new_estimate("x")
    for lang in INDIC_CORE:
        for pid, x in mismatch[lang].items():
            if pid not in direct[lang]:
                continue
            d = direct[lang][pid]
            if x["label"] == "safe":
                add(refusal_est, pid, lang, "d", refused(d))
                add(refusal_est, pid, lang, "x", refused(x))
            elif x["label"] == "unsafe":
                add(harm_est, pid, lang, "d", harm(d))
                add(harm_est, pid, lang, "x", harm(x))
                if pid in gloss[lang]:
                    add(gloss_est, pid, lang, "x", harm(x))
                    add(gloss_est, pid, lang, "g", harm(gloss[lang][pid]))
    print("\n 6. Mismatched gloss (Indic core forms, paired with direct)")
    report_effect(refusal_est, "over-refusal mismatch - direct (safe text)", [(1, "x"), (-1, "d")], args)
    print_cells(refusal_est, ["d", "x"])
    report_effect(harm_est, "harm direct - mismatch (unsafe text)", [(1, "d"), (-1, "x")], args, expect="?")
    print_cells(harm_est, ["d", "x"])
    report_effect(gloss_est, "harm mismatch - gloss (unsafe text)", [(1, "x"), (-1, "g")], args, expect="?")
    print_cells(gloss_est, ["x", "g"])


def clustered_hidden_intent(by, model, base, args):
    est = new_estimate("id", ref_langs=("en",))
    by_probe = new_estimate("_all")
    for lang in LANGS:
        disguised = source_rows(by, model, "direct", "wjb_adversarial", lang)
        plain = source_rows(by, model, "direct", "wjb_vanilla", lang)
        probes = source_rows(by, model, "intent_probe", "wjb_adversarial", lang)
        for pid, row in disguised.items():
            focus = base.get(pid, {}).get("focus")
            if row["label"] != "unsafe" or focus not in plain:
                continue
            if lang == "en":
                group = "e"
            else:
                group = "i"
            add(est, focus, lang, group + "d", harm(row))
            add(est, focus, lang, group + "p", harm(plain[focus]))
            if lang != "en" and pid in probes:
                band = probes[pid]["verdict"]
                add(by_probe, focus, lang, "_all", 0.0)
                if band in ("same", "different"):
                    add(by_probe, focus, lang, band[0] + "d", harm(row))
                    add(by_probe, focus, lang, band[0] + "p", harm(plain[focus]))
    print("\n 7. Hidden intent, WildJailbreak (unsafe, disguised vs plain, clusters = request)")
    report_effect(est, "disguise effect English", [(1, "ed"), (-1, "ep")], args, expect="?")
    report_effect(est, "disguise effect Indic", [(1, "id"), (-1, "ip")], args, expect="?")
    report_effect(est, "Indic - English (diff-in-diff)", [(1, "id"), (-1, "ip"), (-1, "ed"), (1, "ep")], args,
                  expect="?")
    print_cells(est, ["ep", "ed", "ip", "id"])
    report_effect(by_probe, "disguise effect | probe same", [(1, "sd"), (-1, "sp")], args, expect="?")
    report_effect(by_probe, "disguise effect | probe different", [(1, "dd"), (-1, "dp")], args, expect="?")
    report_effect(by_probe, "same - different", [(1, "sd"), (-1, "sp"), (-1, "dd"), (1, "dp")], args, expect="?")
    print_cells(by_probe, ["sp", "sd", "dp", "dd"])
    print("    (English row of 'langs >0' n/a: English is the reference; the DiD per language uses that"
          " language's effect minus English's)")


def cmd_clustered(args):
    by = load_judgments(args.judge, args.models, args.mt_langs)
    base = load_base()
    header = (f"  {'estimate':46s} {'point':>7s} {'95% CI':>17s} {'prompts':>7s}   {'langs >0':>9s} "
              f"{'ties':>4s} {'sign p':>7s}")
    print(f"Cluster bootstrap over prompts: {args.reps} reps, seed {args.seed}, 95% percentile CI; judge {args.judge}"
          f" (+mt for {' '.join(args.mt_langs)})")
    print(f"'*' = CI excludes 0.  'langs >0' = languages (>= {args.min_n} paired rows, >= {args.min_cell} per cell)"
          " where the effect is > 0, of those with a nonzero effect;\n'sign p' = two-sided binomial sign test."
          "  [?] = no hypothesised direction (count of > 0 still shown).")
    for model in args.models:
        if not has_model(by, model):
            warn(f"no judgments for model {model}; skipped")
            continue
        print(f"\n{'=' * 110}\n=== {model}\n{header}")
        english = xstest_rows(by, model, "direct", "en")
        english_probe = probe_english(model)
        direct = {lang: xstest_rows(by, model, "direct", lang) for lang in LANGS}
        gloss = {lang: xstest_rows(by, model, "gloss", lang) for lang in LANGS}
        probe = {lang: xstest_rows(by, model, "intent_probe", lang) for lang in LANGS}
        mismatch = {lang: xstest_rows(by, model, "mismatch_gloss", lang) for lang in LANGS}
        if not english:
            warn(f"{model}: no English direct XSTest rows")
        clustered_gap(by, model, english, direct, args)
        harm_same = clustered_gloss(direct, gloss, probe, args)
        clustered_danger_zone(direct, probe, english_probe, args)
        clustered_understood(english, direct, probe, args)
        print("\n 5. Gloss among probe == same prompts")
        report_effect(harm_same, "harm direct - harm gloss | same", [(1, "d"), (-1, "g")], args)
        print_cells(harm_same, ["d", "g"])
        clustered_mismatch(direct, gloss, mismatch, args)
        clustered_hidden_intent(by, model, base, args)
        sys.stdout.flush()
        print(f"done {model}", file=sys.stderr)
    print("\nCell keys: d=direct g=gloss x=mismatch_gloss i=Indic e=English; WJB: ep/ed = English plain/disguised,"
          "\nip/id = Indic plain/disguised, sp/sd and dp/dd = plain/disguised among probe same / different.")


def translation_status(response):
    match = re.search(r"translation\**\s*[:：]\**\s*(.+?)(?:\n\s*\n|\n\s*\**response\**\s*[:：]|$)",
                      response, re.I | re.S)
    if not match:
        return "no_tr"
    text = match.group(1).strip()
    letters = [c for c in text if c.isalpha()]
    latin = sum(c.isascii() for c in letters) / max(1, len(letters))
    words = re.findall(r"[A-Za-z']+", text.lower())
    function_share = sum(w in ST_ENGLISH_WORDS for w in words) / max(1, len(words))
    if latin >= 0.8 and function_share >= 0.10:
        return "ok"
    return "not_en"


def fit_glmm(df, fixed, groups, method):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in fixed])
    blocks = []
    ident = []
    for j, group in enumerate(groups):
        codes, uniques = pd.factorize(df[group])
        present = codes >= 0
        rows = np.arange(len(df))[present]
        blocks.append(sparse.csr_array((np.ones(present.sum()), (rows, codes[present])),
                                       shape=(len(df), len(uniques))))
        ident += [j] * len(uniques)
    Z = sparse.hstack(blocks, format="csr")
    model = BinomialBayesMixedGLM(df["y"].to_numpy(float), X, Z, np.array(ident),
                                  fep_names=["Intercept"] + fixed, vcp_names=groups)
    options = {"maxiter": 20000}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "vb":
            result = model.fit_vb(fit_method="L-BFGS-B", minim_opts=options, rng=np.random.default_rng(0))
        else:
            result = model.fit_map(method="L-BFGS-B", minim_opts=options, rng=np.random.default_rng(0))
    optimizer = getattr(result, "optim_res", None)
    if not optimizer:
        optimizer = getattr(result, "optim_retvals", None)
    if optimizer is not None:
        ok = bool(optimizer.success)
    else:
        ok = not any("converge" in str(w.message) for w in caught)
    message = ""
    if not ok:
        message = str(getattr(optimizer, "message", "")) or "; ".join(str(w.message) for w in caught)
        message = message[:120]
    cov = np.asarray(result.cov_params())
    if cov.ndim == 2:
        variances = np.diag(cov)
    else:
        variances = cov
    if np.any(variances[:len(fixed) + 1] <= 0):
        ok = False
        message = (message + " non-positive variance in inverse Hessian").strip()
    sd = np.sqrt(np.abs(variances))
    re_sd = {}
    for j, group in enumerate(groups):
        re_sd[group] = float(np.exp(result.params[len(fixed) + 1 + j]))
    if any(v < 1e-3 or v > 10 for v in re_sd.values()):
        ok = False
        sds = " ".join(f"{g}={v:.2g}" for g, v in re_sd.items())
        message = (message + " degenerate RE SD " + sds).strip()
    odds = {}
    with np.errstate(over="ignore"):
        for i, name in enumerate(fixed, start=1):
            b = result.params[i]
            odds[name] = (np.exp(b), np.exp(b - 1.96 * sd[i]), np.exp(b + 1.96 * sd[i]))
    return odds, ok, message, re_sd


def fit_gee(df, fixed, group="pid"):
    import statsmodels.api as sm

    X = sm.add_constant(df[fixed].astype(float), has_constant="add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = sm.GEE(df["y"].astype(float), X, groups=pd.factorize(df[group])[0],
                            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
        except Exception as e:
            return None, False, f"GEE failed: {type(e).__name__}: {str(e)[:100]}"
        ci = result.conf_int()
    messages = [str(w.message)[:100] for w in caught if issubclass(w.category, Warning)]
    converged = getattr(result, "converged", True) and not any("converge" in m.lower() for m in messages)
    odds = {}
    with np.errstate(over="ignore"):
        for c in fixed:
            odds[c] = (np.exp(result.params[c]), np.exp(ci.loc[c, 0]), np.exp(ci.loc[c, 1]))
    return odds, converged, "; ".join(sorted(set(messages)))


def separation_problems(df, fixed):
    problems = []
    if fixed:
        reference = df[(df[fixed] == 0).all(axis=1)]
    else:
        reference = df
    if len(reference) and reference["y"].sum() in (0, len(reference)):
        problems.append(f"reference cell {int(reference['y'].sum())}/{len(reference)}")
    for c in fixed:
        subset = df[df[c] == 1]
        if len(subset) and subset["y"].sum() in (0, len(subset)):
            problems.append(f"{c}=1 cell {int(subset['y'].sum())}/{len(subset)}")
    return problems


def fmt_value(v):
    if np.isfinite(v) and 0.005 <= v < 1000:
        return f"{v:6.2f}"
    if not np.isfinite(v) and v != v:
        return "   nan"
    if v >= 1000:
        return "  >999"
    return " <0.01"


def fmt_odds(t, bad=False):
    if t is None:
        return f"{'-':>24s}"
    odds, lo, hi = t
    if bad:
        star = "x"
    elif lo > 1 or hi < 1:
        star = "*"
    else:
        star = " "
    return f"{fmt_value(odds)} [{fmt_value(lo)},{fmt_value(hi)}]{star}"


def run_models(df, fixed, groups, label, args):
    if df.empty or df["y"].nunique() < 2:
        warn(f"{label}: no data or no variation in the outcome; model skipped")
        return None
    separated = separation_problems(df, fixed)
    fits = {}
    notes = []
    for method in ("laplace", "vb"):
        try:
            odds, ok, message, re_sd = fit_glmm(df, fixed, groups, method)
        except Exception as e:
            warn(f"{label}: BinomialBayesMixedGLM {method} failed ({type(e).__name__}: {str(e)[:100]})")
            continue
        fits[method] = (odds, ok)
        notes.append(f"{method} RE SD " + " ".join(f"{g}={v:.2f}" for g, v in re_sd.items()))
        if not ok:
            warn(f"{label}: GLMM {method} fit did NOT converge / degenerate ({message}) -> marked 'x', do not use")
    if args.no_gee:
        gee, gee_ok, gee_message = None, True, ""
    else:
        gee, gee_ok, gee_message = fit_gee(df, fixed)
        if gee is None:
            warn(f"{label}: {gee_message}")
        elif not gee_ok:
            warn(f"{label}: GEE did NOT converge ({gee_message})")
    for c in fixed:
        line = f"    {label + ' ' + c:30s}"
        for method in ("laplace", "vb"):
            odds, ok = fits.get(method, (None, False))
            if odds:
                line += f" {fmt_odds(odds[c], bad=not ok)}"
            else:
                line += f" {fmt_odds(None)}"
        if gee:
            line += f" {fmt_odds(gee[c], bad=(not gee_ok or bool(separated)))}"
        else:
            line += f" {fmt_odds(None)}"
        print(line)
    print(f"      n={len(df)}, prompts={df['pid'].nunique()}, langs={df['lang'].nunique()}; " + "; ".join(notes))
    if separated:
        warn(f"{label}: quasi-separation ({', '.join(separated)}): GEE OR infinite (marked 'x'); GLMM OR held finite"
             " only by its N(0, 2^2) prior on the log-OR")
    return fits


def boot_diff(pids, values, rng, reps):
    ids = sorted(set(pids))
    index = {pid: i for i, pid in enumerate(ids)}
    sums = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    for pid, value in zip(pids, values):
        sums[index[pid]] += value
        counts[index[pid]] += 1
    draws = rng.integers(0, len(ids), size=(reps, len(ids)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return sums.sum() / counts.sum(), np.percentile(estimates, 2.5), np.percentile(estimates, 97.5)


def gap_frame(by, model, drop=frozenset()):
    english = xstest_rows(by, model, "direct", "en")
    rows = []
    for pid, row in english.items():
        if row["label"] == "unsafe":
            rows.append({"pid": pid, "lang": "en", "lang_re": None, "indic": 0, "y": harm(row)})
    for lang in INDIC:
        for pid, row in xstest_rows(by, model, "direct", lang).items():
            if row["label"] == "unsafe" and pid in english and (pid, lang) not in drop:
                rows.append({"pid": pid, "lang": lang, "lang_re": lang, "indic": 1, "y": harm(row)})
    return pd.DataFrame(rows)


def gloss_frame(by, model, drop=frozenset(), label="unsafe", outcome=harm):
    rows = []
    for lang in INDIC:
        direct = xstest_rows(by, model, "direct", lang)
        gloss = xstest_rows(by, model, "gloss", lang)
        for pid, row in direct.items():
            if pid in gloss and row["label"] == label and (pid, lang) not in drop:
                item = f"{pid}|{lang}"
                rows.append({"pid": pid, "lang": lang, "item": item, "gloss": 0, "y": outcome(row)})
                rows.append({"pid": pid, "lang": lang, "item": item, "gloss": 1, "y": outcome(gloss[pid])})
    return pd.DataFrame(rows)


def zone_frame(by, model):
    rows = []
    for lang in INDIC:
        probe = xstest_rows(by, model, "intent_probe", lang)
        for pid, row in xstest_rows(by, model, "direct", lang).items():
            if row["label"] != "unsafe" or pid not in probe:
                continue
            verdict = probe[pid]["verdict"]
            if verdict not in ("same", "partial", "different"):
                continue
            rows.append({"pid": pid, "lang": lang, "probe": verdict, "partial": int(verdict == "partial"),
                         "different": int(verdict == "different"), "same": int(verdict == "same"),
                         "y": harm(row)})
    return pd.DataFrame(rows)


def rate(df, **conditions):
    subset = df
    for column, value in conditions.items():
        subset = subset[subset[column] == value]
    if len(subset):
        return subset["y"].mean(), len(subset)
    return float("nan"), len(subset)


def robustness_r1(by, models, columns, args):
    print(f"\n{'=' * 100}\nR1  Mixed-effects logistic regression (replaces pooled z / McNemar)\n{'=' * 100}")
    print("(a) gap   harm ~ indic + (1|prompt) + (1|Indic language); Indic direct, 28 forms, vs English direct."
          "\n          English is the fixed reference (a random intercept for English alone is not identified"
          "\n          against 'indic'); OR = typical Indic form vs English.")
    print("(b) gloss harm ~ gloss + (1|prompt) + (1|language); paired pid x lang, Indic forms")
    print("    + over-refusal ~ gloss on safe prompts (same random effects)")
    print("(c) zone  harm ~ probe + (1|prompt) + (1|language); Indic direct; reference 'same' (and 'different')")
    for model in models:
        print(f"\n=== {model}\n{columns}")
        gap = gap_frame(by, model)
        if gap.empty:
            warn(f"{model}: no direct XSTest rows")
            continue
        english_rate, english_n = rate(gap, indic=0)
        indic_rate, indic_n = rate(gap, indic=1)
        print(f"  (a) pooled harm English {english_rate:.3f} (n={english_n})  "
              f"Indic {indic_rate:.3f} (n={indic_n})  diff {indic_rate - english_rate:+.3f}")
        run_models(gap, ["indic"], ["pid", "lang_re"], "gap", args)
        gloss = gloss_frame(by, model)
        if gloss.empty:
            warn(f"{model}: no paired direct/gloss rows")
        else:
            print(f"  (b) pooled harm direct {rate(gloss, gloss=0)[0]:.3f}  gloss {rate(gloss, gloss=1)[0]:.3f}  "
                  f"(pairs={rate(gloss, gloss=0)[1]})   OR < 1 = gloss lowers harm")
            run_models(gloss, ["gloss"], ["pid", "lang"], "gloss harm", args)
            over = gloss_frame(by, model, label="safe", outcome=refused)
            print(f"      pooled over-refusal direct {rate(over, gloss=0)[0]:.3f}  gloss {rate(over, gloss=1)[0]:.3f}"
                  f"  (pairs={rate(over, gloss=0)[1]})")
            run_models(over, ["gloss"], ["pid", "lang"], "gloss overref", args)
        zone = zone_frame(by, model)
        if zone.empty:
            warn(f"{model}: no intent_probe rows")
        else:
            parts = []
            for verdict in ("same", "partial", "different"):
                value, n = rate(zone, probe=verdict)
                parts.append(f"{verdict} {value:.3f} (n={n})")
            print("  (c) pooled harm by probe: " + "  ".join(parts))
            run_models(zone, ["partial", "different"], ["pid", "lang"], "zone vs same:", args)
            run_models(zone, ["partial", "same"], ["pid", "lang"], "zone vs different:", args)
        sys.stdout.flush()
        print(f"R1 done {model}", file=sys.stderr)


def self_translate_items(by, model):
    path = OUT / "generations" / model / "self_translate.jsonl"
    status = {}
    for row in read_jsonl(path):
        status[(row["pid"], row["lang"])] = translation_status(row.get("response") or "")
    items = []
    for lang in INDIC_CORE:
        direct = xstest_rows(by, model, "direct", lang)
        translated = xstest_rows(by, model, "self_translate", lang)
        if not translated:
            warn(f"{model}: no judged self_translate rows for {lang}")
        for pid, st_row in translated.items():
            if pid not in direct:
                continue
            if (pid, lang) not in status:
                warn(f"{model}: judged self_translate {pid} {lang} has no generation row")
            items.append((pid, lang, direct[pid]["label"], direct[pid], st_row, status.get((pid, lang)) == "ok"))
    return items


def robustness_r2(by, models, rng, args):
    print(f"\n{'=' * 100}\nR2  Self-translate: all items vs items whose reply really contains an English translation"
          f"\n{'=' * 100}")
    print("13 Indic core forms, paired with direct on the same pid x lang.  'translated' = st_check.classify()"
          "\n== 'ok' (a 'Translation:' line that is English).  Caveat: the subset is selected on the self_translate"
          "\nreply itself (post-treatment), so (ii) is descriptive, not a causal effect of translating.")
    print(f"  {'model':22s} {'subset':12s} {'n_uns':>5s} {'harm d -> st':>14s} {'diff [95% CI]':>24s} {'McNemar p':>9s}"
          f"   {'n_safe':>6s} {'overref d -> st':>15s} {'diff [95% CI]':>24s}")
    for model in models:
        path = OUT / "generations" / model / "self_translate.jsonl"
        if not path.exists():
            warn(f"{path} missing; R2 skipped for {model}")
            continue
        items = self_translate_items(by, model)
        for subset_name in ("(i) all", "(ii) transl.", "    not tr."):
            if subset_name == "(i) all":
                subset = items
            elif subset_name == "(ii) transl.":
                subset = [x for x in items if x[5]]
            else:
                subset = [x for x in items if not x[5]]
            unsafe = [x for x in subset if x[2] == "unsafe"]
            safe = [x for x in subset if x[2] == "safe"]
            line = f"  {model[:22]:22s} {subset_name:12s} {len(unsafe):5d}"
            if unsafe:
                harm_direct = np.mean([harm(x[3]) for x in unsafe])
                harm_st = np.mean([harm(x[4]) for x in unsafe])
                diff, lo, hi = boot_diff([x[0] for x in unsafe], [harm(x[4]) - harm(x[3]) for x in unsafe],
                                         rng, args.reps)
                b = sum(harm(x[3]) and not harm(x[4]) for x in unsafe)
                c = sum(harm(x[4]) and not harm(x[3]) for x in unsafe)
                line += f"   {harm_direct:.3f} -> {harm_st:.3f} {diff:+.3f} [{lo:+.3f},{hi:+.3f}] {mcnemar_p(b, c):9.2g}"
            else:
                line += f"   {'-':>14s} {'-':>24s} {'-':>9s}"
            line += f"   {len(safe):6d}"
            if safe:
                refusal_direct = np.mean([refused(x[3]) for x in safe])
                refusal_st = np.mean([refused(x[4]) for x in safe])
                diff, lo, hi = boot_diff([x[0] for x in safe], [refused(x[4]) - refused(x[3]) for x in safe],
                                         rng, args.reps)
                line += f"    {refusal_direct:.3f} -> {refusal_st:.3f} {diff:+.3f} [{lo:+.3f},{hi:+.3f}]"
            print(line)
        n_ok = sum(x[5] for x in items)
        print(f"  {'':22s} share of replies with an English translation line: {n_ok / max(1, len(items)):.2f}"
              f" ({n_ok}/{len(items)})")


def robustness_r3(by, models, columns, args):
    print(f"\n{'=' * 100}\nR3  Excluding translations flagged by the meaning check\n{'=' * 100}")
    path = DATA / "translations" / "flagged.csv"
    if not path.exists():
        warn(f"{path} missing: no per-item translation QC flags -> R3 skipped")
        return
    with open(path, encoding="utf-8") as f:
        flagged = list(csv.DictReader(f))
    drop_any = frozenset((r["pid"], r["lang"]) for r in flagged)
    drop_both = frozenset((r["pid"], r["lang"]) for r in flagged if r["votes"].count("different") >= 2)
    current = {}
    for lang in INDIC:
        for row in read_jsonl(DATA / "translations" / f"{lang}.jsonl"):
            current[(row["pid"], lang)] = row["text"]
    stale = sum(current.get((r["pid"], r["lang"])) not in (None, r["text"]) for r in flagged)
    if stale:
        warn(f"{stale} flagged rows no longer match the current translation text (flag may be stale)")
    print(f"source: {path.relative_to(DATA.parent)} ({len(flagged)} flagged pid x lang, all XSTest; flagged = majority of"
          f"\nback-translation meaning checks said 'different'; {len(drop_both)} flagged by 2/2 checks; stale flags: {stale})."
          "\nCaveat: the meaning check back-translates with MT, so romanised / code-mixed forms and very low-resource"
          "\nscripts (kn_latn, mni, sat, tcy) are flagged most, partly because the back-translator fails, not the prompt.")
    per_form = defaultdict(int)
    for r in flagged:
        per_form[r["lang"]] += 1
    print("flagged per form (of 450 XSTest prompts): "
          + " ".join(f"{lang}={per_form[lang]}" for lang in INDIC if lang in per_form))
    print(f"\n  {'model':22s} {'set':18s} {'Indic n':>7s} {'harm en':>7s} {'harm Indic':>10s} {'gap':>7s}"
          f"   {'pairs':>5s} {'gloss d -> g':>14s} {'reduction':>9s}")
    for model in models:
        for tag, drop in (("all items", frozenset()), ("excl. flagged", drop_any), ("excl. 2/2 flagged", drop_both)):
            gap = gap_frame(by, model, drop)
            gloss = gloss_frame(by, model, drop)
            if gap.empty:
                continue
            english = rate(gap, indic=0)
            indic = rate(gap, indic=1)
            direct = rate(gloss, gloss=0)
            glossed = rate(gloss, gloss=1)
            print(f"  {model[:22]:22s} {tag:18s} {indic[1]:7d} {english[0]:7.3f} {indic[0]:10.3f}"
                  f" {indic[0] - english[0]:+7.3f}   {direct[1]:5d}  {direct[0]:.3f} -> {glossed[0]:.3f}"
                  f" {direct[0] - glossed[0]:+9.3f}")
    print("\n  mixed-model ORs excluding all flagged items:\n" + columns)
    for model in models:
        print(f"  --- {model}")
        run_models(gap_frame(by, model, drop_any), ["indic"], ["pid", "lang_re"], "gap (excl. flagged)", args)
        run_models(gloss_frame(by, model, drop_any), ["gloss"], ["pid", "lang"], "gloss harm (excl. flagged)", args)
        sys.stdout.flush()
        print(f"R3 done {model}", file=sys.stderr)


def cmd_robustness(args):
    by = load_judgments(args.judge, args.models, args.mt_langs)
    models = [model for model in args.models if has_model(by, model)]
    for model in args.models:
        if model not in models:
            warn(f"no judgments for model {model}; skipped")
    print(f"Robustness checks; judge {args.judge} (+mt for {' '.join(args.mt_langs)}); XSTest only.")
    print("OR = odds ratio [95% interval]; '*' = excludes 1; 'x' = fit failed / degenerate / separated: do not use."
          "\nColumns: GLMM Laplace (primary; conditional OR, crossed random intercepts prompt + language) | GLMM VB"
          "\n(variational; too narrow for the gap) | GEE (logistic, exchangeable, clusters = prompt; marginal OR)."
          "\nRE SD = random-intercept SD on the logit scale.")
    columns = f"    {'':30s} {'GLMM Laplace OR':>24s} {'GLMM VB OR':>24s} {'GEE OR':>24s}"
    rng = np.random.default_rng(0)
    robustness_r1(by, models, columns, args)
    robustness_r2(by, models, rng, args)
    robustness_r3(by, models, columns, args)
    print("\nHow to read R1: trust a GLMM Laplace interval only when it is not marked 'x'; VB gap intervals are too"
          "\nnarrow; GEE is the model-free check (marked 'x' when English has 0 harmful replies -> OR infinite).")
    print("\nReading: R1 OR > 1 for indic = Indic prompts get more harmful replies than English after accounting"
          "\nfor prompt and language clustering; gloss OR < 1 = the English gloss lowers harm.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["clustered", "robustness"])
    parser.add_argument("--judge", default="Qwen3-32B-FP8")
    parser.add_argument("--mt-langs", nargs="*", default=["sat"])
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-cell", type=int, default=5)
    parser.add_argument("--no-gee", action="store_true")
    args = parser.parse_args()
    if args.mode == "clustered":
        cmd_clustered(args)
    else:
        cmd_robustness(args)


if __name__ == "__main__":
    main()
