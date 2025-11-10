#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math
import re
from glob import glob
from typing import List, Tuple, Optional
import numpy as np

# ------------------------- parsing -------------------------

def parse_loss_file(path: str) -> Tuple[int, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) < 1:
        raise ValueError(f"{path}: file is empty")
    try:
        N = int(lines[0].strip())
    except Exception as e:
        raise ValueError(f"{path}: line 1 must be integer N, got: {lines[0]!r}") from e
    if len(lines) < 2:
        raise ValueError(f"{path}: line 2 missing; must be present and empty")
    if lines[1].strip() != "":
        raise ValueError(f"{path}: line 2 must be empty to avoid hard gaps; got: {lines[1]!r}")

    S: List[int] = []
    if len(lines) >= 3:
        raw = lines[2].strip()
        if raw:
            parts = re.split(r"[\\s,]+", raw)
            for tok in parts:
                if tok == "":
                    continue
                try:
                    idx = int(tok)
                except Exception as e:
                    raise ValueError(f"{path}: invalid lost index token: {tok!r}") from e
                S.append(idx)
    S = sorted(set(S))
    for idx in S:
        if idx < 1 or idx > N:
            raise ValueError(f"{path}: lost index {idx} out of range [1..{N}]")
    return N, S

def build_X(N: int, S: List[int]) -> np.ndarray:
    X = np.zeros(N, dtype=int)
    if S:
        X[np.array(S, dtype=int) - 1] = 1
    return X.reshape(-1, 1)


# ------------------------- run helpers -------------------------

def loss_runs_from_S(S: List[int]) -> List[int]:
    if not S:
        return []
    runs = []
    last = S[0]
    streak = 1
    for idx in S[1:]:
        if idx == last + 1:
            streak += 1
        else:
            runs.append(streak)
            streak = 1
        last = idx
    runs.append(streak)
    return runs

def success_runs_from_N_S(N: int, S: List[int]) -> List[int]:
    if not S:
        return [N] if N > 0 else []
    runs = []
    g0 = S[0] - 1
    if g0 >= 1: runs.append(g0)
    for j in range(len(S) - 1):
        gap = S[j+1] - S[j] - 1
        if gap >= 1: runs.append(gap)
    g_last = N - S[-1]
    if g_last >= 1: runs.append(g_last)
    return runs


# ------------------------- utilities -------------------------

def _run_stats(run_lengths):
    n = len(run_lengths)
    if n == 0:
        return {"n": 0, "mean": float('nan'), "std": float('nan'), "max": 0, "min": 0}
    s = sum(run_lengths)
    mean = s / n
    var = sum((x-mean)**2 for x in run_lengths) / n
    return {"n": n, "mean": mean, "std": math.sqrt(var), "max": max(run_lengths), "min": min(run_lengths)}

def _format_bins(bins):
    lines = []
    for k0, k1, obs, exp in bins:
        lab = f"k={k0}" if k0 == k1 else f"{k0}-{k1}"
        lines.append(f"  [{lab}] obs={int(obs)}, exp={exp:.3f}")
    return "\n".join(lines)

def _flatten_X(X_list):
    return [x.reshape(-1).astype(int) for x in X_list]

def _N_S_from_X_list(X_list):
    N_list = [int(x.shape[0]) for x in X_list]
    S_list = []
    for x in X_list:
        arr = x.reshape(-1).astype(int)
        S = list((arr.nonzero()[0] + 1).tolist())
        S_list.append(S)
    return N_list, S_list

def lag1_autocorr(X_list):
    xs = _flatten_X(X_list)
    T = sum(len(x) for x in xs)
    if T < 3:
        return float("nan")
    p = sum(int(x.sum()) for x in xs) / T
    num = 0.0
    den = 0.0
    for x in xs:
        xm = x - p
        if len(x) >= 2:
            num += float((xm[1:] * xm[:-1]).sum())
        den += float((xm * xm).sum())
    return num / den if den > 0 else float("nan")

def _runs_from_binary_array(arr):
    arr = arr.astype(int).reshape(-1)
    if arr.size == 0:
        return [], []
    runs1, runs0 = [], []
    cur_val = arr[0]
    cur_len = 1
    for v in arr[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            (runs1 if cur_val == 1 else runs0).append(cur_len)
            cur_val = v
            cur_len = 1
    (runs1 if cur_val == 1 else runs0).append(cur_len)
    return runs1, runs0

def _adaptive_geom_chisquare(run_lengths, theta, alpha=0.05, min_exp=5):
    """
    Chi-square GOF for Geometric(theta) on support {1,2,...}
    P(K=k) = (1-theta)^(k-1) * theta
    """
    n = int(len(run_lengths))
    if n == 0:
        return {"stat": float("nan"), "df": 0, "pvalue": float("nan"), "critical": float("nan"), "passed": True, "bins": []}

    from collections import Counter
    cnt = Counter(run_lengths)
    max_obs = max(cnt.keys())

    bins = []
    cur_k_start = 1
    cur_obs = 0
    cur_prob = 0.0
    cum_prob = 0.0

    def geom_prob(k):  # P(K=k)
        return (1 - theta) ** (k - 1) * theta

    k = 1
    while k <= max_obs:
        pk = geom_prob(k)
        cur_prob += pk
        cur_obs += cnt.get(k, 0)
        if (cur_prob * n) >= min_exp:
            bins.append((cur_k_start, k, cur_obs, cur_prob * n))
            cum_prob += cur_prob
            cur_k_start = k + 1
            cur_obs = 0
            cur_prob = 0.0
        k += 1

    tail_prob = max(0.0, 1.0 - cum_prob)
    tail_obs = sum(v for kk, v in cnt.items() if kk >= cur_k_start)
    if bins and (tail_prob * n) < min_exp:
        k0, k1, o, e = bins.pop()
        bins.append((k0, k1 + (max_obs - k1), o + tail_obs, e + tail_prob * n))
    else:
        bins.append((cur_k_start, max_obs, tail_obs, tail_prob * n))

    bins = [b for b in bins if b[3] > 1e-12]

    stat = sum((obs - exp) ** 2 / exp for _, _, obs, exp in bins)

    df_raw = len(bins) - 1 - 1  # 1 parameter estimated
    if df_raw <= 0:
        df = 0
        critical = float("nan")
        pval = float("nan")
        passed = True
    else:
        df = df_raw
        if _SCIPY_OK:
            critical = stats.chi2.ppf(1 - alpha, df)
            pval = 1 - stats.chi2.cdf(stat, df)
            passed = (stat <= critical)
        else:
            critical = float("nan")
            pval = float("nan")
            passed = False if len(bins) >= 3 else True

    return {"stat": float(stat), "df": int(df), "pvalue": float(pval), "critical": float(critical), "passed": bool(passed), "bins": bins}

# Posterior decoding / Viterbi helper
def _decode_states(model, X_list, method="viterbi"):
    X = np.vstack([x.reshape(-1,1).astype(int) for x in X_list])
    lengths = [x.shape[0] for x in X_list]
    if method == "viterbi":
        return model.predict(X, lengths=lengths), lengths
    try:
        post = model.predict_proba(X, lengths=lengths)
        return post.argmax(axis=1), lengths
    except Exception:
        return model.predict(X, lengths=lengths), lengths

# Posterior predictive check for state-run chi2 GOF
def ge_ppc_runs(ge_fit: dict, X_list, n_sims=200, use_stationary_pi=True, decode_method="viterbi", alpha=0.05):
    import random
    a = ge_fit["a"]; b = ge_fit["b"]; pG = ge_fit["pG"]; pB = ge_fit["pB"]
    if use_stationary_pi:
        piG = ge_fit["piG_stat"]; piB = ge_fit["piB_stat"]
    else:
        piG = ge_fit["piG_fit"];  piB = ge_fit["piB_fit"]
    lengths = [x.shape[0] for x in X_list]
    model = ge_fit.get("model")
    if model is None or n_sims <= 0:
        return {"ppp_G": float("nan"), "ppp_B": float("nan"), "nsims": 0}
    # observed
    states_obs, _ = _decode_states(model, X_list, method=decode_method)
    offs=0; stG_obs=[]; stB_obs=[]
    for L in lengths:
        s = states_obs[offs:offs+L]; offs+=L
        cur=s[0]; clen=1; runs=[]
        for v in s[1:]:
            if v==cur: clen+=1
            else: runs.append((cur,clen)); cur=v; clen=1
        runs.append((cur,clen))
        for lab, Lr in runs:
            (stG_obs if lab==0 else stB_obs).append(Lr)
    statG_obs = _adaptive_geom_chisquare(stG_obs, theta=max(1e-9, a), alpha=alpha)["stat"]
    statB_obs = _adaptive_geom_chisquare(stB_obs, theta=max(1e-9, b), alpha=alpha)["stat"]
    # simulate
    def sample_one(L, rng):
        s0 = 0 if rng.random() < piG else 1
        states = [s0]
        for _ in range(L-1):
            s = states[-1]
            if s==0: states.append(1 if rng.random() < a else 0)
            else:    states.append(0 if rng.random() < b else 1)
        obs = []
        for st in states:
            if st==0: obs.append(1 if rng.random() < pG else 0)
            else:     obs.append(1 if rng.random() < pB else 0)
        return np.array(obs, dtype=int).reshape(-1,1)
    rng = random.Random(0)
    cntG=0; cntB=0
    for _ in range(n_sims):
        Xs = [sample_one(L, rng) for L in lengths]
        st_sim, _ = _decode_states(model, Xs, method=decode_method)
        offs=0; stG=[]; stB=[]
        for L in lengths:
            s = st_sim[offs:offs+L]; offs+=L
            cur=s[0]; clen=1; runs=[]
            for v in s[1:]:
                if v==cur: clen+=1
                else: runs.append((cur,clen)); cur=v; clen=1
            runs.append((cur,clen))
            for lab, Lr in runs:
                (stG if lab==0 else stB).append(Lr)
        statG = _adaptive_geom_chisquare(stG, theta=max(1e-9, a), alpha=alpha)["stat"]
        statB = _adaptive_geom_chisquare(stB, theta=max(1e-9, b), alpha=alpha)["stat"]
        if statG >= statG_obs: cntG += 1
        if statB >= statB_obs: cntB += 1
    ppp_G = (cntG + 1) / (n_sims + 1)
    ppp_B = (cntB + 1) / (n_sims + 1)
    return {"ppp_G": ppp_G, "ppp_B": ppp_B, "nsims": n_sims}


# ------------------------- debug tee and k-fold helpers -------------------------

class _Tee:
    def __init__(self, filepath):
        import sys
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
    def write(self, data):
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        try:
            self._stdout.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self._file.flush()
        except Exception:
            pass
        try:
            self._stdout.flush()
        except Exception:
            pass
    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

def _kfold_indices(n: int, k: int, seed: int):
    import random
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    base, extra = divmod(n, k)
    folds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(idxs[start:start+size])
        start += size
    return folds

def verify_bernoulli_oos(X_list, p_model: float, alpha=0.05):
    """
    Out-of-sample Bernoulli verification against a fixed p_model.
    Uses the same metrics as verify_bernoulli but compares P11_hat to p_model,
    tests r1 ~ 0, and run-length GOF with theta derived from p_model.
    """
    xs = _flatten_X(X_list)
    T = sum(len(x) for x in xs)
    n1 = sum(int(x.sum()) for x in xs)
    L_hat = n1 / T if T > 0 else float("nan")

    # Empirical conditional P(1|1)
    prev1 = 0
    both11 = 0
    for x in xs:
        if len(x) >= 2:
            prev = x[:-1]
            cur  = x[1:]
            prev1 += int(prev.sum())
            both11 += int(((prev == 1) & (cur == 1)).sum())
    P11_cond_hat = both11 / prev1 if prev1 > 0 else float("nan")

    # Tolerances
    SE_P11 = math.sqrt(max(1e-30, p_model * (1 - p_model)) / max(1, prev1)) if prev1 > 0 else float("nan")
    tol_P11 = 1.96 * SE_P11
    total_pairs = max(0, T - len(xs))
    tol_r1 = 1.96 / math.sqrt(max(1, total_pairs))

    # lag-1 autocorr
    r1 = lag1_autocorr(X_list)

    # Run-length GOF under Bernoulli(p_model)
    loss_runs_all = []
    succ_runs_all = []
    for x in xs:
        r1s, r0s = _runs_from_binary_array(x)
        loss_runs_all.extend(r1s)
        succ_runs_all.extend(r0s)

    res_loss = _adaptive_geom_chisquare(loss_runs_all, theta=max(1e-9, 1 - p_model), alpha=alpha)
    res_succ = _adaptive_geom_chisquare(succ_runs_all, theta=max(1e-9, p_model), alpha=alpha)

    # Pass/Fail
    p11_pass = (abs(P11_cond_hat - p_model) <= tol_P11) if (not math.isnan(P11_cond_hat) and not math.isnan(tol_P11)) else False
    r1_pass  = (abs(r1) <= tol_r1) if (not math.isnan(r1) and not math.isnan(tol_r1)) else False
    indep_pass = bool(p11_pass and r1_pass)
    dist_pass  = bool(res_loss["passed"] and res_succ["passed"])
    accepted = bool(indep_pass and dist_pass)

    return {
        "T": T, "n1": n1,
        "L_hat": L_hat, "p_model": p_model,
        "independence": {
            "P11_cond_hat": P11_cond_hat, "prev1": prev1, "both11": both11,
            "tol_P11": tol_P11, "r1": r1, "tol_r1": tol_r1,
            "p11_pass": p11_pass, "r1_pass": r1_pass, "passed": indep_pass
        },
        "run_length": {
            "loss_geom": res_loss, "success_geom": res_succ,
            "passed": res_loss["passed"] and res_succ["passed"]
        },
        "accepted": accepted
    }

def kfold_verification(X_list, paths, k=5, seed=0, estimate_pi=False, alpha=0.05, max_iter=200, tol=1e-6, decode_method="viterbi", debug=False):
    """
    # Lazy imports to avoid circular dependencies
    try:
        from Bernoulli import fit_bernoulli_multiple
    except Exception:
        from bernoulli_only import fit_bernoulli_multiple
    try:
        from GE import ge_initial_params_from_runs_multi, fit_ge_hmm_multiple, verify_ge
    except Exception:
        from ge_only import ge_initial_params_from_runs_multi, fit_ge_hmm_multiple, verify_ge

    5-fold file-level verification for Bernoulli and GE.
    Returns summary dict and prints per-fold diagnostics.
    """
    assert len(X_list) == len(paths)
    n = len(X_list)
    folds = _kfold_indices(n, k, seed)
    results = {"folds": []}

    bern_indep_pass = []
    bern_run_pass   = []
    ge_mom_pass     = []
    ge_G_pass       = []
    ge_B_pass       = []

    for i, test_idx in enumerate(folds, start=1):
        train_idx = [j for j in range(n) if j not in test_idx]
        X_train = [X_list[j] for j in train_idx]
        X_test  = [X_list[j] for j in test_idx]

        # Bernoulli fit on train
        bern_train = fit_bernoulli_multiple(X_train)
        p_train = bern_train["p_hat"]

        # GE fit on train
        N_list, S_list = _N_S_from_X_list(X_train)
        a0,b0,pG0,pB0,piG0,piB0,_ = ge_initial_params_from_runs_multi(N_list, S_list, X_train)
        ge = fit_ge_hmm_multiple(
            X_train, a0=a0, b0=b0, pG0=pG0, pB0=pB0, piG0=piG0, piB0=piB0,
            estimate_pi=estimate_pi, max_iter=max_iter, tol=tol, random_state=seed
        )

        # Verification on test
        b_oos = verify_bernoulli_oos(X_test, p_model=p_train, alpha=alpha)
        g_oos = verify_ge(X_test, ge, alpha=alpha, use_stationary_pi=True, decode_method=decode_method)

        bern_indep_pass.append(bool(b_oos["independence"]["passed"]))
        bern_run_pass.append(bool(b_oos["run_length"]["passed"]))
        ge_mom_pass.append(bool(g_oos["moments_passed"]))
        ge_G_pass.append(bool(g_oos["state_runlength"]["good_geom"]["passed"] if "state_runlength" in g_oos else False))
        ge_B_pass.append(bool(g_oos["state_runlength"]["bad_geom"]["passed"] if "state_runlength" in g_oos else False))

        if debug:
            train_files = [paths[j] for j in train_idx]
            test_files  = [paths[j] for j in test_idx]
            print(f"\n[CV fold {i}/{k}]")
            print(f"Train files ({len(train_files)}):")
            for f in train_files: print(f"  - {f}")
            print(f"Test files ({len(test_files)}):")
            for f in test_files:  print(f"  - {f}")
            print("Bernoulli OOS: indep_pass=%s, runlen_pass=%s | p_train=%.6g, L_test=%.6g"
                  % (b_oos['independence']['passed'], b_oos['run_length']['passed'], p_train, b_oos['L_hat']))
            sr = g_oos.get("state_runlength", {})
            print("GE OOS: moments_pass=%s, G_pass=%s, B_pass=%s | a=%.3g b=%.3g pG=%.3g pB=%.3g"
                  % (g_oos.get('moments_passed', False),
                     sr.get('good_geom', {}).get('passed', False),
                     sr.get('bad_geom', {}).get('passed', False),
                     ge.get('a', float('nan')), ge.get('b', float('nan')),
                     ge.get('pG', float('nan')), ge.get('pB', float('nan'))))

        results["folds"].append({
            "train_idx": train_idx, "test_idx": test_idx,
            "bernoulli": b_oos, "GE": g_oos,
            "bernoulli_p_train": p_train, "GE_params_train": {k: ge.get(k) for k in ("a","b","pG","pB")}
        })

    # Summary
    results["summary"] = {
        "Bernoulli": {
            "independence_pass_rate": float(sum(bern_indep_pass)) / len(bern_indep_pass) if bern_indep_pass else float("nan"),
            "runlen_pass_rate": float(sum(bern_run_pass)) / len(bern_run_pass) if bern_run_pass else float("nan"),
        },
        "GE": {
            "moments_pass_rate": float(sum(ge_mom_pass)) / len(ge_mom_pass) if ge_mom_pass else float("nan"),
            "G_run_pass_rate": float(sum(ge_G_pass)) / len(ge_G_pass) if ge_G_pass else float("nan"),
            "B_run_pass_rate": float(sum(ge_B_pass)) / len(ge_B_pass) if ge_B_pass else float("nan"),
        },
        "folds": len(folds)
    }
    return results
