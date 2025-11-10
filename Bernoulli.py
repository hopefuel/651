#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math
from typing import List, Tuple
import numpy as np
from glob import glob
from aux_loss_models import (_flatten_X, _N_S_from_X_list, _runs_from_binary_array, lag1_autocorr,
    _adaptive_geom_chisquare, _run_stats, loss_runs_from_S, success_runs_from_N_S, _Tee, _kfold_indices,
    parse_loss_file, build_X)


def fit_bernoulli_multiple(X_list: List[np.ndarray]) -> dict:
    N_total = int(sum(x.shape[0] for x in X_list))
    n1_total = int(sum(int(x.sum()) for x in X_list))
    p_hat = n1_total / N_total if N_total > 0 else float("nan")
    return {"N_total": N_total, "n1_total": n1_total, "p_hat": p_hat}

# ------------------------- GE fit (hmmlearn friendly) -------------------------

def ge_initial_params_from_runs_multi(N_list: List[int], S_list: List[List[int]], X_list: List[np.ndarray]):
    N_total = sum(N_list)
    n1_total = sum(int(x.sum()) for x in X_list)
    p_hat = n1_total / N_total if N_total > 0 else 0.0

    loss_runs_all, succ_runs_all = [], []
    for N, S in zip(N_list, S_list):
        loss_runs_all.extend(loss_runs_from_S(S))
        succ_runs_all.extend(success_runs_from_N_S(N, S))

    def safe_mean(xs, dflt=2.0):
        return (sum(xs) / max(1, len(xs))) if xs else dflt

    Lbar_loss = max(1.0001, safe_mean(loss_runs_all, 2.0))
    Lbar_succ = max(1.0001, safe_mean(succ_runs_all, 2.0))

    def clamp01(v):
        return max(1e-6, min(1 - 1e-6, float(v)))

    a0 = clamp01(1.0 / max(Lbar_succ, 2.0))
    b0 = clamp01(1.0 / max(Lbar_loss, 2.0))

    pG0 = min(0.05, 0.25 * p_hat)
    pB0 = max(p_hat + 0.02, 0.75 * p_hat)
    pB0 = min(0.95, pB0)
    if pB0 <= pG0:
        pB0 = min(0.95, pG0 + 0.05)

    piG0 = b0 / (a0 + b0)
    piB0 = a0 / (a0 + b0)
    return a0, b0, pG0, pB0, piG0, piB0, p_hat



def verify_bernoulli(X_list, alpha=0.05):
    xs = _flatten_X(X_list)
    T = sum(len(x) for x in xs)
    n1 = sum(int(x.sum()) for x in xs)
    p_hat = n1 / T if T > 0 else float("nan")

    # Independence checks
    prev1 = 0
    both11 = 0
    for x in xs:
        if len(x) >= 2:
            prev = x[:-1]
            cur  = x[1:]
            prev1 += int(prev.sum())
            both11 += int(((prev == 1) & (cur == 1)).sum())
    P11_hat = both11 / prev1 if prev1 > 0 else float("nan")
    P1_cond1_hat = P11_hat

    SE_L = math.sqrt(p_hat * (1 - p_hat) / T) if 0 < p_hat < 1 and T > 0 else float("nan")
    SE_P11 = math.sqrt(p_hat * (1 - p_hat) / prev1) if 0 < p_hat < 1 and prev1 > 0 else float("nan")
    tol_P11 = 2.0 * math.sqrt((SE_L if not math.isnan(SE_L) else 0.0) ** 2 +
                              (SE_P11 if not math.isnan(SE_P11) else 0.0) ** 2)

    r1 = lag1_autocorr(X_list)
    tol_r1 = 2.0 / math.sqrt(max(1, (T - len(xs))))

    p11_pass = (not math.isnan(P11_hat)) and (not math.isnan(p_hat)) and (abs(P11_hat - p_hat) <= (tol_P11 if not math.isnan(tol_P11) else float("inf")))
    r1_pass  = (not math.isnan(r1)) and (abs(r1) <= tol_r1)
    indep_pass = bool(p11_pass and r1_pass)

    loss_runs_all, succ_runs_all = [], []
    for x in xs:
        r1s, r0s = _runs_from_binary_array(x)
        loss_runs_all.extend(r1s)
        succ_runs_all.extend(r0s)

    res_loss = _adaptive_geom_chisquare(loss_runs_all, theta=max(1e-9, 1 - p_hat), alpha=alpha)
    res_succ = _adaptive_geom_chisquare(succ_runs_all, theta=max(1e-9, p_hat), alpha=alpha)
    loss_geom_pass = res_loss["passed"]
    succ_geom_pass = res_succ["passed"]
    dist_pass = bool(loss_geom_pass and succ_geom_pass)
    accepted = bool(indep_pass and dist_pass)

    reasons = {"passed": [], "failed": []}
    if p11_pass: reasons["passed"].append("Bernoulli independence: P11~p_hat within tolerance")
    else:        reasons["failed"].append("Bernoulli independence: P11 deviates from p_hat beyond tolerance")
    if r1_pass:  reasons["passed"].append("Bernoulli independence: lag-1 autocorr ~ 0 within tolerance")
    else:        reasons["failed"].append("Bernoulli independence: lag-1 autocorr too large")
    if loss_geom_pass: reasons["passed"].append("Loss-run lengths ~ geometric")
    else:              reasons["failed"].append("Loss-run lengths reject geometric")
    if succ_geom_pass: reasons["passed"].append("Success-run lengths ~ geometric")
    else:              reasons["failed"].append("Success-run lengths reject geometric")

    return {
        "T": T, "n1": n1, "p_hat": p_hat,
        "independence": {
            "P11_hat": P11_hat, "P1_cond1_hat": P1_cond1_hat,
            "SE_L": SE_L, "SE_P11": SE_P11, "tol_diff": tol_P11,
            "prev1": prev1, "both11": both11,
            "lag1_autocorr": r1, "tol_r1": tol_r1,
            "p11_pass": p11_pass, "r1_pass": r1_pass, "passed": indep_pass
        },
        "run_length": {
            "loss_geom": res_loss, "loss_stats": _run_stats(loss_runs_all),
            "success_geom": res_succ, "success_stats": _run_stats(succ_runs_all),
            "loss_pass": loss_geom_pass, "success_pass": succ_geom_pass,
            "passed": dist_pass
        },
        "reasons": reasons,
        "accepted": accepted
    }

# ------------------------- GE verification -------------------------



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



def bernoulli_kfold_verification(X_list, paths, k=5, seed=0, alpha=0.05, debug=False):
    n = len(X_list)
    folds = _kfold_indices(n, k, seed)
    indep_pass = []; run_pass = []
    results = {"folds": []}
    for i, test_idx in enumerate(folds, start=1):
        train_idx = [j for j in range(n) if j not in test_idx]
        X_train = [X_list[j] for j in train_idx]
        X_test  = [X_list[j] for j in test_idx]
        # Fit on train
        bern = fit_bernoulli_multiple(X_train)
        p_model = bern["p_hat"]
        # Verify on test
        oos = verify_bernoulli_oos(X_test, p_model=p_model, alpha=alpha)
        indep_pass.append(bool(oos["independence"]["passed"]))
        run_pass.append(bool(oos["run_length"]["passed"]))
        if debug:
            print(f"\\n[CV fold {i}/{k}]")
            print("Train files:", ", ".join(paths[j] for j in train_idx))
            print("Test files :", ", ".join(paths[j] for j in test_idx))
            print("indep_pass=%s, runlen_pass=%s, p_model=%.6g" % (oos['independence']['passed'], oos['run_length']['passed'], p_model))
        results["folds"].append({"train_idx": train_idx, "test_idx": test_idx, "oos": oos, "p_model": p_model})
    results["summary"] = {
        "independence_pass_rate": float(sum(indep_pass)) / len(indep_pass) if indep_pass else float("nan"),
        "runlen_pass_rate": float(sum(run_pass)) / len(run_pass) if run_pass else float("nan"),
        "folds": len(folds)
    }
    return results



def _load_folder_build_X(folder: str, recursive: bool):
    pat = "**/*.txt" if recursive else "*.txt"
    paths = sorted(glob(os.path.join(folder, pat), recursive=recursive))
    if not paths:
        print(f"No .txt files found in {folder!r}")
        sys.exit(2)
    X_list = []
    for p in paths:
        N, S = parse_loss_file(p)
        X_list.append(build_X(N, S))
    return paths, X_list

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bernoulli-only: fit and verify on .txt files")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kfold-verify", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        tee = _Tee(os.path.join(args.dir, "details.txt"))
        sys.stdout = tee
        sys.stderr = tee

    from aux_loss_models import parse_loss_file, build_X
    paths, X_list = _load_folder_build_X(args.dir, args.recursive)

    # Fit and report
    bern = fit_bernoulli_multiple(X_list)
    print("== Bernoulli fit ==")
    print(f"N_total={bern['N_total']}  n1_total={bern['n1_total']}  p_hat={bern['p_hat']:.8f}")

    # In-sample verify
    print("\n=== Bernoulli verification (in-sample) ===")
    chk = verify_bernoulli(X_list, alpha=args.alpha)
    print("Independence passed:", chk["independence"]["passed"])
    print("Run-length passed :", chk["run_length"]["passed"])

    # Optional 5-fold
    if args.kfold_verify:
        print("\n=== 5-fold cross verification (Bernoulli) ===")
        cv = bernoulli_kfold_verification(X_list, paths, k=5, seed=0, alpha=args.alpha, debug=args.debug)
        summ = cv["summary"]
        print("Bernoulli: independence pass rate=%.2f, run-length pass rate=%.2f" %
              (summ["independence_pass_rate"], summ["runlen_pass_rate"]))

if __name__ == "__main__":
    main()
