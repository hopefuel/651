#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, math, warnings
from typing import List, Tuple, Optional
import numpy as np
from aux_loss_models import (parse_loss_file, build_X, _flatten_X, _N_S_from_X_list, _runs_from_binary_array, lag1_autocorr,
    _adaptive_geom_chisquare, _run_stats, loss_runs_from_S, success_runs_from_N_S, _Tee, _kfold_indices)


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



def fit_ge_hmm_multiple(X_list: List[np.ndarray],
                        a0: float, b0: float, pG0: float, pB0: float, piG0: float, piB0: float,
                        estimate_pi: bool,
                        max_iter: int,
                        tol: float,
                        random_state: int) -> Optional[dict]:

    X_list = [x.astype(int, copy=False).reshape(-1,1) for x in X_list]
    X = np.vstack(X_list)
    lengths = [x.shape[0] for x in X_list]

    u = np.unique(X)
    if u.min() < 0 or u.max() > 1:
        raise ValueError(f"observations must be 0/1; got unique={u}")

    if _HAVE_CATEG:
        model = _CatHMM(
            n_components=2, n_iter=max_iter, tol=tol,
            init_params="", params=("ste" if estimate_pi else "te"),
            random_state=random_state, verbose=False
        )
    else:
        model = MultinomialHMM(
            n_components=2, n_iter=max_iter, tol=tol,
            init_params="", params=("ste" if estimate_pi else "te"),
            random_state=random_state, verbose=False
        )
        # If only one symbol present, append a dummy of the other to satisfy hmmlearn
        if len(np.unique(X)) == 1:
            X = np.vstack([X, np.array([[1 - int(X[0,0])]], dtype=int)])
            lengths = lengths + [1]

    startprob = np.array([piG0, piB0], float); startprob = np.clip(startprob, 1e-12, 1); startprob /= startprob.sum()
    transmat  = np.array([[1 - a0, a0],[b0, 1 - b0]], float); transmat = np.clip(transmat, 1e-12, 1); transmat /= transmat.sum(1, keepdims=True)
    emission  = np.array([[1 - pG0, pG0],[1 - pB0, pB0]], float); emission = np.clip(emission, 1e-12, 1); emission /= emission.sum(1, keepdims=True)

    model.startprob_ = startprob
    model.transmat_  = transmat
    try:
        setattr(model, "emissionprob_", emission)
    except Exception:
        setattr(model, "emissionprob", emission)

    model.fit(X, lengths)

    A = model.transmat_.copy()
    E = getattr(model, "emissionprob_", None)
    if E is None:
        E = getattr(model, "emissionprob", None)
        if E is None:
            raise AttributeError("Could not access emission probabilities from hmmlearn model.")
    E = E.copy()
    pi = model.startprob_.copy()

    pG = float(E[0, 1]); pB = float(E[1, 1])
    if pG > pB:
        A = A[[1, 0]][:, [1, 0]]
        E = E[[1, 0], :]
        pi = pi[[1, 0]]
        pG, pB = pB, pG

    a = float(A[0, 1]); b = float(A[1, 0])
    piG = float(pi[0]);  piB = float(pi[1])

    piG_stat = b / (a + b);  piB_stat = a / (a + b)
    L_hat = float(X.mean())
    L_model = piG * pG + piB * pB

    return {
        "a": a, "b": b, "pG": pG, "pB": pB,
        "piG_fit": piG, "piB_fit": piB,
        "piG_stat": piG_stat, "piB_stat": piB_stat,
        "L_hat": L_hat, "L_model": L_model,
        "model": model,
    }

# ------------------------- utilities -------------------------

def _run_stats(run_lengths):
    n = len(run_lengths)
    if n == 0:
        return {"n": 0, "mean": float('nan'), "std": float('nan'), "max": 0, "min": 0}
    s = sum(run_lengths)
    mean = s / n
    var = sum((x-mean)**2 for x in run_lengths) / n
    return {"n": n, "mean": mean, "std": math.sqrt(var), "max": max(run_lengths), "min": min(run_lengths)}



def verify_ge(X_list, ge_fit: dict, alpha=0.05, use_stationary_pi=True, decode_method="viterbi"):
    xs = _flatten_X(X_list)
    T = sum(len(x) for x in xs)
    if T < 2:
        return {"accepted": False, "reason": "insufficient data", "pi_source": ("stationary" if use_stationary_pi else "fitted"),
                "empirical": {"L_hat": float("nan"), "P11_hat": float("nan"), "P11_joint_hat": float("nan"), "r1_hat": float("nan"), "both11": 0},
                "model": {"a": float("nan"), "b": float("nan"), "pG": float("nan"), "pB": float("nan"),
                          "piG": float("nan"), "piB": float("nan"),
                          "L_model": float("nan"), "P11_model": float("nan"), "r1_model": float("nan")},
                "tolerances": {"SE_L": float("nan"), "SE_P11": float("nan"), "tol_r1": float("nan")},
                "moments_passed": False, "moments_breakdown": {"L": False, "P11": False, "r1": False},
                "state_runlength": {"good_geom": {"stat": float("nan"), "df": 0, "pvalue": float("nan"), "critical": float("nan"), "passed": False, "bins": []},
                                    "bad_geom": {"stat": float("nan"), "df": 0, "pvalue": float("nan"), "critical": float("nan"), "passed": False, "bins": []},
                                    "good_stats": {"n": 0, "mean": float("nan"), "std": float("nan"), "max": 0, "min": 0},
                                    "bad_stats": {"n": 0, "mean": float("nan"), "std": float("nan"), "max": 0, "min": 0},
                                    "passed": False},
                "T": T, "pairs": 0, "reasons": {"passed": [], "failed": ["insufficient data"]}}

    # empirical moments
    n1 = sum(int(x.sum()) for x in xs)
    L_hat = n1 / T
    pairs = sum(len(x) - 1 for x in xs if len(x) >= 2)
    both11 = 0
    for x in xs:
        if len(x) >= 2:
            both11 += int(((x[:-1] == 1) & (x[1:] == 1)).sum())
    P11_hat = both11 / pairs if pairs > 0 else float("nan")
    P11_joint_hat = P11_hat
    r1_hat = (P11_hat - L_hat * L_hat) / (L_hat * (1 - L_hat)) if 0 < L_hat < 1 else float("nan")

    # model moments
    a, b, pG, pB = ge_fit["a"], ge_fit["b"], ge_fit["pG"], ge_fit["pB"]
    if use_stationary_pi:
        piG = ge_fit["piG_stat"]; piB = ge_fit["piB_stat"]
        pi_src = "stationary"
    else:
        piG = ge_fit["piG_fit"];  piB = ge_fit["piB_fit"]
        pi_src = "fitted"

    L_model = piG * pG + piB * pB
    P11_model = piG * ((1 - a) * pG * pG + a * pG * pB) + \
                piB * (b * pB * pG + (1 - b) * pB * pB)
    r1_model = (P11_model - L_model * L_model) / (L_model * (1 - L_model)) if 0 < L_model < 1 else float("nan")

    # tolerances
    SE_L = math.sqrt(L_hat * (1 - L_hat) / T) if 0 < L_hat < 1 else float("nan")
    SE_P11 = math.sqrt(P11_hat * (1 - P11_hat) / pairs) if 0 < P11_hat < 1 and pairs > 0 else float("nan")
    tol_r1 = 2.0 / math.sqrt(max(1, pairs))

    pass_L   = abs(L_model - L_hat)   <= (2 * SE_L if not math.isnan(SE_L) else float("inf"))
    pass_P11 = abs(P11_model - P11_hat) <= (2 * SE_P11 if not math.isnan(SE_P11) else float("inf"))
    pass_r1  = abs(r1_model - r1_hat) <= tol_r1
    mom_pass = bool(pass_L and pass_P11 and pass_r1)

    # 2.2 state run-lengths ~ geometric
    model = ge_fit.get("model", None)
    if model is None:
        warnings.warn("GE verify: model object not found; skipping state run-length tests.")
        run_pass = False
        res_G = {"stat": float("nan"), "df": 0, "pvalue": float("nan"), "critical": float("nan"), "passed": False, "bins": []}
        res_B = {"stat": float("nan"), "df": 0, "pvalue": float("nan"), "critical": float("nan"), "passed": False, "bins": []}
        st_runs_G_stats = {"n": 0, "mean": float("nan"), "std": float("nan"), "max": 0, "min": 0}
        st_runs_B_stats = {"n": 0, "mean": float("nan"), "std": float("nan"), "max": 0, "min": 0}
    else:
        states, _ = _decode_states(model, X_list, method=decode_method)
        lengths = [x.shape[0] for x in X_list]
        st_runs_G, st_runs_B = [], []
        offs = 0
        for L in lengths:
            s = states[offs:offs+L]; offs += L
            runs = []
            cur = s[0]; clen = 1
            for v in s[1:]:
                if v == cur: clen += 1
                else:
                    runs.append((cur, clen)); cur = v; clen = 1
            runs.append((cur, clen))
            for lab, Lr in runs:
                (st_runs_G if lab == 0 else st_runs_B).append(Lr)

        res_G = _adaptive_geom_chisquare(st_runs_G, theta=max(1e-9, a), alpha=alpha)
        res_B = _adaptive_geom_chisquare(st_runs_B, theta=max(1e-9, b), alpha=alpha)
        st_runs_G_stats = _run_stats(st_runs_G)
        st_runs_B_stats = _run_stats(st_runs_B)
        run_pass = res_G["passed"] and res_B["passed"]

    accepted = bool(mom_pass and run_pass)

    reasons = {"passed": [], "failed": []}
    if pass_L:   reasons["passed"].append("GE moment: L within tolerance")
    else:        reasons["failed"].append("GE moment: L mismatch beyond tolerance")
    if pass_P11: reasons["passed"].append("GE moment: P11 within tolerance")
    else:        reasons["failed"].append("GE moment: P11 mismatch beyond tolerance")
    if pass_r1:  reasons["passed"].append("GE moment: r1 within tolerance")
    else:        reasons["failed"].append("GE moment: r1 mismatch beyond tolerance")
    if res_G["passed"]: reasons["passed"].append("GE state-run: G runs ~ geometric(a)")
    else:               reasons["failed"].append("GE state-run: G runs reject geometric")
    if res_B["passed"]: reasons["passed"].append("GE state-run: B runs ~ geometric(b)")
    else:               reasons["failed"].append("GE state-run: B runs reject geometric")

    return {
        "T": T, "pairs": pairs, "pi_source": pi_src,
        "empirical": {"L_hat": L_hat, "P11_hat": P11_hat, "P11_joint_hat": P11_joint_hat, "r1_hat": r1_hat, "both11": both11},
        "model": {"a": a, "b": b, "pG": pG, "pB": pB, "piG": piG, "piB": piB,
                  "L_model": L_model, "P11_model": P11_model, "r1_model": r1_model},
        "tolerances": {"SE_L": SE_L, "SE_P11": SE_P11, "tol_r1": tol_r1},
        "moments_passed": mom_pass,
        "moments_breakdown": {"L": pass_L, "P11": pass_P11, "r1": pass_r1},
        "state_runlength": {"good_geom": res_G, "bad_geom": res_B, "good_stats": st_runs_G_stats, "bad_stats": st_runs_B_stats, "passed": run_pass},
        "reasons": reasons,
        "accepted": accepted
    }

# ------------------------- Model selection -------------------------



def ge_kfold_verification(X_list, paths, k=5, seed=0, estimate_pi=False, alpha=0.05, max_iter=200, tol=1e-6, decode_method="viterbi", debug=False):
    n = len(X_list)
    folds = _kfold_indices(n, k, seed)
    ge_mom_pass = []; ge_G_pass = []; ge_B_pass = []
    results = {"folds": []}
    for i, test_idx in enumerate(folds, start=1):
        train_idx = [j for j in range(n) if j not in test_idx]
        X_train = [X_list[j] for j in train_idx]
        X_test  = [X_list[j] for j in test_idx]
        # GE fit
        N_list, S_list = _N_S_from_X_list(X_train)
        a0,b0,pG0,pB0,piG0,piB0,_ = ge_initial_params_from_runs_multi(N_list, S_list, X_train)
        ge = fit_ge_hmm_multiple(X_train, a0,b0,pG0,pB0,piG0,piB0,
                                 estimate_pi=estimate_pi, max_iter=max_iter, tol=tol, random_state=seed)
        # Verify
        g = verify_ge(X_test, ge, alpha=alpha, use_stationary_pi=True, decode_method=decode_method)
        sr = g.get("state_runlength", {})
        ge_mom_pass.append(bool(g.get("moments_passed", False)))
        ge_G_pass.append(bool(sr.get("good_geom", {}).get("passed", False)))
        ge_B_pass.append(bool(sr.get("bad_geom", {}).get("passed", False)))
        if debug:
            print(f"\\n[CV fold {i}/{k}]")
            print("Train files:", ", ".join(paths[j] for j in train_idx))
            print("Test files :", ", ".join(paths[j] for j in test_idx))
            print("moments_pass=%s, G_pass=%s, B_pass=%s" % (ge_mom_pass[-1], ge_G_pass[-1], ge_B_pass[-1]))
        results["folds"].append({"train_idx": train_idx, "test_idx": test_idx, "GE": g})
    results["summary"] = {
        "moments_pass_rate": float(sum(ge_mom_pass)) / len(ge_mom_pass) if ge_mom_pass else float("nan"),
        "G_run_pass_rate": float(sum(ge_G_pass)) / len(ge_G_pass) if ge_G_pass else float("nan"),
        "B_run_pass_rate": float(sum(ge_B_pass)) / len(ge_B_pass) if ge_B_pass else float("nan"),
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
    ap = argparse.ArgumentParser(description="GE-only: fit and verify 2-state GE HMM on .txt files")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--estimate-pi", action="store_true")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--decode", choices=["viterbi","posterior"], default="viterbi")
    ap.add_argument("--ppc-sims", type=int, default=0)
    ap.add_argument("--kfold-verify", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        tee = _Tee(os.path.join(args.dir, "details.txt"))
        sys.stdout = tee
        sys.stderr = tee

    # Build data
    from aux_loss_models import parse_loss_file, build_X
    from glob import glob
    paths, X_list = _load_folder_build_X(args.dir, args.recursive)

    # Fit initials and model
    N_list, S_list = _N_S_from_X_list(X_list)
    a0,b0,pG0,pB0,piG0,piB0,_ = ge_initial_params_from_runs_multi(N_list, S_list, X_list)
    print("\n== GE initials ==")
    print(f"a0={a0:.6f}  b0={b0:.6f}  pG0={pG0:.6f}  pB0={pB0:.6f}  piG0={piG0:.6f}  piB0={piB0:.6f}")
    ge = fit_ge_hmm_multiple(X_list, a0,b0,pG0,pB0,piG0,piB0,
                             estimate_pi=args.estimate_pi, max_iter=args.max_iter, tol=args.tol, random_state=args.seed)
    print("\n== GE fit ==")
    print(f"a={ge['a']:.8f}  b={ge['b']:.8f}  pG={ge['pG']:.8f}  pB={ge['pB']:.8f}")

    # Verify in-sample
    print("\n=== GE verification (in-sample) ===")
    ge_chk = verify_ge(X_list, ge, alpha=args.alpha, use_stationary_pi=True, decode_method=args.decode)
    print(f"moments+correlation: passed={ge_chk['moments_passed']}")
    sr = ge_chk.get("state_runlength", {})
    print(f"State-run GOF: G_pass={sr.get('good_geom',{}).get('passed')}, B_pass={sr.get('bad_geom',{}).get('passed')}")

    # Optional: 5-fold verification
    if args.kfold_verify:
        print("\n=== 5-fold cross verification (GE) ===")
        cv = ge_kfold_verification(X_list, paths, k=5, seed=args.seed, estimate_pi=args.estimate_pi,
                                   alpha=args.alpha, max_iter=args.max_iter, tol=args.tol,
                                   decode_method=args.decode, debug=args.debug)
        summ = cv["summary"]
        print("GE: moments pass rate=%.2f, G-run pass rate=%.2f, B-run pass rate=%.2f" %
              (summ["moments_pass_rate"], summ["G_run_pass_rate"], summ["B_run_pass_rate"]))

if __name__ == "__main__":
    main()
