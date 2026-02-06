import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def _aic_from_rss(rss, n, k):
    if not np.isfinite(rss) or rss <= 0 or n <= k:
        return np.nan
    return n * np.log(rss / n) + 2 * k

def fit_plateau_models(m, y, c_grid=None, b_grid=None):

    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(m) & np.isfinite(y)
    m = m[mask]; y = y[mask]
    n = len(y)
    if n < 4:
        return None

    if c_grid is None:

        y_min = float(np.nanmin(y))
        c_grid = np.linspace(0.0, max(1e-12, 1.2 * y_min), 60)
    if b_grid is None:
        b_grid = np.logspace(-4, 1, 120)

    fits = []


    best = dict(name="exp_plateau", rss=np.inf, params=None, aic=np.nan)
    for c in c_grid:
        yc = y - c

        for b in b_grid:
            f = np.exp(-b * m)
            denom = np.dot(f, f)
            if denom <= 0:
                continue
            a = np.dot(f, yc) / denom
            yhat = c + a * f
            rss = float(np.sum((y - yhat) ** 2))
            if rss < best["rss"]:
                best["rss"] = rss
                best["params"] = (a, b, c)
    best["aic"] = _aic_from_rss(best["rss"], n=n, k=3)
    fits.append(best)


    best = dict(name="power_plateau", rss=np.inf, params=None, aic=np.nan)
    for c in c_grid:
        yc = y - c
        for b in b_grid:
            f = np.power(m, -b)
            denom = np.dot(f, f)
            if denom <= 0:
                continue
            a = np.dot(f, yc) / denom
            yhat = c + a * f
            rss = float(np.sum((y - yhat) ** 2))
            if rss < best["rss"]:
                best["rss"] = rss
                best["params"] = (a, b, c)
    best["aic"] = _aic_from_rss(best["rss"], n=n, k=3)
    fits.append(best)


    best = dict(name="rational_plateau", rss=np.inf, params=None, aic=np.nan)

    b2_grid = np.linspace(0.0, max(1.0, float(np.max(m))), 120)
    for c in c_grid:
        yc = y - c
        for b2 in b2_grid:
            f = 1.0 / (m + b2 + 1e-12)
            denom = np.dot(f, f)
            if denom <= 0:
                continue
            a = np.dot(f, yc) / denom
            yhat = c + a * f
            rss = float(np.sum((y - yhat) ** 2))
            if rss < best["rss"]:
                best["rss"] = rss
                best["params"] = (a, b2, c)
    best["aic"] = _aic_from_rss(best["rss"], n=n, k=3)
    fits.append(best)


    fits = [f for f in fits if np.isfinite(f["aic"])]
    if not fits:
        return None
    best = min(fits, key=lambda d: d["aic"])


    m_grid = np.linspace(float(np.min(m)), float(np.max(m)), 500)
    a, b, c = best["params"]
    if best["name"] == "exp_plateau":
        y_grid = c + a * np.exp(-b * m_grid)
        eq = f"y = {c:.3g} + {a:.3g} exp(-{b:.3g} m)"
    elif best["name"] == "power_plateau":
        y_grid = c + a * (m_grid ** (-b))
        eq = f"y = {c:.3g} + {a:.3g} m^(-{b:.3g})"
    else:
        y_grid = c + a / (m_grid + b)
        eq = f"y = {c:.3g} + {a:.3g} / (m + {b:.3g})"

    return {
        "best_name": best["name"],
        "params": best["params"],
        "aic": best["aic"],
        "m_grid": m_grid,
        "y_grid": y_grid,
        "equation": eq,
    }




def fit_delta_vs_lambda2(df, xcol="lambda2_mean", ycol="delta_mean",
                         models=("exp_plateau", "power_plateau", "rational_plateau"),
                         eps=1e-8, do_plot=True, out_png=None):
    """
    Fit Δ(λ2) using a few simple analytical forms, pick best by AIC.
    df: your per-(k,p) summary table (in your code: df_sum or df_kp).
    """
    x = np.asarray(df[xcol], dtype=float)
    y = np.asarray(df[ycol], dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    order = np.argsort(x)
    x = x[order]; y = y[order]

    def aic(rss, n, k_params):
        rss = max(float(rss), 1e-30)
        return n * np.log(rss / n) + 2 * k_params

    # Models (Δ∞ + ...)
    def f_exp(x, A, b, c):
        return c + A * np.exp(-b * x)

    def f_power(x, A, beta, c, x0):
        return c + A * (x + x0)**(-beta)

    def f_rat(x, A, b, c):
        return c + A / (1.0 + b * x)

    fits = []

    try:
        from scipy.optimize import curve_fit

        if "exp_plateau" in models:
            c0 = float(np.min(y))
            A0 = float(np.max(y) - c0)
            b0 = 1.0 / (float(np.mean(x)) + eps)
            popt, _ = curve_fit(
                f_exp, x, y, p0=[A0, b0, c0],
                bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0]),
                maxfev=20000
            )
            yhat = f_exp(x, *popt)
            rss = np.sum((y - yhat)**2)
            fits.append(dict(name="exp_plateau",
                             params=dict(A=float(popt[0]), b=float(popt[1]), delta_inf=float(popt[2])),
                             rss=float(rss), aic=float(aic(rss, len(x), 3))))

        if "power_plateau" in models:
            c0 = float(np.min(y))
            A0 = float(np.max(y) - c0)
            popt, _ = curve_fit(
                f_power, x, y, p0=[A0, 1.0, c0, 0.05],
                bounds=([-np.inf, 0.0, 0.0, 1e-6], [np.inf, 10.0, 1.0, 5.0]),
                maxfev=40000
            )
            yhat = f_power(x, *popt)
            rss = np.sum((y - yhat)**2)
            fits.append(dict(name="power_plateau",
                             params=dict(A=float(popt[0]), beta=float(popt[1]),
                                         delta_inf=float(popt[2]), x0=float(popt[3])),
                             rss=float(rss), aic=float(aic(rss, len(x), 4))))

        if "rational_plateau" in models:
            c0 = float(np.min(y))
            A0 = float(np.max(y) - c0)
            popt, _ = curve_fit(
                f_rat, x, y, p0=[A0, 1.0, c0],
                bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, 1.0]),
                maxfev=20000
            )
            yhat = f_rat(x, *popt)
            rss = np.sum((y - yhat)**2)
            fits.append(dict(name="rational_plateau",
                             params=dict(A=float(popt[0]), b=float(popt[1]), delta_inf=float(popt[2])),
                             rss=float(rss), aic=float(aic(rss, len(x), 3))))

    except Exception:

        if "rational_plateau" in models:
            b_grid = np.logspace(-3, 3, 200)
            best = None
            for b in b_grid:
                phi = 1.0 / (1.0 + b * x)
                X = np.column_stack([phi, np.ones_like(phi)])  # y = A*phi + c
                (A_est, c_est), *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat = A_est * phi + c_est
                rss = float(np.sum((y - yhat)**2))
                cand = dict(name="rational_plateau",
                            params=dict(A=float(A_est), b=float(b), delta_inf=float(c_est)),
                            rss=rss, aic=float(aic(rss, len(x), 3)))
                if (best is None) or (cand["aic"] < best["aic"]):
                    best = cand
            fits.append(best)

    best = min(fits, key=lambda d: d["aic"])

    if do_plot:
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 400)

        def eval_fit(name, params, xx):
            if name == "exp_plateau":
                return params["delta_inf"] + params["A"] * np.exp(-params["b"] * xx)
            if name == "power_plateau":
                return params["delta_inf"] + params["A"] * (xx + params["x0"])**(-params["beta"])
            if name == "rational_plateau":
                return params["delta_inf"] + params["A"] / (1.0 + params["b"] * xx)
            raise ValueError(name)

        plt.figure(dpi=250)
        plt.scatter(x, y, s=45, alpha=0.85, label="(k,p) means")
        for f in fits:
            yy = eval_fit(f["name"], f["params"], xx)
            lw = 2.7 if f["name"] == best["name"] else 1.2
            plt.plot(xx, yy, linewidth=lw, label=f"{f['name']} (AIC={f['aic']:.1f})")
        plt.xlabel("Mean λ₂")
        plt.ylabel("Mean Δ")
        plt.title("Δ vs λ₂: fitted analytical forms")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if out_png is not None:
            plt.savefig(out_png)
            plt.close()
        else:
            plt.show()

    return best, fits
