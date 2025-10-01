
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sim_advanced import REWARD, BASE_FAIL, odds_scale, after_probs, simulate

def gen_spec_with_caps(target, money_cap, mult_cap, out_csv, out_png, out_alert):
    cap_mult_effective = min(mult_cap, money_cap/1.0)
    lo, hi = 0.1, 200.0
    best = None
    for _ in range(40):
        mid = 0.5*(lo+hi)
        p = odds_scale(BASE_FAIL, mid)
        rm = np.minimum(np.array(REWARD,float), cap_mult_effective)
        after = after_probs(p)
        ev = rm*after
        gap = abs(ev.max()-target)
        if (best is None) or (gap<best[0]):
            best = (gap, mid, p, ev, after)
        if ev.max()>target: lo = mid
        else: hi = mid
    gap, lam, p, ev, after = best
    step = int(np.argmax(ev))+1

    df = pd.DataFrame({
        "Step": np.arange(1,51),
        "RewardMultiplier": REWARD,
        f"ScaledFailProb(odds,lambda={lam:.6f})": p,
        "ScaledSurvivalProb": 1-p,
        "ReachProb(before step)": np.r_[1.0, after[:-1]],
        "AfterProb(through step)": after,
        "EV_at_step(with caps)": ev,
    })
    df["Alert(>0.96)"] = (df["EV_at_step(with caps)"]>0.96).astype(int)
    df.to_csv(out_csv, index=False)
    df[df["Alert(>0.96)"]==1][["Step","RewardMultiplier","EV_at_step(with caps)"]].to_csv(out_alert, index=False)

    plt.figure(figsize=(9,5))
    plt.plot(df["Step"], df["EV_at_step(with caps)"], label="EV (with caps)")
    plt.axhline(0.96, linestyle="--"); plt.axvline(step, linestyle=":")
    plt.xlabel("Step"); plt.ylabel("EV × bet"); plt.title(f"EV with caps (lambda≈{lam:.4f})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png)
    print(f"[spec] lambda*={lam:.6f}, best_step={step}, wrote {out_csv}, alerts: {out_alert}, chart: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("spec")
    sp.add_argument("--target", type=float, default=0.96)
    sp.add_argument("--money_cap", type=float, default=50000.0)
    sp.add_argument("--mult_cap", type=float, default=5000.0)
    sp.add_argument("--out_csv", default="GK_Probability_Spec_withCaps.csv")
    sp.add_argument("--out_png", default="GK_EV_curve_withCaps.png")
    sp.add_argument("--out_alert", default="GK_Alert_Report.csv")

    adv = sub.add_parser("adv")
    adv.add_argument("--bet", type=float, default=1.0)
    adv.add_argument("--lam1", type=float, default=1.0)
    adv.add_argument("--lam2", type=float, default=1.05)
    adv.add_argument("--lam3", type=float, default=1.3)
    adv.add_argument("--sims", type=int, default=20000)
    adv.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()
    if args.cmd=="spec":
        gen_spec_with_caps(args.target, args.money_cap, args.mult_cap, args.out_csv, args.out_png, args.out_alert)
    elif args.cmd=="adv":
        res = simulate(args.bet, args.lam1, args.lam2, args.lam3, args.sims, args.seed)
        print(json.dumps(dict(bet=args.bet, lam1=args.lam1, lam2=args.lam2, lam3=args.lam3, sims=args.sims, **res), indent=2))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
