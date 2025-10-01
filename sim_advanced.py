
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

REWARD = [1.05, 1.12, 1.21, 1.32, 1.45, 1.6, 1.77, 1.96, 2.18, 2.5, 2.75, 3.05, 3.4, 3.8, 4.25, 4.75, 5.3, 5.9, 6.55, 10.0, 11.2, 12.5, 13.9, 15.4, 17.0, 18.7, 20.5, 22.4, 24.4, 50.0, 58.0, 66.0, 74.0, 82.0, 90.0, 98.0, 106.0, 114.0, 122.0, 300.0, 400.0, 550.0, 750.0, 1000.0, 2000.0, 4000.0, 8000.0, 20000.0, 50000.0, 100000.0]
BASE_FAIL = [0.002, 0.003, 0.004, 0.005, 0.006, 0.006999999999999999, 0.008, 0.009000000000000001, 0.0095, 0.01, 0.012, 0.013999999999999999, 0.016, 0.018000000000000002, 0.02, 0.022000000000000002, 0.024, 0.026000000000000002, 0.027999999999999997, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.3]

CFG = dict(
    canvas_prob=0.08,            # step<=35, +1.0x
    landmark_prob=0.01,          # steps 28..44, fail=0.75; success +1000x; drone invalid; nozzle mutex
    landmark_fail=0.75,
    landmark_bonus=1000.0,
    lookout_cost_mult=0.50,      # one death shield (on fail)
    drone_cost_mult=0.50,        # next step safe (unless landmark)
    nozzle_cost_mult=0.40,       # ×2 at step 45 (unless landmark)
    template_cost_mult=0.60,     # jump +3 at step 35 -> 38
    money_cap=50000.0,
    mult_cap=5000.0,
    alpha_by_bet=0.5,            # extra bet-related cap (mult <= alpha * money_cap / bet)
    min_fail_after_45=0.35,      # floor after step>=45
    post40_abs_bump=0.05         # +5% absolute bump for 40..49 before scaling
)

def odds_scale(p, lam):
    p = np.clip(p, 0.0, 1.0-1e-12)
    odds = p/(1.0-p)
    odds_scaled = lam*odds
    p_scaled = odds_scaled/(1.0+odds_scaled)
    return np.clip(p_scaled, 0.0, 1.0)

def apply_zone_lambda(p_base, lam1=1.0, lam2=1.0, lam3=1.0):
    p = np.array(p_base, dtype=float)
    p[:30] = odds_scale(p[:30], lam1)
    p[30:40] = odds_scale(p[30:40], lam2)
    p[40:] = odds_scale(p[40:], lam3)
    return p

def after_probs(p_fail):
    ps = 1-p_fail
    out = np.empty(len(ps))
    acc = 1.0
    for i, s in enumerate(ps):
        acc *= s
        out[i] = acc
    return out

def effective_mult(mult, bet, cfg):
    caps = []
    if cfg.get("money_cap") is not None:
        caps.append(cfg["money_cap"]/float(bet))
    if cfg.get("mult_cap") is not None:
        caps.append(float(cfg["mult_cap"]))
    if cfg.get("alpha_by_bet") is not None and cfg.get("money_cap") is not None:
        caps.append(cfg["alpha_by_bet"]*(cfg["money_cap"]/float(bet)))
    if caps:
        return min(mult, min(caps))
    return mult

def build_fail_curve(lam1, lam2, lam3, cfg=CFG):
    p = np.array(BASE_FAIL, dtype=float)
    p[39:49] = np.clip(p[39:49] + cfg["post40_abs_bump"], 0.0, 1.0)
    p = apply_zone_lambda(p, lam1, lam2, lam3)
    for i in range(44, 49):
        p[i] = max(p[i], cfg["min_fail_after_45"])
    return p

def best_step_no_items(p_fail, bet, cfg=CFG):
    after = after_probs(p_fail)
    rm = np.array(REWARD, dtype=float)
    rm = np.array([effective_mult(m, bet, cfg) for m in rm])
    ev = rm * after
    return int(np.argmax(ev)) + 1

def simulate_round(bet, p_fail, rng, cfg=CFG):
    lookout_left = 1
    drone_left = 1
    nozzle_left = 1
    template_left = 1
    cost_spent = 0.0
    step = 0
    alive = True
    current_mult = 1.0
    mult_bonus = 0.0
    drone_buffer = 0

    stop_step = best_step_no_items(p_fail, bet, cfg)

    while alive and step < stop_step:
        next_step = step + 1

        if template_left>0 and next_step==35:
            template_left -= 1
            cost_spent += cfg["template_cost_mult"]*bet
            step = 38
            current_mult = REWARD[step-1]
            continue

        is_canvas = (next_step <= 35) and (rng.random() < cfg["canvas_prob"])
        is_landmark = (28 <= next_step <= 44) and (rng.random() < cfg["landmark_prob"])

        if drone_left>0 and (next_step in (41,43,45)) and (not is_landmark):
            drone_buffer = 1
            drone_left -= 1
            cost_spent += cfg["drone_cost_mult"]*bet

        step_fail = cfg["landmark_fail"] if is_landmark else p_fail[next_step-1]
        if drone_buffer==1 and not is_landmark:
            step_fail_eff = 0.0; drone_buffer = 0
        else:
            step_fail_eff = step_fail

        if rng.random() < step_fail_eff:
            if lookout_left>0:
                lookout_left -= 1
                cost_spent += cfg["lookout_cost_mult"]*bet
            else:
                alive = False
                break
        else:
            step = next_step
            current_mult = REWARD[step-1]
            if is_canvas:
                current_mult += 1.0
            if is_landmark:
                mult_bonus += cfg["landmark_bonus"]
            if nozzle_left>0 and step==45 and (not is_landmark):
                nozzle_left -= 1
                cost_spent += cfg["nozzle_cost_mult"]*bet
                current_mult *= 2.0

    if alive and step>=1:
        total_mult = current_mult + mult_bonus
        total_mult = effective_mult(total_mult, bet, cfg)
        payout = total_mult * bet
    else:
        payout = 0.0

    return max(0.0, payout - cost_spent)

def simulate(bet, lam1=1.0, lam2=1.0, lam3=1.0, sims=20000, seed=123):
    rng = np.random.default_rng(seed)
    p_fail = build_fail_curve(lam1, lam2, lam3)
    payouts = np.zeros(sims, dtype=float)
    for i in range(sims):
        payouts[i] = simulate_round(bet, p_fail, rng)
    rtp = payouts.sum()/(bet*sims)
    return dict(RTP=rtp, Mean=float(payouts.mean()), Std=float(payouts.std(ddof=1)))
