
# GK Enhanced Spec + Simulators

## What you get
1) **Spec with Caps** (money cap + multiplier cap) + **alerts** for any step where EV_at_step > 0.96.  
2) **Advanced Monte Carlo** with items & curve breakers (conservative defaults).  
3) **CLI** (`cli.py`) and **Makefile** to run common tasks quickly.

### Quick start
```bash
make spec   # writes GK_Probability_Spec_withCaps.csv + alert + chart
make adv    # run advanced simulation
```

- Step50 base success = 70% is built into BASE_FAIL.
- You can tweak items/curve-breakers in `sim_advanced.py` (CFG dict).
