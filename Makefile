
.PHONY: all spec adv clean
all: spec

spec:
	python cli.py spec --target 0.96 --money_cap 50000 --mult_cap 5000

adv:
	python cli.py adv --bet 1.0 --lam1 1.0 --lam2 1.05 --lam3 1.3 --sims 20000

clean:
	rm -f GK_Probability_Spec_withCaps.csv GK_EV_curve_withCaps.png GK_Alert_Report.csv
