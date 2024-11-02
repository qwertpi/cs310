with open("logreg_arglist.txt", "w") as f:
    for C in [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]:
        for l1ratio in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            f.write(f"{l1ratio} {C}\n")
