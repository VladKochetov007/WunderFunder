_fig, _ax = plt.subplots(1, 3, figsize=(13, 3.4))
for _c in range(4):
    _sel = CLS == _c
    _y = np.clip(Y[_sel][MASK[_sel]], -2, 2)
    _ax[0].hist(_y[:, 0], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c], label=f"class {_c}")
    _ax[1].hist(_y[:, 1], bins=81, range=(-2, 2), histtype="step", density=True, color=CLASS_COLORS[_c])
_ax[0].set_title("t0 on scored rows (clipped)"); _ax[1].set_title("t1 on scored rows (clipped)"); _ax[0].legend(fontsize=8)
_y3 = np.clip(Y[CLS == 3][MASK[CLS == 3]], -2, 2)
_ax[2].hexbin(_y3[:, 0], _y3[:, 1], gridsize=40, bins="log", cmap="Blues"); _ax[2].set_title("class 3: t0 vs t1 (log count)")
_ax[2].set_xlabel("t0"); _ax[2].set_ylabel("t1")
save(_fig, "01_target_distributions")
