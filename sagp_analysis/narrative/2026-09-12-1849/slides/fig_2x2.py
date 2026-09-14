"""The 2x2 design figure for the sagp slides: kernel structure x sparsity parameterisation.
Paper style: white ground, STIX serif text and math, one accent colour, curves are real Matern-5/2 draws.
Run: srun -t 00:10:00 --mem=4G latest/venv/bin/python fig_2x2.py  (writes fig_2x2.png and fig_2x2.pdf here)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
INK, GREY, LINE, ACC, ACC_BG = "#1a1a1a", "#555555", "#9a9a9a", "#1f6f68", "#eef5f4"

def matern52_draw(n=400, ell=0.12, seed=3):
    x = np.linspace(0.0, 1.0, n)
    r = np.abs(x[:, None] - x[None, :]) / ell
    K = (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r) + 1e-8 * np.eye(n)
    return np.linalg.cholesky(K) @ np.random.default_rng(seed).standard_normal(n)

def curve(ax, x0, y0, w, h, f, color, lw=1.3, xfrac=1.0, scale=1.0, amax=None, start=0, recentre=False):
    """Plot a window of the draw f (fraction xfrac, from index start), stretched to width w around y0.
    amax fixes the vertical scale so the three states of one draw are comparable; recentre removes the window's mean."""
    amax = np.abs(f).max() if amax is None else amax
    m = max(2, int(len(f) * xfrac)); y = f[start:start + m] * scale
    if recentre:
        y = y - y.mean()
    ax.plot(np.linspace(x0, x0 + w, len(y)), y0 + 0.5 * h * y / amax, color=color, lw=lw, solid_capstyle="round")

def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK, shrinkA=0, shrinkB=0, mutation_scale=10))

W, H = 11.0, 6.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

f0 = matern52_draw(seed=3); A = np.abs(f0).max()
g1, g2 = matern52_draw(ell=0.18, seed=11), matern52_draw(ell=0.18, seed=7)

cols = {"ell": dict(x=3.2, c=5.05), "amp": dict(x=7.2, c=9.05)}; CW = 3.7
rows = {"add": dict(y=2.5, c=3.45), "prod": dict(y=0.4, c=1.35)}; RH = 1.9

# ---- column headers: the two ways a component is switched off
ax.text(cols["ell"]["c"], 6.18, r"Sparsity on $\rho_i = \ell_i^{-2}$", ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
ax.text(cols["ell"]["c"], 5.84, r"$\rho_i \to 0$: the component flattens", ha="center", va="center", fontsize=10, color=GREY)
ax.text(cols["amp"]["c"], 6.18, r"Sparsity on $a_i^2$", ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
ax.text(cols["amp"]["c"], 5.84, r"$a_i^2 \to 0$: the component vanishes", ha="center", va="center", fontsize=10, color=GREY)
# the stretched state is a window of the same draw around one of its extrema (zero slope), recentred
win = len(f0) // 20
d = np.diff(f0); interior = [i for i in range(1, len(d)) if d[i - 1] * d[i] < 0 and win <= i <= len(f0) - win]
i_ext = max(interior, key=lambda i: abs(f0[i]))  # the interior extremum with the largest amplitude
start = i_ext - win // 2
for key, xfrac, scale, st, rc_, cap in (("ell", 1 / 20, 1.0, start, True, r"$\ell_i \to \infty$: the same draw, stretched"),
                                        ("amp", 1.0, 0.1, 0, False, r"$a_i \to 0$: the same draw, scaled down")):
    c = cols[key]["c"]; yg = 5.25
    curve(ax, c - 1.75, yg, 1.2, 0.55, f0, INK, amax=A, recentre=True)
    arrow(ax, c - 0.42, c + 0.02, yg)
    curve(ax, c + 0.2, yg, 1.5, 0.55, f0, ACC, lw=1.7, xfrac=xfrac, scale=scale, amax=A, start=st, recentre=rc_)
    ax.text(c, 4.78, cap, ha="center", va="center", fontsize=8.5, color=GREY)

# ---- row headers: the two kernel structures
for key, name, formula, note, op in (("add", "Additive", r"$k(x,x') = \sum_i k_i(x_i, x_i')$", "first-order effects only", "+"),
                                     ("prod", "Product", r"$k(x,x') = \prod_i k_i(x_i, x_i')$", "interactions representable", r"$\times$")):
    rc = rows[key]["c"]
    ax.text(0.3, rc + 0.55, name, ha="left", va="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(0.3, rc + 0.2, formula, ha="left", va="center", fontsize=10.5, color=INK)
    ax.text(0.3, rc - 0.1, note, ha="left", va="center", fontsize=9.5, color=GREY)
    yg = rc - 0.62
    curve(ax, 0.35, yg, 0.65, 0.4, g1, INK, lw=1.1, recentre=True); ax.text(1.16, yg, op, ha="center", va="center", fontsize=11, color=INK)
    curve(ax, 1.32, yg, 0.65, 0.4, g2, INK, lw=1.1, recentre=True); ax.text(2.13, yg, op, ha="center", va="center", fontsize=11, color=INK)
    ax.text(2.45, yg, r"$\cdots$", ha="center", va="center", fontsize=11, color=GREY)

# ---- the four cells
cells = [
    ("add", "ell", "additive · lengthscale", r"$\sigma_f^2 \sum_i \tilde k_{\ell_i}$", r"unnormalised; $\rho_i \sim \mathrm{HC}(\tau)$", None, False),
    ("add", "amp", "additive · amplitude", r"$\sum_i a_i^2\, \bar k_i$", r"$\bar k_i = \tilde k_{\ell_i} / v(\ell_i)$; $a_i^2 \sim \mathrm{HC}(\tau)$", "the proposal", True),
    ("prod", "ell", "product · lengthscale", r"Matérn-5/2 ARD", r"$\rho_i \sim \mathrm{HC}(\tau)$", "SAASBO", False),
    ("prod", "amp", "product · amplitude", r"$\prod_i (1 + a_i^2\, \bar k_i)$", r"$a_i^2 \sim \mathrm{HC}(\tau)$", None, False),
]
for rk, ck, title, formula, note, tag, hi in cells:
    x, y = cols[ck]["x"], rows[rk]["y"]; c, rc = cols[ck]["c"], rows[rk]["c"]
    ax.add_patch(Rectangle((x, y), CW, RH, facecolor=ACC_BG if hi else "white", edgecolor=ACC if hi else LINE, lw=1.4 if hi else 0.8))
    ax.text(c, rc + 0.58, title, ha="center", va="center", fontsize=12, weight="bold", color=INK)
    ax.text(c, rc + 0.12, formula, ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(c, rc - 0.3, note, ha="center", va="center", fontsize=9.5, color=GREY)
    if tag:
        ax.text(c, rc - 0.68, tag, ha="center", va="center", fontsize=10, style="italic", color=ACC if hi else GREY)

fig.savefig("fig_2x2.png", dpi=300); fig.savefig("fig_2x2.pdf")
print("wrote fig_2x2.png and fig_2x2.pdf")
