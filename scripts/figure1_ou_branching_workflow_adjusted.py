import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def rounded_box(ax, xy, width, height, text="", fc="white", ec="black",
                lw=1.0, fontsize=8.2, weight="normal", radius=0.018,
                ha="center", va="center", zorder=2):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + width / 2, y + height / 2, text,
            ha=ha, va=va, fontsize=fontsize,
            weight=weight, zorder=zorder + 1,
        )
    return box


def arrow(ax, start, end, lw=1.25, mutation_scale=11,
          connectionstyle="arc3,rad=0.0", zorder=4):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color="black",
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def draw_panel_frame(ax, x, y, w, h, lab, title):
    rounded_box(ax, (x, y), w, h, fc="white", ec="black", lw=1.05, radius=0.025)
    ax.text(x + 0.025, y + h - 0.045, lab,
            fontsize=13, fontweight="bold", ha="left", va="top")
    ax.text(x + 0.075, y + h - 0.047, title,
            fontsize=9.3, fontweight="bold", ha="left", va="top")


def mini_timeseries(ax, x0, y0, w, h):
    rng = np.random.default_rng(7)
    t = np.arange(8)

    series = {
        "WT": 0.73 + 0.04 * np.sin(t / 1.4) + rng.normal(0, 0.018, len(t)),
        "priA": 0.43 + 0.09 * np.sin(t / 1.2 + 0.5) + rng.normal(0, 0.025, len(t)),
        "recG": 0.38 + 0.07 * np.cos(t / 1.5) + rng.normal(0, 0.022, len(t)),
    }

    label_y_offsets = {"WT": 0.000, "priA": 0.001, "recG": -0.003}

    for lab, vals in series.items():
        xs = x0 + w * (t / (len(t) - 1))
        ys = y0 + h * vals
        ax.plot(xs, ys, lw=1.5, zorder=5)
        ax.scatter(xs, ys, s=11, zorder=6)
        ax.text(x0 + w + 0.020, ys[-1] + label_y_offsets[lab],
                lab, fontsize=7.2, va="center", ha="left")

    ax.plot([x0, x0], [y0, y0 + h], lw=0.75, color="black")
    ax.plot([x0, x0 + w], [y0, y0], lw=0.75, color="black")
    ax.text(x0 + w / 2, y0 - 0.060, "time", fontsize=7, ha="center")
    ax.text(x0 - 0.025, y0 + h / 2, "log mutation\nfrequency",
            fontsize=7, ha="center", va="center", rotation=90)


def mini_ou_curve(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 160)
    mu = 0.55
    y = mu + 0.22 * np.exp(-3.0 * t) * np.cos(9.0 * t)
    band = 0.07 + 0.015 * np.sin(2 * np.pi * t)

    xs = x0 + w * t
    ys = y0 + h * y

    ax.fill_between(xs, y0 + h * (y - band), y0 + h * (y + band), alpha=0.22, zorder=3)
    ax.plot(xs, ys, lw=1.7, zorder=4)
    ax.plot([x0, x0 + w], [y0 + h * mu, y0 + h * mu],
            lw=0.95, linestyle="--", color="black", zorder=3)

    ax.text(x0 + w + 0.015, y0 + h * mu, r"$\mu_i$", fontsize=8, va="center")
    ax.text(x0 + w / 2, y0 - 0.047, "latent evolutionary time", fontsize=7, ha="center")


def mini_hierarchy(ax, x0, y0, w, h):
    """Draw hierarchical Bayesian partial-pooling schematic with clean spacing."""

    r = 0.027

    top = (x0 + w * 0.50, y0 + h * 0.66)

    mids = [
        (x0 + w * 0.24, y0 + h * 0.30),
        (x0 + w * 0.50, y0 + h * 0.30),
        (x0 + w * 0.76, y0 + h * 0.30),
    ]

    labels = [r"$\mu_{WT}$", r"$\mu_{priA}$", r"$\mu_{recG}$"]

    # Helper: draw arrow from edge of parent circle to edge of child circle
    def circle_arrow(parent, child, r_parent=r, r_child=r,
                     start_scale=1.10, end_scale=1.25):
        px, py = parent
        cx, cy = child
    
        dx = cx - px
        dy = cy - py
        dist = (dx**2 + dy**2) ** 0.5
    
        ux = dx / dist
        uy = dy / dist
    
        start = (px + ux * r_parent * start_scale,
                 py + uy * r_parent * start_scale)
    
        end = (cx - ux * r_child * end_scale,
               cy - uy * r_child * end_scale)
    
        arrow(
            ax,
            start,
            end,
            lw=0.85,
            mutation_scale=8,
            zorder=4,
        )

    for j, (pt, lab) in enumerate(zip(mids, labels)):
        ax.add_patch(
            Circle(
                pt,
                radius=r,
                facecolor="white",
                edgecolor="black",
                lw=1.0,
                zorder=4,
            )
        )
    
        ax.text(
            pt[0],
            pt[1],
            lab,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=5,
        )
    
    if j == 1:
        arrow(
            ax,
            (top[0], top[1] - r * 0.95),
            (pt[0], pt[1] + r * 0.95),
            lw=0.85,
            mutation_scale=8,
            zorder=4,
        )
    else:
        circle_arrow(top, pt, start_scale=1.10, end_scale=1.25)
    # Hyper-mean node
    ax.add_patch(
        Circle(
            top,
            radius=r,
            facecolor="white",
            edgecolor="black",
            lw=1.0,
            zorder=4,
        )
    )
    ax.text(
        top[0],
        top[1],
        r"$\mu_{\mathrm{hyper}}$",
        fontsize=7.0,
        ha="center",
        va="center",
        zorder=5,
    )

    # Genotype-specific nodes
    for pt, lab in zip(mids, labels):
        ax.add_patch(
            Circle(
                pt,
                radius=r,
                facecolor="white",
                edgecolor="black",
                lw=1.0,
                zorder=4,
            )
        )
        ax.text(
            pt[0],
            pt[1],
            lab,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=5,
        )
        circle_arrow(top, pt)

    # Panel C explanatory text
    ax.text(
        x0 + w / 2,
        y0 + h * 0.001,
        "partial pooling across backgrounds",
        fontsize=7.0,
        ha="center",
    )


def mini_count_layer(ax, x0, y0, w, h):
    t = np.linspace(0, 1, 100)
    latent = 0.18 + 0.58 / (1 + np.exp(-8 * (t - 0.5)))
    xs = x0 + w * 0.40 * t
    ys = y0 + h * (0.14 + 0.42 * latent)

    ax.plot(xs, ys, lw=1.6, zorder=5)
    ax.text(x0 + w * 0.17, y0 + h * 0.53, r"$X_t$", fontsize=8, ha="center")
    ax.text(x0 + w * 0.53, y0 + h * 0.53,
            r"$p(t)=10^{X_t}$",
            fontsize=7.4, ha="center")

    arrow(ax, (x0 + w * 0.46, y0 + h * 0.45),
          (x0 + w * 0.58, y0 + h * 0.45), lw=1.1, mutation_scale=10)

    counts = np.array([2, 4, 6, 5, 9, 7])
    bar_x0 = x0 + w * 0.63
    bar_w = w * 0.040
    max_h = h * 0.42

    for i, c in enumerate(counts):
        bx = bar_x0 + i * w * 0.052
        bh = max_h * c / counts.max()
        ax.add_patch(Rectangle(
            (bx, y0 + h * 0.15), bar_w, bh,
            facecolor="white", edgecolor="black", lw=0.9, zorder=5
        ))

    ax.text(x0 + w * 0.80, y0 + h * 0.65,
            r"$\tilde{K}_t \sim \mathrm{Binomial}(N_{\mathrm{eff}},p(t))$",
            fontsize=7.6, ha="center")
    ax.text(x0 + w * 0.78, y0 + h * 0.005,
            "posterior predictive counts",
            fontsize=7.1, ha="center")


def mini_outputs(ax, x0, y0, w, h):
    labels = [
        "latent\ntrajectories",
        "posterior\npredictive checks",
        "PSIS-LOO\nbenchmarking",
    ]

    # Lower and slightly shrink the three boxes to avoid Panel E title overlap
    box_w = w * 0.275
    gap = w * 0.045
    base_y = y0 + h * 0.38
    box_h = h * 0.28

    for i, lab in enumerate(labels):
        bx = x0 + i * (box_w + gap)
    
        # Draw empty box first
        rounded_box(
            ax,
            (bx, base_y),
            box_w,
            box_h,
            "",
            fc="white",
            ec="black",
            lw=0.9,
            fontsize=6.7,
            radius=0.018,
            zorder=4,
        )
    
        # Add label near the top of the box
        ax.text(
            bx + box_w / 2,
            base_y + box_h * 0.72,
            lab,
            fontsize=6.7,
            ha="center",
            va="center",
            zorder=7,
        )
    
        # Add symbol underneath the label
        if i == 0:
            xx = np.linspace(bx + box_w * 0.22, bx + box_w * 0.78, 50)
            yy = base_y + box_h * 0.30 + box_h * 0.10 * np.sin(np.linspace(0, 2.4, 50))
            ax.plot(xx, yy, lw=1.0, zorder=6)
    
        elif i == 1:
            z = np.linspace(-2, 2, 60)
            d1 = np.exp(-z**2 / 0.9)
            d2 = np.exp(-(z - 0.2)**2 / 1.2)
            d1 /= d1.max()
            d2 /= d2.max()
    
            ax.plot(
                bx + box_w * (0.22 + 0.56 * (z + 2) / 4),
                base_y + box_h * (0.25 + 0.10 * d1),
                lw=0.95,
                zorder=6,
            )
            ax.plot(
                bx + box_w * (0.22 + 0.56 * (z + 2) / 4),
                base_y + box_h * (0.25 + 0.10 * d2),
                lw=0.95,
                linestyle="--",
                zorder=6,
            )
    
        else:
            heights = [0.10, 0.17, 0.26]
            for j, ht in enumerate(heights):
                ax.add_patch(
                    Rectangle(
                        (
                            bx + box_w * (0.34 + 0.12 * j),
                            base_y + box_h * 0.20,
                        ),
                        box_w * 0.075,
                        box_h * ht,
                        facecolor="white",
                        edgecolor="black",
                        lw=0.8,
                        zorder=6,
                    )
                )

    # Move bottom text to a cleaner middle-lower position
    ax.text(
        x0 + w / 2,
        y0 + h * 0.200,
        "reproducible inference outputs",
        fontsize=8.0,
        ha="center",
        fontweight="bold",
    )

    ax.text(
        x0 + w / 2,
        y0 + h * 0.100,
        "parameters • uncertainty • predictions • model comparison",
        fontsize=7.3,
        ha="center",
    )


def make_figure(output_prefix="Figure1_OU_Branching_workflow_adjusted"):
    fig = plt.figure(figsize=(11.8, 7.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965,
        "OU–Branching Bayesian inference workflow",
        ha="center", va="top",
        fontsize=15, fontweight="bold"
    )
    ax.text(
        0.5, 0.932,
        "Coupling continuous stochastic evolutionary dynamics with discrete mutation/count observations",
        ha="center", va="top",
        fontsize=9.5
    )

    # Wider spacing and slightly shorter panels to avoid all label/title overlaps.
    panels = {
        "A": (0.050, 0.590, 0.275, 0.275),
        "B": (0.365, 0.590, 0.275, 0.275),
        "C": (0.680, 0.590, 0.275, 0.275),
        "D": (0.135, 0.190, 0.330, 0.265),
        "E": (0.540, 0.190, 0.350, 0.265),
    }

    draw_panel_frame(ax, *panels["A"], "A", "Input longitudinal data")
    draw_panel_frame(ax, *panels["B"], "B", "Latent OU dynamics")
    draw_panel_frame(ax, *panels["C"], "C", "Hierarchical Bayesian pooling")
    draw_panel_frame(ax, *panels["D"], "D", "Branching/count observation layer")
    draw_panel_frame(ax, *panels["E"], "E", "Validation and reusable outputs")

    # Panel-specific drawings, shifted downward where needed to avoid titles.
    x, y, w, h = panels["A"]
    mini_timeseries(ax, x + 0.065, y + 0.080, w * 0.58, h * 0.50)
    ax.text(x + w / 2, y + 0.045, "genotype × replicate × time", fontsize=7.6, ha="center")

    x, y, w, h = panels["B"]
    ax.text(x + w / 2, y + h - 0.090,
            r"$dX_t=-\theta(X_t-\mu_i)dt+\sigma_i dW_t$",
            fontsize=8.1, ha="center")
    mini_ou_curve(ax, x + 0.060, y + 0.070, w * 0.66, h * 0.46)
    ax.text(x + w / 2, y + 0.055,
            r"estimate $\mu_i$, $\theta$, $\sigma_i$, and latent $X_t$",
            fontsize=7.5, ha="center")

    x, y, w, h = panels["C"]
    mini_hierarchy(ax, x + 0.042, y + 0.060, w * 0.78, h * 0.65)
    ax.text(x + w / 2, y + 0.025,
            "stabilized inference for sparse longitudinal data",
            fontsize=7.5, ha="center")

    x, y, w, h = panels["D"]
    mini_count_layer(ax, x + 0.050, y + 0.060, w * 0.84, h * 0.68)

    x, y, w, h = panels["E"]
    mini_outputs(ax, x + 0.055, y + 0.040, w * 0.84, h * 0.78)

    # Simple, non-overlapping arrows.
    arrow(ax, (0.325, 0.730), (0.365, 0.730), lw=1.4)
    arrow(ax, (0.640, 0.730), (0.680, 0.730), lw=1.4)

    # Replace the long central bridge with a compact, lower bridge box.
    rounded_box(
        ax, (0.305, 0.500), 0.390, 0.045,
        "Posterior sampling links latent dynamics to observable mutation processes",
        fc="white", ec="black", lw=0.85, fontsize=7.9, radius=0.016, zorder=3
    )

    # Downward connection from OU panel to bridge.
    arrow(ax, (0.503, 0.590), (0.503, 0.548), lw=1.0, mutation_scale=9)

    # Clean split from bridge to panels D and E; curves routed above the lower panels.
    arrow(ax, (0.430, 0.500), (0.250, 0.465), lw=1.0, mutation_scale=9,
          connectionstyle="arc3,rad=0.10")
    arrow(ax, (0.575, 0.500), (0.735, 0.465), lw=1.0, mutation_scale=9,
          connectionstyle="arc3,rad=-0.10")

    # Main pipeline arrow from D to E.
    arrow(ax, (0.465, 0.333), (0.540, 0.333), lw=1.4)

    ax.text(
        0.5, 0.135,
        "Longitudinal bacterial mutation-frequency data provide a controlled validation and demonstration platform for the OU–Branching inference framework",
        ha="center", va="center", fontsize=8.8
    )

    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=900, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make_figure()
