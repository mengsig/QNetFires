import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cmap = plt.get_cmap('hot')
cmap.set_bad("purple")
def save_matrix_as_heatmap(
        matrix, colors, units, title, filename,
        vmin=None, vmax=None, ticks=None, norm=False):
    import numpy as np
    # mask out invalid entries so that cmap.set_bad() actually catches them
    mtx = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots()
    if colors == "hot":
        if norm:
            im = ax.imshow(
                mtx,
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                interpolation='nearest',   # <— no smoothing
                aspect='equal'            # <— square cells
            )
        else:
            im = ax.imshow(
                mtx,
                origin="lower",
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation='nearest',
                aspect='equal'
            )
    else:
        im = ax.imshow(
            mtx,
            origin="lower",
            cmap=colors,
            vmin=vmin, vmax=vmax,
            interpolation='nearest',
            aspect='equal'
        )

    cbar = plt.colorbar(im, orientation="vertical", ticks=ticks)
    cbar.set_label(units)
    ax.set_title(title)
    ax.axis("off")

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def save_matrix_as_contours(
    matrix,
    title,
    filename,
    levels=None,
    extend: str = None
):
    """
    Save a contour plot of `matrix` to `filename`.  
    If `levels` is None, automatically bin the data using NumPy's
    'auto' strategy (Freedman–Diaconis / Sturges) and use the midpoints
    of those bins as contour levels.
    """
    # 1) mask invalids
    import numpy as np
    mat = np.array(matrix, copy=False, dtype=np.float64)
    mat = np.ma.masked_invalid(mat)
    data = mat.compressed()

    # 2) auto‐compute levels if needed
    levels = 10
    if levels is None:
        if data.size == 0:
            # no data → a single zero‐level
            levels = [0]
        else:
            # compute histogram bin‐edges with the 'auto' rule
            edges = np.histogram_bin_edges(data, bins='auto')
            if len(edges) > 1:
                # take midpoints of each bin as our contour values
                levels = 0.5 * (edges[:-1] + edges[1:])
            else:
                levels = edges  # degenerate case

    # 3) plot
    fig, ax = plt.subplots()
    cs = ax.contour(mat, levels=levels, extend=extend)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

