import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.path as mpath

def main():
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    # Perfect filled crescent path for sheep
    alpha = np.arccos(0.61)
    beta = np.arctan2(0.79246, 0.11)
    theta_outer = np.linspace(alpha, 2*np.pi - alpha, 40)
    outer_arc = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    theta_inner = np.linspace(2*np.pi - beta, beta, 40)
    inner_arc = np.column_stack([np.cos(theta_inner)*0.8 + 0.5, np.sin(theta_inner)*0.8])
    verts = np.vstack([outer_arc, inner_arc, outer_arc[0:1]])
    codes = np.full(len(verts), mpath.Path.LINETO)
    codes[0] = mpath.Path.MOVETO
    verts -= np.array([0.15, 0.0])  # slightly center it
    crescent_path = mpath.Path(verts, codes)

    # Settings
    N_POINTS = 350
    np.random.seed(42)
    
    concepts = ['cat', 'sheep', 'dog']
    palette = {
        'cat': '#F9E79F',    # lighter pastel yellow
        'sheep': '#A9DFBF',  # lighter pastel green
        'dog': '#C39BD3',    # light pastel purple
    }
    
    markers = {
        'cat': '*',
        'sheep': crescent_path,  # Solid filled crescent moon
        'dog': 'X'           # Filled cross (X)
    }

    # Artificial means (Before)
    # Cat and Dog overlap LESS now (shifted them apart by ~0.3 instead of 0.05)
    means_before = {
        'cat': [0.1, -0.2],
        'dog': [0.35, 0.15],
        'sheep': [-0.2, 0.6]  # further up to stay away from the stretched dog/cat
    }

    # Artificial means (After)
    means_after = {
        'cat': [-2.0, -1.8],
        'sheep': [2.5, 0.0],
        'dog': [-0.2, 2.5]
    }
    
    # Covariance matrices
    cov_cat_dog = [[0.06, 0.03], [0.03, 0.06]] # tighter so they don't smear as much into each other
    cov_sheep = [[0.06, -0.02], [-0.02, 0.06]]

    # Generate data
    data_before = {}
    data_after = {}
    
    for c in concepts:
        cov = cov_cat_dog if c in ['cat', 'dog'] else cov_sheep
        data_before[c] = np.random.multivariate_normal(means_before[c], cov, N_POINTS)
        data_after[c] = np.random.multivariate_normal(means_after[c], cov, N_POINTS)

    
    POINT_SIZE = 250
    margin = 0.2
    FONT_SIZE_LABEL = 32

    def draw_figure(data_points, save_path, is_after=False):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        all_pts = []
        if is_after:
            all_pts.extend(list(data_before.values()))
            all_pts.extend(list(data_after.values()))
        else:
            all_pts.extend(list(data_points.values()))

        all_pts = np.vstack(all_pts)
        px_min, px_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        py_min, py_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        px_range = max(px_max - px_min, 1e-6)
        py_range = max(py_max - py_min, 1e-6)

        for c in concepts:
            color = palette[c]
            pts = data_points[c]
            marker = markers[c]

            # Scatter points
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=POINT_SIZE, c=color,
                edgecolors='black', linewidths=0.5,
                alpha=0.85, zorder=4,
                marker=marker,
            )

            # Centers
            cx, cy = pts.mean(axis=0)
            
            if is_after:
                # Calculate before center for arrow
                cx_before, cy_before = data_before[c].mean(axis=0)
                
                # Draw arrow from before center to after center
                # using FancyArrowPatch for a curved or straight clean arrow
                import matplotlib.patches as patches
                style = "Simple, tail_width=2.5, head_width=12, head_length=15"
                kw = dict(arrowstyle=style, color='black', alpha=0.3, zorder=2)
                
                a = patches.FancyArrowPatch((cx_before, cy_before), (cx, cy), **kw)
                ax.add_patch(a)
                
                # Plot the faded "before" center for reference
                ax.scatter(cx_before, cy_before, s=300, c='none', edgecolors='black',
                           linewidths=1.5, zorder=3, alpha=0.3, marker='o')

            ax.scatter(cx, cy, s=700, c='none', edgecolors='black',
                       linewidths=2.5, zorder=6, marker='o')
            ax.scatter(cx, cy, s=400, c=color, edgecolors='black',
                       linewidths=1.5, zorder=7, marker='o')
            
            ax.annotate(
                c, (cx, cy),
                textcoords="offset points", xytext=(18, 14),
                fontsize=FONT_SIZE_LABEL, fontweight='bold', fontstyle='italic',
                color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground='black')],
                zorder=8,
            )

        ax.set_xlim(px_min - margin * px_range, px_max + margin * px_range)
        ax.set_ylim(py_min - margin * py_range, py_max + margin * py_range)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(1.5)

        plt.tight_layout()

        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {save_path}.png")
        plt.savefig(save_path + ".pdf", bbox_inches='tight', facecolor='white')
        print(f"Saved {save_path}.pdf")
        plt.close()

    os.makedirs('scripts/outputs', exist_ok=True)
    draw_figure(data_before, 'scripts/outputs/synthetic_before_omp', is_after=False)
    draw_figure(data_after, 'scripts/outputs/synthetic_after_omp', is_after=True)

if __name__ == "__main__":
    main()
