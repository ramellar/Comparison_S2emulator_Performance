from plotting.base_plotter import PlotManager
import yaml

# 1. Load your config and data
with open('config_plots.yaml') as f:
    plot_cfg = yaml.safe_load(f)

plotter = PlotManager(args, "./output_plots")

# 2. Example: Comparing Triangle Sizes
if plot_cfg['triangle_comparison']['enabled']:
    datasets = []
    for s in plot_cfg['triangle_comparison']['samples']:
        # Load the specific triangle size file
        ev = load_parquet(f"data_tri_{s['tri']}.parquet") 
        datasets.append({'data': ev.pt / ev.gen_pt, 'label': s['label'], 'color': s['color']})
    
    # Plot Response Distribution
    plotter.plot_distributions(datasets, "Response $p_T^{cl}/p_T^{gen}$", 50, [0, 2], "Response_Comparison_Triangles")

# 3. Example: Calibration Comparison
if plot_cfg['calibration_comparison']['enabled']:
    # ... similar logic calling plotter.plot_performance_vs_x ...
