"""Plot benchmark entanglement cost and relative improvement."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path,
                    default=Path(__file__).resolve().parents[1]/'results/fig6')
path = parser.parse_args().results_dir
comparison = pd.read_csv(path/'baseline_comparison.csv')
summary = pd.read_csv(path/'summary.csv')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linestyle': ':',
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
benchmarks = ['Telegate-SA', 'Telegate-RD', 'Teledata-ZS', 'GC', 'GC+LABUBU*']
colors = ['#009ADE', '#FF1F5B', '#FFC61E', '#8ECFC9', '#BE88DC']
markers = ['o', '^', 's', 'x', 'd']
for row, typ in enumerate(['RD', 'QFT']):
    for col, capacity in enumerate([5, 15, 25]):
        ax = axes[row, col]
        cost_axis = ax.twinx()
        cost_axis.grid(False)
        data = summary[(summary.type==typ)&(summary.capacity==capacity)].sort_values('width')
        cost_axis.bar(data.width-2.8, data.min_cost, width=4, color='#B3876A',
                      alpha=.4, edgecolor='black', label='LABUBU')
        cost_axis.set_ylabel('#Entanglement', color='#865C3F')
        cost_axis.set_ylim(0, data.min_cost.max()*1.15)
        for i, name in enumerate(benchmarks):
            values = comparison[(comparison.type==typ)&(comparison.capacity==capacity)&
                                (comparison.benchmark==name)].sort_values('width')
            ax.plot(values.width, values.gain_pct, label='vs '+name, color=colors[i],
                    marker=markers[i], linewidth=1.6, markersize=4)
        limits = (-20,80) if typ=='RD' else [(0,100),(-25,100),(-200,100)][col]
        ax.set(title=f'({chr(97+row*3+col)}) {typ}, capacity {capacity}',
               xlabel='Circuit Width', ylabel='Relative Improvement (%)',
               xticks=list(range(30,101,10)), ylim=limits)
        ax.patch.set_visible(False)
        ax.set_zorder(cost_axis.get_zorder()+1)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.5,.95))
fig.suptitle('LABUBU: entanglement cost and relative improvement', fontsize=17, y=.99)
counts = sorted(summary.runs.unique())
selection = f'best of {int(counts[0])} runs' if len(counts)==1 else 'minimum per point'
fig.text(.5,.012,f'Bars: LABUBU entanglement cost ({selection}). Lines: improvement relative to the baseline cost table.',
         ha='center', fontsize=10)
fig.tight_layout(rect=[0,.035,1,.9])
for extension in ['png','pdf','svg']:
    fig.savefig(path/f'fig06.{extension}', dpi=300, bbox_inches='tight')
plt.close(fig)
