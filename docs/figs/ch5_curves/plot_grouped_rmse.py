"""Render review-only curves from audited pooled RMSE; no manuscript edits."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cmgan-matplotlib')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/research/2026-10-09_seed42_result_audit.json'
OUT = Path(__file__).resolve().parent
DATA = json.loads(SOURCE.read_text())['grouped']
SCENARIOS = [
    ('**仿真 RIR** (test1+3)', '(a) Simulated RIRs', 'Test 1 + Test 3'),
    ('**真实 RIR** (test2+4)', '(b) Real RIRs', 'Test 2 + Test 4'),
    ('**全部** (test1–4)', '(c) All RIRs', 'Test 1–4'),
]
STYLES = {
    'BERP': ('BERP', '#557DA5', 's', '--'),
    'DAMTL': ('DAMTL', '#8E74A6', '^', '-.'),
    'noiseaware': ('noiseaware', '#38958C', 'D', ':'),
    '本文': ('Proposed', '#D56838', 'o', '-'),
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.edgecolor': '#525960', 'axes.linewidth': .8,
    'xtick.color': '#343B42', 'ytick.color': '#343B42',
    'savefig.facecolor': 'white',
})

def draw(kind, labels, xlabel, ylim, step):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.55), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.77, wspace=.14)
    for ax, (key, title, subtitle) in zip(axes, SCENARIOS):
        for method, (label, color, marker, linestyle) in STYLES.items():
            values = DATA[kind + key][method][:-1]  # exclude pooled overall RMSE
            assert len(values) == len(labels)
            ax.plot(range(len(labels)), values, label=label, color=color,
                    marker=marker, linestyle=linestyle,
                    linewidth=2.5 if method == '本文' else 1.65,
                    markersize=7 if method == '本文' else 6,
                    markerfacecolor=color if method == '本文' else 'white',
                    markeredgewidth=1.3, zorder=4 if method == '本文' else 3)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlim(-.18, len(labels)-.82)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.grid(axis='y', color='#E0E4E8', linewidth=.75)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3.5)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_title(title, pad=28, fontweight='semibold')
        ax.text(.5, 1.035, subtitle, transform=ax.transAxes, ha='center',
                color='#65717C', fontsize=10)
    axes[0].set_ylabel('RMSE (ms)', labelpad=9)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(.52, .995),
               ncol=4, frameon=False, handlelength=2.8, columnspacing=2.6)
    file=OUT / f'{kind}_rmse_curves_v1.png'
    fig.savefig(file, dpi=300)
    plt.close(fig)
    print(file)

draw('snr', ['−5', '0', '5', '10', '≥15'], 'SNR group (dB)', (0, 275), 50)
draw('t60', ['[0.1, 0.4)', '[0.4, 0.7)', '[0.7, 1.0)', '[1.0, 1.5]'],
     r'$T_{60}$ interval (s)', (0, 330), 50)
