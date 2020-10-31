import matplotlib.pyplot as plt
from matplotlib import rc,rcParams 
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import math
import numpy as np

import json
import os
import re

from contactnets.utils import dirs

from collections import defaultdict

from typing import *

import pdb

def load_results(instance_regex: str) -> Dict:
    pattern = re.compile(instance_regex + '\Z')
    results = defaultdict(list) 
    for instance_name in os.listdir(dirs.results_path('cloud')):
        if pattern.match(instance_name):
            base_path = ['cloud', str(instance_name), 'sweep-no-tb']
            if 'sweep-no-tb' in os.listdir(dirs.results_path(*base_path)):
                base_path.append('sweep-no-tb')
            for run_num in os.listdir(dirs.results_path(*base_path)):
                stats_file = base_path + [str(run_num), 'best', 'stats.json']
                #print(stats_file)
                stats = json.load(open(dirs.results_path(*stats_file)))
                results[int(run_num)].append(stats)
    return results

def extract_xys(results, y_field):
    extracted = defaultdict(list)
    for i in results.keys():
        for result in results[i]:
            extracted[i].append(float(result[y_field] * val_scale))
    return extracted

def extract_points(results, y_field):
    extracted = extract_xys(results, y_field)
    xs, ys = [], []
    for x in extracted.keys():
        for y in extracted[x]:
            xs.append(x)
            ys.append(y)
    return xs, ys

def scatter_to_errorplot(extracted):
    confidence = 0.95
    def get_log_mean(v):
        v = np.clip(v, a_min=0.0001, a_max=None)
        mean = np.log(v).mean() 
        var = np.log(v).var() 

        return np.exp(mean + var/2)
    
    def get_log_bound(v, upper):
        v = np.clip(v, a_min=0.0001, a_max=None)
        n = len(v)
        v = np.log(v)
        
        interval = confidence * math.sqrt(v.var() / n + (v.var() ** 2) / (2 * (n-1)))
        bound = v.mean() + v.var() / 2 + (interval if upper else -interval)
        
        return np.exp(bound)
    
    if use_log:
        means = {k: get_log_mean(v) for k, v in extracted.items()}
        # means = {k: np.percentile(v, 50) for k, v in extracted.items()}
        lowers = {k: get_log_bound(v, False) for k, v in extracted.items()}
        uppers = {k: get_log_bound(v, True) for k, v in extracted.items()}
    else:
        means = {k: np.mean(v) for k, v in extracted.items()}
        lowers = {k: np.mean(v) - confidence * np.var(v) / math.sqrt(len(v)) for k, v in extracted.items()}
        uppers = {k: np.mean(v) + confidence * np.var(v) / math.sqrt(len(v)) for k, v in extracted.items()}

    xs = list(means.keys())
    ys, y_lowers, y_uppers = [], [], []

    for x in xs:
        ys.append(means[x] * yscale)
        y_lowers.append(lowers[x] * yscale)
        y_uppers.append(uppers[x] * yscale)
    
    xs, ys, y_lowers, y_uppers = zip(*sorted(zip(xs, ys, y_lowers, y_uppers)))
    
    return xs, ys, y_lowers, y_uppers

def percentile_vals(extracted, percentile):
    return {k: np.percentile(v, percentile) for k, v in extracted.items()}

fig = plt.figure()
ax = plt.gca()

# models = ['sweep-e2e-.+', 'sweep-vert-.+']
# label_lookup = {'sweep-e2e-.+': 'End-to-end', 'sweep-vert-.+': 'Vertex'}
# color_lookup = {'sweep-e2e-.+': '#95001a', 'sweep-vert-.+': '#01256e'}

# models = ['sweep-e2e-.+', 'sweep-vert-.+', 'sweep-deepvert-.+']
# label_lookup = {'sweep-e2e-.+': 'End-to-end', 'sweep-vert-.+': 'Vertex', 'sweep-deepvert-.+': 'Deepvert'}
# color_lookup = {'sweep-e2e-.+': '#95001a', 'sweep-vert-.+': '#01256e', 'sweep-deepvert-.+': '#4a0042'}

        #'lcp_parallel': '#95001a'
        #'traj_residual_inner': '#f2c100', 'traj_residual_outer': '#c35a00', 'force': '#4a0042'}

for model in models:
    results = load_results(model)

    if plot_points:
        xs, ys = extract_points(results, yfield)
        xs = [x / 2 for x in xs]
        plt.scatter(xs, ys, label=label_lookup[model], alpha=0.5)
    else:
        extracted = extract_xys(results, yfield)
        # for percentile in [50]:
        # linestyles = [':', '--', '-']
        # for percentile, linestyle in zip([10, 50, 90], linestyles):
            # percentile_vals_dict = percentile_vals(extracted, 100-percentile)
            # xs = list(percentile_vals_dict.keys())
            # #xs = list(filter(lambda x: x <= 160, xs))
            # xs.sort()
            # block_diam = 3.07
            # ys = [100 * percentile_vals_dict[x] / block_diam for x in xs]
            # # ys = [percentile_vals_dict[x] * 57 for x in xs]
            # xs = list(map(lambda x: int(x / 2), xs)) 
            # ax.plot(xs, ys, linestyle=linestyle,
                    # linewidth=5, color=color_lookup[model])
        xs, ys, y_lowers, y_uppers = scatter_to_errorplot(extracted)
        xs = [x / 2 for x in xs]
        ax.plot(xs, ys, label=label_lookup[model], linewidth=3, color=color_lookup[model])
        ax.fill_between(xs, y_lowers, y_uppers, alpha=0.3, color=color_lookup[model])

if use_log:
    ax.set_yscale('log')
    ax.set_xscale('log')

xs = [2 * 2**j for j in range(1, 8)]
ax.set_xlim(min(xs), max(xs))
# ax.set_xlim(8, 2048)
xs_rounded = [round(x, 1) for x in xs]
ax.set_xticks(xs_rounded)

ax.tick_params(axis='x', which='minor', bottom=False)
ax.xaxis.set_major_formatter(FormatStrFormatter("\\textbf{{%.0f}}"))

plt.xlabel('\\textbf{{Training tosses}}')
plt.ylabel(ylabel)
#plt.ylabel('\\textbf{{Trajectory rotational error ($^{\circ}$)}}')
#plt.ylabel('\\textbf{{Trajectory positional error (\% block width)}}', fontsize=34)
# plt.ylabel('\\textbf{{Trajectory penetartion error (\% block width)}}')

plt.tick_params(axis='y', which='minor')
# ax.yaxis.set_major_formatter(FormatStrFormatter("\\textbf{{%.1f}}"))
# ax.yaxis.set_minor_formatter(FormatStrFormatter("\\textbf{{%.1f}}"))
ax.yaxis.set_major_formatter(FormatStrFormatter("\\textbf{{%.0f}}"))
ax.yaxis.set_minor_formatter(FormatStrFormatter("\\textbf{{%.0f}}"))
yticks = ax.yaxis.get_minor_ticks()
for i, ytick in enumerate(yticks):
    if i % 2 == 1:
        ytick.label1.set_visible(False)

ax.yaxis.grid(True, which='both')

# ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
# ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

lines = ax.get_lines()
# legend1 = plt.legend([lines[i] for i in [5,4,3]], ["\\textbf{90\%}", "\\textbf{50\%}", "\\textbf{10\%}"], loc=2)
# legend2 = plt.legend([lines[i] for i in [5,2]], ['\\textbf{Structured}', '\\textbf{End-to-end}'], loc=3)
# ax.add_artist(legend1)
# ax.add_artist(legend2)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.legend(loc=1, prop=dict(weight='bold'))

fig.set_size_inches(13, 13)
fig.savefig('test.png', dpi=100)
fig.savefig('data.png', transparent=True, dpi=100)
#fig.savefig('test.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
# fig.savefig('zshift.png', transparent=True, dpi=100, bbox_inches='tight', pad_inches=0.2)
