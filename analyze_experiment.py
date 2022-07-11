from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_result

results_path = Path('OLD/results')

results = []
for p in results_path.iterdir():
    r = re.search('l2_g_(\d+)_p_(\d+)\.log', p.name)
    if r:
        gen = int(r.group(1))
        pop_size = int(r.group(2))
        with open(p) as f:
            try:
                results.append(get_result(pop_size, gen, f.read()))
            except TypeError:
                pass

gen = [d['gen'] for d in results]
pop_size = [d['pop_size'] for d in results]
df = pd.DataFrame(results, index=[pop_size, gen])
# del (df['gen'])
# del (df['pop_size'])
df.to_csv(results_path / 'experiment.csv')

delta_df = df[['delta_queries', 'delta_asr']]
better_delta = delta_df[(delta_df['delta_queries'] < 0) & (delta_df['delta_asr'] > 0)]
better_df = df[df.index.isin(better_delta.index)]
better_df.to_csv(results_path / 'experiment_better.csv')

# delta_df.plot()
# plt.show()
print(df)
print(df.describe())

print(f'\nevo is better at {len(better_df)} out of {len(df)} samples:\n')
print(better_df[['sq_asr', 'evo_asr']])
print(better_df[['sq_queries', 'evo_queries']])
