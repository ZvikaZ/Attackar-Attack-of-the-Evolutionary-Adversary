from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from sorcery import dict_of

results_path = Path('results_2')


def re_get(pattern, line, current):
    r = re.search(pattern, line)
    if r:
        return int(float(r.group(1)))
    else:
        return current


results = []

for p in results_path.iterdir():
    r = re.search('l2_g_(\d+)_p_(\d+)\.log', p.name)
    if r:
        gen = int(r.group(1))
        pop_size = int(r.group(2))
        sq_asr = None
        sq_queries = None
        evo_asr = None
        evo_queries = None
        with open(p) as f:
            for line in f:
                # print(line, end='')
                sq_asr = re_get('Square - attack success rate: (.+)%', line, sq_asr)
                sq_queries = re_get('Square - queries \(median\): (.+)', line, sq_queries)
                evo_asr = re_get('Evo - attack success rate: (.+)%', line, evo_asr)
                evo_queries = re_get('Evo - queries \(median\): (.+)', line, evo_queries)
        try:
            delta_asr = evo_asr - sq_asr
            delta_queries = evo_queries - sq_queries
            results.append(dict_of(pop_size, gen, sq_asr, sq_queries, evo_asr, evo_queries, delta_asr, delta_queries))
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
