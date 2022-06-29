from pathlib import Path
import re
import pandas as pd
from sorcery import dict_of

results = []


def re_get(pattern, line, current):
    r = re.search(pattern, line)
    if r:
        return int(float(r.group(1)))
    else:
        return current


for p in Path('results').iterdir():
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
        results.append(dict_of(gen, pop_size,sq_asr, sq_queries, evo_asr, evo_queries))

df = pd.DataFrame(results)
print(df)
