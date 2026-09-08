#!/usr/bin/env python3
"""Tabulate measure.sh results: per prompt, per arm serial and MTP s/token, local ratio, and the
ranked-style estimate (base serial over each arm's MTP)."""
import json, sys, glob, os, statistics as st
res = sys.argv[1]
rows = {}
for f in glob.glob(os.path.join(res, 'score-*.json')):
    name, prompt = os.path.basename(f)[6:-5].split('-', 1)
    s = json.load(open(f)); m = s.get('metrics', {})
    rows.setdefault(prompt, {})[name] = dict(score=s.get('score'), serial=m.get('serial_seconds_per_token'), mtp=m.get('mtp_seconds_per_token'), eff=m.get('effective_mean_draft_len'), acc=m.get('accepted_draft_rate'))
arms = sorted({a for p in rows.values() for a in p})
print('| prompt | ' + ' | '.join(f'{a} serial s/tok | {a} MTP s/tok | {a} local ratio' for a in arms) + ' | ' + ' | '.join(f'ranked-style {a} (base serial / {a} MTP)' for a in arms if a != 'base') + ' |')
print('|' + ' --- |' * (1 + 3*len(arms) + len([a for a in arms if a != 'base'])))
est = {a: [] for a in arms}; loc = {a: [] for a in arms}
for prompt in sorted(rows):
    r = rows[prompt]; cells = [prompt]
    for a in arms:
        d = r.get(a)
        if d and d['serial'] and d['mtp']:
            cells += [f"{d['serial']:.4f}", f"{d['mtp']:.4f}", f"{d['score']:.3f}"]; loc[a].append(d['score'])
        else: cells += ['', '', '']
    base = r.get('base')
    for a in arms:
        if a == 'base': continue
        d = r.get(a)
        if base and d and base['serial'] and d['mtp']:
            v = base['serial'] / d['mtp']; est[a].append(v); cells.append(f'{v:.3f}')
        else: cells.append('')
    print('| ' + ' | '.join(cells) + ' |')
print()
for a in arms:
    if loc[a]: print(f'{a}: local ratio median {st.median(loc[a]):.3f} over {len(loc[a])} prompts', end='')
    if a != 'base' and est[a]: print(f'; ranked-style median {st.median(est[a]):.3f}', end='')
    print()
