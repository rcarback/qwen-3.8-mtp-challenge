"""Greedy-completion agreement of every arm against a control arm.

usage: agree.py DIR CONTROL ARM [ARM...]

DIR holds the out-ARM-PROMPT.txt files serve-sweep.sh writes. One line per
(arm, prompt): the first differing character and the identical prefix
fraction, or "identical".
"""
import os
import sys

DIR, ctl, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
prompts = sorted(
    f[len("out-%s-" % ctl):-4]
    for f in os.listdir(DIR)
    if f.startswith("out-%s-" % ctl) and f.endswith(".txt")
)
for arm in arms:
    for p in prompts:
        a = open(os.path.join(DIR, "out-%s-%s.txt" % (ctl, p))).read()
        b_path = os.path.join(DIR, "out-%s-%s.txt" % (arm, p))
        if not os.path.exists(b_path):
            print("%s\t%s\tmissing" % (arm, p))
            continue
        b = open(b_path).read()
        n = min(len(a), len(b))
        i = next((k for k in range(n) if a[k] != b[k]), None)
        first = "identical" if (i is None and len(a) == len(b)) else (i if i is not None else n)
        frac = (n if i is None else i) / max(1, len(a))
        print("%s\t%s\tfirst_diff_char=%s\tprefix_frac=%.3f\tlen=%d/%d" % (arm, p, first, frac, len(a), len(b)))
