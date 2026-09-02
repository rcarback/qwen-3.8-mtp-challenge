#!/usr/bin/env python3
"""Reduce an mtp-accept CSV to a per-config, per-depth median table.

Usage: mtp-accept-summary.py CSV            # prints the table
       mtp-accept-summary.py --self-test    # runs the built-in check
"""
import csv
import statistics
import sys
from collections import defaultdict

COLS = ("accepted_draft_rate", "effective_mean_draft_len",
        "serial_seconds_per_token", "mtp_seconds_per_token", "mtp_decode_speedup")


def summarize(rows):
    """rows: iterable of dicts with config, depth and the COLS. Returns
    {(config, depth): {col: median, "n": count, "matched": all_matched}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], int(r["depth"]))].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        entry = {c: statistics.median(float(r[c]) for r in rs) for c in COLS}
        entry["n"] = len(rs)
        entry["matched"] = all(r["all_tokens_matched"].lower() == "true" for r in rs)
        out[key] = entry
    return out


def render(summary):
    head = f"{'config':10} {'depth':>5} {'n':>3} {'accept':>7} {'draft':>6} " \
           f"{'serial s/tok':>13} {'mtp s/tok':>10} {'speedup':>8} {'matched':>8}"
    lines = [head, "-" * len(head)]
    for (cfg, depth), e in summary.items():
        lines.append(
            f"{cfg:10} {depth:>5} {e['n']:>3} {e['accepted_draft_rate']:>7.3f} "
            f"{e['effective_mean_draft_len']:>6.2f} {e['serial_seconds_per_token']:>13.4f} "
            f"{e['mtp_seconds_per_token']:>10.4f} {e['mtp_decode_speedup']:>8.3f} "
            f"{str(e['matched']):>8}")
    return "\n".join(lines)


def self_test():
    rows = [
        dict(config="gpu", prompt="a", depth="8", accepted_draft_rate="0.5",
             effective_mean_draft_len="2.0", serial_seconds_per_token="0.10",
             mtp_seconds_per_token="0.08", mtp_decode_speedup="1.25", all_tokens_matched="true"),
        dict(config="gpu", prompt="b", depth="8", accepted_draft_rate="0.7",
             effective_mean_draft_len="3.0", serial_seconds_per_token="0.10",
             mtp_seconds_per_token="0.06", mtp_decode_speedup="1.67", all_tokens_matched="true"),
        dict(config="gpu", prompt="c", depth="8", accepted_draft_rate="0.6",
             effective_mean_draft_len="2.5", serial_seconds_per_token="0.10",
             mtp_seconds_per_token="0.07", mtp_decode_speedup="1.43", all_tokens_matched="false"),
    ]
    s = summarize(rows)
    e = s[("gpu", 8)]
    assert e["n"] == 3, e
    assert abs(e["accepted_draft_rate"] - 0.6) < 1e-9, e
    assert abs(e["effective_mean_draft_len"] - 2.5) < 1e-9, e
    assert e["matched"] is False, e
    print("self-test ok")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], newline="") as f:
        print(render(summarize(csv.DictReader(f))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
