"""Compare two harness result files: volume (relative), bounding box (absolute mm), validity."""
import json, sys

ref = {(r["model"], r["function"]): r for r in json.load(open(sys.argv[1]))}
new = {(r["model"], r["function"]): r for r in json.load(open(sys.argv[2]))}
VOL_TOL, BOX_TOL = 1e-4, 1e-3
rows, counts = [], {"match": 0, "differs": 0, "error": 0, "skipped": 0}
for key in sorted(ref):
    a, b = ref[key], new.get(key, {"status": "missing"})
    name = key[1] or key[0]
    if a["status"] == "skipped":
        counts["skipped"] += 1
        continue
    if b["status"] != "ok" or a["status"] != "ok":
        counts["error"] += 1
        rows.append((name, "ERROR", b.get("error", b["status"])[:90]))
        continue
    dv = (b["volume"] - a["volume"]) / a["volume"]
    db = max(abs(x - y) for p, q in zip(a["bbox"], b["bbox"]) for x, y in zip(p, q))
    ok = abs(dv) <= VOL_TOL and db <= BOX_TOL and b.get("valid") is not False
    counts["match" if ok else "differs"] += 1
    rows.append((name, "match" if ok else "DIFFERS", f"vol {dv:+.1e}  bbox {db:.1e} mm  valid={b.get('valid')}  ({a['volume']:.1f} vs {b['volume']:.1f})"))
for r in sorted(rows, key=lambda r: r[1] != "DIFFERS"):
    print(f"{r[1]:8} {r[0]:36} {r[2]}")
print(counts)
