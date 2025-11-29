#!/usr/bin/env python3
# collect_results.py
import re, json, csv, os, sys
attns = ["bahdanau","luong","scaleddot"]
out = []
for a in attns:
    # try logs/<attn>/metrics.csv last row for BLEU or run_logs/eval_<attn>.log final line
    metrics_csv = os.path.join("logs", a, "metrics.csv")
    bleu = None; rouge = None; ppl = None
    if os.path.exists(metrics_csv):
        # metrics.csv contains epoch,train_loss,valid_bleu — use best row
        with open(metrics_csv) as fh:
            rows = [r for r in csv.reader(fh)]
            if len(rows)>1:
                # find max bleu row
                header = rows[0]; data = rows[1:]
                try:
                    best = max(data, key=lambda r: float(r[2]))
                    bleu = float(best[2])
                except:
                    pass
    # fallback: parse run_logs/eval_<attn>.log
    eval_log = os.path.join("run_logs", f"eval_{a}.log")
    if os.path.exists(eval_log):
        with open(eval_log) as fh:
            txt = fh.read()
        m = re.search(r"BLEU:\s*([\d\.]+).*ROUGE-L:\s*([\d\.]+).*Perplexity:\s*([\d\.]+)", txt)
        if m:
            bleu = float(m.group(1)); rouge = float(m.group(2)); ppl = float(m.group(3))
        else:
            # try other common patternerd\s*\|\s*Perplexity:\s*([\d\.]+)", txt)
            if m2:
                bleu = float(m2.group(1)); rouge = float(m2.group(2)); ppl = float(m2.group(3))
    # As last resort try eval_metrics output file (results.json etc.)
    out.append({"attn":a,"bleu":bleu,"rouge":rouge,"ppl":ppl})
# write csv
with open("results_summary.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["attn","bleu","rouge","ppl"])
    for r in out: w.writerow([r["attn"], r["bleu"], r["rouge"], r["ppl"]])
print("Wrote results_summary.csv")
