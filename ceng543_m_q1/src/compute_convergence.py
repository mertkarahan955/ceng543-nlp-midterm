# src/compute_convergence.py
import argparse, os, numpy as np, pandas as pd
from math import isfinite
from numpy import trapz

def analyze_log(csv_path):
    df = pd.read_csv(csv_path)
    # required columns: epoch,train_loss,train_acc,val_loss,val_acc,val_macroF1,epoch_time_s
    avg_epoch_time = df['epoch_time_s'].mean()
    best_idx = df['val_acc'].idxmax()
    best_epoch = int(df.loc[best_idx,'epoch'])
    best_val_acc = float(df.loc[best_idx,'val_acc'])
    time_to_best = df.loc[:best_idx,'epoch_time_s'].sum()
    # AUC of validation loss (trapezoid)
    x = df['epoch'].values
    y = df['val_loss'].values
    val_loss_auc = trapz(y, x)
    # epochs to reach threshold (95% of best)
    thresh = 0.95 * best_val_acc
    reached = df[df['val_acc'] >= thresh]
    epochs_to_thresh = int(reached['epoch'].iloc[0]) if not reached.empty else None
    time_to_thresh = None
    if epochs_to_thresh is not None:
        time_to_thresh = float(df.loc[df['epoch'] <= epochs_to_thresh, 'epoch_time_s'].sum())

    return {
        'avg_epoch_time_s': float(avg_epoch_time),
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'time_to_best_s': float(time_to_best),
        'val_loss_auc': float(val_loss_auc),
        'epochs_to_95pct_of_best': epochs_to_thresh,
        'time_to_95pct_s': time_to_thresh
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True, help='Paths to train_log.csv files')
    args = parser.parse_args()
    out = {}
    for p in args.logs:
        if not os.path.exists(p):
            print("Not found:", p); continue
        name = os.path.basename(os.path.dirname(p))  # use parent folder name as key
        out[name] = analyze_log(p)
    import json
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
