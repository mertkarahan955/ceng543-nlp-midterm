#!/usr/bin/env python3
"""
Quick test script to verify Q3 setup and run a fast sanity check.
Tests both Seq2Seq and Transformer with minimal epochs.
"""

import subprocess
import sys

def run_test(model, emb_mode, exp_name):
    """Run a single quick test."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {model} + {emb_mode}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'train_q3_complete.py',
        '--model', model,
        '--emb_mode', emb_mode,
        '--epochs', '2',  # Just 2 epochs for quick test
        '--batch', '32',
        '--exp_name', exp_name
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"{exp_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{exp_name} failed with error code {e.returncode}")
        return False

def main():
    print("\n" + "="*60)
    print("Q3 SETUP VERIFICATION TEST")
    print("="*60)
    print("\nThis will run quick 2-epoch tests to verify your setup.")
    print("Full experiments will take much longer (12 epochs each).\n")
    
    tests = [
        ('seq2seq', 'learnable', 'test_seq2seq_learnable'),
        ('transformer', 'learnable', 'test_transformer_learnable'),
    ]
    
    results = []
    for model, emb_mode, exp_name in tests:
        success = run_test(model, emb_mode, exp_name)
        results.append((exp_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for exp_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{exp_name:<30} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run full experiments.")
        print("\nNext steps:")
        print("  1. Run all experiments: ./run_all_q3_experiments.sh")
        print("  2. Or run individual: python train_q3_complete.py --model [MODEL] --emb_mode [MODE]")
        return 0
    else:
        print("\nSome tests failed. Please check:")
        print("  - Dependencies installed: pip install -r requirements_q3.txt")
        print("  - CUDA available (if using GPU)")
        print("  - GloVe downloaded (if testing glove mode)")
        return 1

if __name__ == '__main__':
    sys.exit(main())