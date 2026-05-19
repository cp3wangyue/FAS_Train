"""Quick smoke test: run train.py for 1 epoch with tiny config"""
import os, sys, torch

# Monkey-patch Config to run just 1 epoch with small batch
import train
train.Config.num_epochs = 1
train.Config.batch_size = 8
train.Config.num_workers = 0

# Run
train.main()
print("\n[SMOKE TEST PASSED] train.py runs 1 epoch OK")
