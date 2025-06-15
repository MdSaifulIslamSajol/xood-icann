#!/bin/bash

# Output directory for aggregate results
mkdir -p results/aggregate_cifar10

# Loop for 10 runs
for i in {1..10}
do
  echo "Running iteration $i"
  python3 test_ood_robustness.py --seed $i
done

echo "All runs completed"
