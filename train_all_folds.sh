#!/bin/bash
python3 train.py 0 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 train.py 1 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 train.py 2 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 train.py 3 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 train.py 4 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 CrossValidationScore.py 0 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
python3 Predictions.py 0 32 efficientnet-b3 240 5 1e-5 25 3e-4 initial_test_b3
