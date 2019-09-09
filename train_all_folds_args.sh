#!/bin/bash
for FOLD in 0 1 2 3 4
do
	python3 train_1cycle.py $FOLD $1 $2 $3 $4 $5 $6 $7 $8
done
python3 CrossValidationScore.py 0 $1 $2 $3 $4 $5 $6 $7 $8
python3 Predictions.py 0 $1 $2 $3 $4 $5 $6 $7 $8
