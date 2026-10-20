# Hyperparameter Tuning Analysis

## 1. Baseline Model

Model: DecisionTreeClassifier

Reference CV F1 Macro: 0.9328

Reference Test Accuracy: 0.9333

## 2. Grid Search

Model: RandomForestClassifier

Total combinations: 72

Cross-validation: 5-fold

Total fits: 360

Reference Best CV F1 Macro: 0.9583

Reference Test Accuracy: 0.9667

## 3. Random Search

Model: RandomForestClassifier

Number of iterations: 30

Cross-validation: 5-fold

Total fits: 150

Reference Best CV F1 Macro: 0.9581

Reference Test Accuracy: 0.9667

## 4. Comparison

Grid Search evaluates every combination in the specified
hyperparameter grid.

Random Search evaluates a selected number of combinations
from the search space.

In this experiment, Grid Search required 360 model fits,
while Random Search required 150 model fits.

Both tuning approaches produced higher CV F1 Macro scores
than the baseline model.

Note: The scores above are reference results. Replace them
with your actual results if they differ.
