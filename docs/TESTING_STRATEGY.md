# Testing Strategy – Experiment 8

## 1. Objective
To test data preprocessing, feature engineering, data validation, model training, and model quality using pytest.

## 2. Test Files

| Test File | Purpose |
|---|---|
| test_preprocess.py | Tests duplicate removal, missing values, and empty data |
| test_features.py | Tests feature calculations and edge cases |
| test_validate.py | Tests missing columns, invalid species, and out-of-range values |
| test_training_pipeline.py | Tests model training and prediction outputs |
| test_model_quality.py | Tests prediction validity and minimum model accuracy |

## 3. Test Execution
Command: `pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml`

## 4. Test Results
All 16 automated tests passed successfully.

## 5. Coverage and Reports
Pytest-Cov measures source-code coverage. The JUnit XML report is generated as `test-results.xml`.

## 6. Conclusion
The automated tests successfully verified the main data processing, validation, model training, and model quality checks.
