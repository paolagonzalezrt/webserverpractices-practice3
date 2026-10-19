# Practice 3: Implementation of Advanced Data Cleaning in Python

## Description

This project identifies missing values and anomalous records in a synthetic dataset using pandas, NumPy, and scikit-learn.

The script detects outliers with the Interquartile Range (IQR) technique, replaces invalid values with NaN, and imputes missing values using KNNImputer. It also compares KNN imputation with mean and median methods.

## Features

- Synthetic dataset generation with age, monthly income, and credit score.
- Missing value identification and statistical analysis.
- Outlier detection using the IQR method.
- Boxplot visualization before and after data cleaning.
- Export of the cleaned dataset to a CSV file.

## Requirements

- Python 3.8+
- pandas
- NumPy
- scikit-learn
- SciPy
- Matplotlib

## Installation

1. Clone this repository and navigate to the project directory.
2. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Run the script from the project root directory:

```bash
python scripts/clean_data.py
```

The program pauses after each step. Press Enter to continue through dataset generation, quality analysis, outlier detection, KNN imputation, method comparison, and visualization.

## Output

The script generates the following files in the project root directory:

- `income_boxplots.png`: Boxplots comparing monthly income before and after cleaning and imputation.
- `cleaned_data.csv`: Dataset containing the imputed values and no remaining missing values.