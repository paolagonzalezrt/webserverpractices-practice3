
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer


def pause():
    input("\nPress Enter to continue...")


def main():
    # STEP 1: Generate the noisy dataset
    data = {
        "edad": [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],
        "ingreso_mensual": [
            15000, 22000, 18000, np.nan, 12000,
            950000, 25000, 21000, 19000, np.nan
        ],
        "scoring_credito": [
            650, 700, 680, 710, 590, 800, 670, 640, 620, 730
        ]
    }

    df_dirty = pd.DataFrame(data)
    df_original = df_dirty.copy()

    print("STEP 1: ORIGINAL DATASET")
    print(df_dirty)
    print("\nMissing values per column:")
    print(df_dirty.isnull().sum())
    print("\nDataset information:")
    print(df_dirty.describe())

    pause()

    # STEP 2: Detect income outliers using IQR
    Q1 = df_dirty["ingreso_mensual"].quantile(0.25)
    Q3 = df_dirty["ingreso_mensual"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("\nSTEP 2: OUTLIER DETECTION")
    print(f"Q1: ${Q1:,.2f}")
    print(f"Q3: ${Q3:,.2f}")
    print(f"IQR: ${IQR:,.2f}")
    print(f"Lower bound: ${lower_bound:,.2f}")
    print(f"Upper bound: ${upper_bound:,.2f}")

    income_outliers = (
        (df_dirty["ingreso_mensual"] < lower_bound)
        | (df_dirty["ingreso_mensual"] > upper_bound)
    )
    print("\nDetected income outliers:")
    print(df_dirty.loc[income_outliers])

    pause()

    # STEP 3: Replace anomalous values with NaN
    df_cleaning = df_dirty.copy()

    df_cleaning.loc[income_outliers, "ingreso_mensual"] = np.nan
    df_cleaning.loc[df_cleaning["edad"] > 100, "edad"] = np.nan

    print("\nSTEP 3: DATA AFTER OUTLIER TREATMENT")
    print(df_cleaning)
    print("\nMissing values after treatment:")
    print(df_cleaning.isnull().sum())

    pause()

    # STEP 4: Impute missing values using KNN (k=2)
    knn_imputer = KNNImputer(n_neighbors=2)
    knn_array = knn_imputer.fit_transform(df_cleaning)

    df_knn = pd.DataFrame(
        knn_array,
        columns=df_cleaning.columns
    )

    print("\nSTEP 4: KNN IMPUTATION (k=2)")
    print(df_knn.round(2))
    print("\nMissing values after KNN:")
    print(df_knn.isnull().sum())

    pause()

    # STEP 5: Compare KNN with mean and median imputation
    mean_imputer = SimpleImputer(strategy="mean")
    median_imputer = SimpleImputer(strategy="median")

    df_mean = pd.DataFrame(
        mean_imputer.fit_transform(df_cleaning),
        columns=df_cleaning.columns
    )
    df_median = pd.DataFrame(
        median_imputer.fit_transform(df_cleaning),
        columns=df_cleaning.columns
    )

    comparison = pd.DataFrame({
        "KNN": df_knn.loc[[2, 3, 5, 8, 9]].mean(),
        "Mean": df_mean.loc[[2, 3, 5, 8, 9]].mean(),
        "Median": df_median.loc[[2, 3, 5, 8, 9]].mean()
    })

    print("\nSTEP 5: IMPUTATION METHOD COMPARISON")
    print("Mean of selected rows (for illustration only):")
    print(comparison.round(2))
    print("\nKNN results:")
    print(df_knn.round(2))
    print("\nMean imputation results:")
    print(df_mean.round(2))
    print("\nMedian imputation results:")
    print(df_median.round(2))

    pause()

    # STEP 6: Visualize income before and after treatment
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot(df_original["ingreso_mensual"].dropna())
    axes[0].set_title("Income Before Outlier Treatment")
    axes[0].set_ylabel("Monthly Income")

    axes[1].boxplot(df_knn["ingreso_mensual"])
    axes[1].set_title("Income After Cleaning and KNN")
    axes[1].set_ylabel("Monthly Income")

    plt.tight_layout()
    plt.savefig("income_boxplots.png")
    plt.show()

    print("\nSTEP 6: VISUALIZATION")
    print("Chart saved as income_boxplots.png")

    pause()

    # STEP 7: Save the cleaned dataset
    df_knn.to_csv("cleaned_data.csv", index=False)

    print("\nSTEP 7: SAVE RESULTS")
    print("Cleaned dataset saved as cleaned_data.csv")
    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()
