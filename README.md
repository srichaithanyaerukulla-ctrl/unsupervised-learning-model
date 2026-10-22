# Week 3 — Unsupervised Learning & Model Evaluation

## Project overview

This project demonstrates clustering, dimensionality reduction, model evaluation, cross-validation, and hyperparameter tuning using Python and Scikit-learn. It uses the Wine Recognition dataset packaged with Scikit-learn (178 samples, 13 numeric features, and 3 reference classes).

The repository has two complementary tracks:

1. **Unsupervised analysis:** scale the 13 numeric features; tune the number of K-Means clusters using silhouette score; compare K-Means with Agglomerative/Hierarchical Clustering; evaluate cluster structure using silhouette score; and use Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) as post-hoc external comparisons against the known reference labels.
2. **Evaluation concepts:** compare tuned Logistic Regression and Random Forest classifiers on a stratified hold-out set, use 5-fold stratified cross-validation and GridSearchCV, and report confusion matrices, precision, recall, F1-score, and multiclass ROC-AUC.

**Leakage control:** the `target` and `target_name` fields are excluded from all clustering inputs. Known labels are used only to calculate external clustering metrics and in the separate supervised classification exercise. Standardization is included inside the Logistic Regression pipeline so it is fitted only on each training fold during cross-validation.

## Dataset

`data/wine_dataset.csv` is derived from `sklearn.datasets.load_wine`, the Wine Recognition dataset distributed with Scikit-learn. It contains 178 rows, 13 continuous chemical-analysis features, a numeric class label, and a readable class name. The reference labels are not inputs to clustering.

Dataset documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html>  
Original dataset page: <https://archive.ics.uci.edu/dataset/109/wine>

## Concepts demonstrated

- K-Means clustering and selection of `n_clusters` using silhouette score
- Hierarchical (Agglomerative) Clustering and linkage comparison
- StandardScaler and why feature scaling matters for distance-based algorithms
- Principal Component Analysis (PCA) for two-dimensional visualization
- Silhouette score, Adjusted Rand Index, and Normalized Mutual Information
- Stratified train-test split and 5-fold stratified cross-validation
- GridSearchCV hyperparameter tuning using macro F1-score
- Confusion matrix, accuracy, macro precision, macro recall, macro F1, and multiclass ROC-AUC
- ROC curves for one-vs-rest class comparisons
- Limitations, post-hoc evaluation, and the difference between clustering and classification

## Repository layout

```text
nsdc-week3-unsupervised-evaluation/
├── README.md
├── requirements.txt
├── data/
│   └── wine_dataset.csv
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   └── build_report.py
├── tests/
│   └── test_pipeline.py
├── outputs/
│   ├── analysis_results.json
│   ├── clustering_model_comparison.csv
│   ├── kmeans_cluster_tuning.csv
│   ├── hierarchical_linkage_comparison.csv
│   ├── cluster_assignments.csv
│   ├── classification_model_comparison.csv
│   ├── cross_validation_and_tuning_summary.csv
│   ├── confusion_matrix_*.csv
│   └── grid_search_*.csv
├── visualizations/
│   ├── 01_kmeans_silhouette_tuning.png
│   ├── 02_pca_cluster_visualization.png
│   ├── 03_hierarchical_dendrogram.png
│   ├── 04_classification_confusion_matrices.png
│   ├── 05_multiclass_roc_curves.png
│   └── 06_classification_metrics_comparison.png
└── docs/
    └── Week3_Unsupervised_Learning_Model_Evaluation_Report.docx
```

## Run the project

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.analysis
python -m src.build_report
python -m pytest -q
```

The analysis uses a fixed random seed (`42`) for reproducibility. The scripts regenerate analysis tables and figures from the included CSV dataset. The report builder then generates the Word report from those outputs.

## Results summary

In this run, K-Means selected **3 clusters** by the highest silhouette score among `k=2..6` (silhouette approximately 0.285; ARI approximately 0.897). Ward Agglomerative Clustering also used 3 clusters and obtained silhouette approximately 0.277 (ARI approximately 0.790). The two leading supervised classifiers achieved perfect scores on this specific 45-record hold-out set; five-fold cross-validation macro F1 was about 0.985. These metrics are actual run outputs, not guarantees about future data. The dataset is small and relatively separable, so further validation on external data and repeated splits would be needed before making real-world performance claims.

## Interpretation caution

Clustering is unsupervised: cluster IDs have no intrinsic meaning and do not necessarily map one-to-one to the known categories. ARI/NMI are external evaluation measures calculated after clustering; they did not choose the number of clusters. The supervised classification metrics belong to a separate experiment and should not be interpreted as cluster quality.

## License and attribution

The Wine Recognition dataset is distributed via Scikit-learn and originates from the UCI Machine Learning Repository. Consult the source dataset terms and Scikit-learn documentation for attribution and use conditions. This project code is provided for educational use.
