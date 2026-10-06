# SpaceX Falcon 9 First Stage Landing Prediction

This project analyzes SpaceX launch data and builds a machine learning pipeline to predict whether the Falcon 9 first stage will land successfully.

## Project structure

- `data/` — CSV and JSON datasets used in the analysis
- `notebooks/` — Jupyter notebooks for data collection, cleaning, analysis, and modeling
- `outputs/` — generated reports and output artifacts

## Goals

- Explore the launch dataset
- Prepare features and labels for model training
- Train and compare classification models
- Evaluate model performance and select the best approach

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open the notebook in the `notebooks/` directory and run the cells.

## Dependencies

The project uses Python packages such as:

- pandas
- numpy
- requests
- matplotlib
- seaborn
- folium
- scikit-learn
- jupyter
- ipykernel

## Data

The project uses SpaceX launch and landing datasets stored in the `data/` folder.

## Key findings and results

### Launch site location findings

The launch site analysis identifies the main SpaceX launch facilities used in the project:

- CCAFS LC-40
- KSC LC-39A
- VAFB SLC-4E

These sites are geographically distinct and represent the main operational launch corridors used for Falcon 9 missions. The spatial analysis showed that launch-site location is an important contextual feature because it affects launch trajectory, recovery options, and the feasibility of first-stage landing attempts. In particular, the east-coast Florida sites (CCAFS and KSC) are strongly linked to the reusable-rocket mission profile and landing/recovery strategy used in this project.

### Prediction model results

The machine learning notebook evaluates several classifiers on the standardized dataset:

- Logistic Regression: cross-validation accuracy ≈ 0.7917
- SVM: cross-validation accuracy ≈ 0.8482
- Decision Tree: cross-validation accuracy ≈ 0.8750
- KNN: cross-validation accuracy ≈ 0.8482

The best-performing model was the Decision Tree classifier, with a validation accuracy of about 87.5%. When evaluated on the held-out test set, the logistic regression model achieved an accuracy of 0.8333.


```python
from sklearn.metrics import classification_report, f1_score

pred = best_model.predict(X_test)
print(classification_report(Y_test, pred))
print("F1 score:", f1_score(Y_test, pred))
```

The overall conclusion is that launch-site information and related mission features are relevant to landing prediction, and the Decision Tree model performed best on the validation set for this project.

## Notes

This repository is intended for learning and exploratory machine learning work in a Jupyter environment.
