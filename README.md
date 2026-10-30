# Delivery Time Prediction Using Machine Learning

**DELIV PRED** is a beginner-friendly Flask mini project that estimates delivery duration in
minutes from the uploaded food-order dataset. It compares Linear Regression, Decision Tree
Regressor and Random Forest Regressor, explains prediction errors, and provides interactive
exploration and an order-detail prediction form.

## Problem statement and objectives

Delivery durations vary with route distance, traffic, weather and order characteristics.
The project learns associations in past records to estimate the target `Time_taken_min`.
It demonstrates correct regression evaluation, chronological validation, prediction-time
feature review, shared preprocessing, and a clear Flask interface.

The goals are to inspect the actual CSV, compare three regressors, select a model on earlier
training data, evaluate it on later dates, and demonstrate an in-memory prediction pipeline.
This is an educational application, not a guaranteed ETA service or production routing system.

## Dataset: actual inspection and provenance

File: `data/Food_Delivery_Time_Prediction.csv`, copied from the workspace upload. The original
file is preserved without modification.

- **50,000 orders and 24 original columns**.
- **No missing values and no exact duplicate rows** in the uploaded file.
- Target: **`Time_taken_min`**, a positive numerical duration recorded in minutes.
- Observed target range: **10–180 minutes**; median **78 minutes**.
- **18 supported predictor columns** after feature review.
- No additional duplicate predictor/date/target rows were removed during training.

This file has a different schema from the earlier recommended nine-column Kaggle dataset.
**Its original source and whether it is synthetic are unverified.** We do not claim these are
real operational delivery records. Confirm the source, license, and target definition before
making real-world claims. The precise start and end events represented by `Time_taken_min`
are not documented in the file itself; the app predicts the recorded duration rather than
asserting a guaranteed order-to-door arrival deadline. No new data was downloaded or fabricated.

### Features and information availability

Numerical inputs: order hour, weekend/festival flags, rider experience, prior rider and
restaurant ratings, order items, road distance, and number of planned route signals.
Categorical inputs: day of week, weather, pickup/dropoff zones, vehicle, cuisine, restaurant
load, traffic level, and delivery priority.

Prediction is intended once the order, rider assignment and planned route are known, before
the delivery outcome. Historical ratings must have been known then; weather/traffic/load must
be contemporaneous estimates. This availability assumption needs source verification.

Excluded columns:

| Column | Reason |
|---|---|
| `Order_ID` | Identifier, not a meaningful predictor |
| `Order_Date` | Used only for chronological partitioning |
| `Preparation_Time_Min` | May be realized preparation duration, unavailable when estimating |
| `Average_Speed_kmph` | May be a realized trip average rather than a forecast |
| `Delivery_Distance_Category` | Redundant with numerical road distance |
| `Time_taken_min` | Target; never enters the predictor matrix |

If source documentation later establishes that preparation time or speed is a forecast
available at prediction time, those inputs can be reviewed for inclusion in a new experiment.
They are currently excluded conservatively. Unknown/unreviewed columns are also excluded.

## Folder structure

```text
delivery-time-project/
├── app.py                  # Dashboard and in-memory model lifecycle
├── charts.py               # Interactive graph JSON endpoints
├── project.py              # Loading, validation, splits, pipelines and predictions
├── train_model.py          # Chronological CV, model selection, real test reports
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── .gitkeep
│   └── Food_Delivery_Time_Prediction.csv
├── reports/
│   ├── .gitkeep
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── test_predictions.csv
├── templates/page.html
├── static/style.css
└── tests/
    ├── conftest.py
    ├── test_project.py
    └── test_app.py
```

Data, generated reports, caches and the virtual environment are ignored by version control.
The existing airline project is separate and has not been replaced.

## Installation and execution

Python 3.10+ and a typical student laptop with around 8 GB RAM are sufficient. This project
was verified in the current Python 3.14 environment. It uses Pandas, NumPy, Scikit-learn,
Flask, Plotly and Pytest. There are no paid services or database requirements.

From the workspace root:

```bash
cd delivery-time-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open **http://127.0.0.1:5002**. Port 5002 keeps this project separate from the airline app.
The environment and reports have already been created in this workspace, so you can activate
the environment and run `python app.py` directly. Stop the server with Ctrl+C.
On Windows, use `python -m venv .venv` and `.venv\Scripts\activate`.

For a different dataset location:

```bash
python train_model.py --csv /absolute/path/Food_Delivery_Time_Prediction.csv
```

Use a compatible schema. The Flask app defaults to its `data/` copy; for custom paths create
an app with `create_app({'DATA_PATH': Path(...), 'REPORTS_DIR': Path(...)})` in Python.
Dataset fingerprints and a feature-version check prevent the app from displaying stale results.

To run tests:

```bash
python -m pytest -q
```

### Model handling without serialization

No trained model objects are written to or read from disk. There are no serialization imports
in project code and no model-object files. Only CSV/JSON results and summaries are persisted.

The standalone training script fits all regressors for evaluation and discards them when it
exits. On the Flask **Predict delivery time** page, click **Train prediction model** once.
This fits the CV-selected model on exactly the same earlier training period and retains the
entire fitted preprocessing/model pipeline in server memory. Repeated predictions and normal
page requests reuse it. Restarting the server, changing the data, or generating a new experiment
run requires training again. The button does not rerun all cross-validation.

The local server is one process; separate worker processes would have separate model memory.
The implementation does not import Joblib or pickle. Scikit-learn maintains its own internal
package dependencies; the project does not use them to save or load models.

## Training and validation methodology

1. Load and validate the actual CSV. Print preview, shape, dtypes, missing counts, duplicate
   counts and target distribution. Invalid/missing labels, invalid dates, non-finite or negative
   numbers, invalid flags/hours/ratings, and incompatible schemas are reported clearly.
2. Remove exact duplicate non-ID predictor/date/target records and sort chronologically.
   The source CSV is not edited. Reserve the final 20% of unique dates for testing. All
   orders on a given date stay in the same partition, so the row ratio is approximately 80/20.
3. Use **three expanding-window folds** within the earlier training dates. Training always
   precedes validation, with no date crossing a fold boundary. This simulates predicting later
   orders more realistically than random row shuffling.
4. Each model is a Scikit-learn Pipeline. Its ColumnTransformer sends numerical columns through
   training-median imputation and StandardScaler, and categories through training-mode imputation
   and OneHotEncoder (`handle_unknown='ignore'`). Every fold learns preprocessing only from its
   own training rows. No target is included as an input and no resampling is applied.
5. Compare fixed, modest-capacity regressors: Linear Regression; a depth-6 Decision Tree with
   minimum leaf size 30; a 100-tree Random Forest with depth 12, minimum leaf size 10, two worker
   threads and random seed 42. No hyperparameter search is performed on the test set.
6. Select the lowest mean training-only validation MAE. Refit each regressor on the full earlier
   training period, then evaluate once on the later test period. Save actual metrics and selected
   model test predictions. Full-training scores are diagnostic, not independent validation.
7. Compare with a constant baseline that predicts the training target median for every test order.

**Actual partition:** 40,101 training orders and 9,899 test orders. The test period starts
**2025-10-20**. The precise fold ranges and counts are recorded in `reports/metrics.json`.

## Actual experimental results

| Model | CV MAE (min) | Test MAE (min) | Test RMSE (min) | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 6.63 | 6.69 | 9.58 | 0.9266 |
| Decision Tree | 9.86 | 9.72 | 12.78 | 0.8694 |
| Random Forest | 5.81 | 5.33 | 6.98 | 0.9611 |

The selected model is **Random Forest**, based on training-only CV MAE. The training-median
baseline's test MAE is **27.07 minutes**. These are measured results on the uploaded dataset;
they do not establish equivalent performance on operational delivery data.

### Metric formulas and explanations

For actual duration y, predicted duration ŷ, and n test orders:

- **MAE = Σ|y − ŷ| / n.** Average error magnitude, measured in minutes; lower is better.
- **RMSE = √(Σ(y − ŷ)² / n).** Also minutes; large mistakes receive extra weight.
- **R² = 1 − Σ(y − ŷ)² / Σ(y − mean(y))².** Relative squared-error fit; 1 is perfect,
  0 matches a constant test-mean predictor, and values below 0 are possible.
- **Residual = actual − predicted.** Positive residuals mean the estimate was too short.

R² is **not classification accuracy** and should not be described as a percentage of orders
predicted correctly. Test MAE is an aggregate historical error; it is not a per-order confidence
interval. Small training/validation/test error gaps help assess generalization, while high R²
alone neither proves nor disproves overfitting. CV standard deviation is the population standard
deviation across the three observed fold MAEs, not a confidence interval.

## Complete interactive Flask dashboard

The dashboard now includes anchored navigation, summary cards, the six-field prediction
form, all three regressors' actual comparisons, delivery-pattern filters and model insights.
All 13 graphs are interactive Plotly figures served from local JSON endpoints:

- Test MAE, RMSE and R² comparisons, plus training-versus-test MAE.
- Actual-versus-predicted estimates and residual distributions.
- Recorded duration distribution, means by traffic/weather/hour, and distance relationships.
- Random Forest feature importance and numerical correlations.

Hover to inspect values; use the toolbar to zoom, pan, reset axes or download a PNG. Legends
can hide/show traces. **Expand** opens a larger interactive dialog; Close or Escape dismisses
it. Mouse-wheel zoom is disabled so scrolling through the page remains comfortable. The
weather filter updates descriptive delivery graphs and correlation, while comparison and
prediction-error graphs retain the complete test experiment. Scatterplots use up to 1,500
actual records with seed 42; histograms show aggregated counts of actual records.

Plotly JavaScript is bundled locally in `static/plotly.min.js`, with its license header.
No CDN, external API or internet connection is needed to render the dashboard after setup.
Chart endpoints live in `charts.py`. Only near-visible graphs load initially. The numerical
reports and underlying models are unchanged by these interface improvements.

The six prediction fields are distance, traffic, weather, vehicle, item count and order hour.
Remaining predictors use medians/modes learned from the training period via the existing
imputer. Historical test MAE uses full recorded inputs; the reduced-input form may be less
precise. Click **Prepare model** once per server session before making predictions.

### Background and transparency

A decorative delivery-scene image is saved at `static/delivery-background.png`. It was
created with the **built-in imagegen tool**, not downloaded as a dataset record or generated
with the fallback CLI. The final generation prompt is preserved verbatim in
[`static/background-prompt.txt`](static/background-prompt.txt).

The **Panel opacity** slider ranges from 70–100% (default 94%). The separate **Graph
background** slider ranges from 80–100% (default 100%, solid white). Only surface backgrounds
change; graph marks and text remain opaque. Preferences are stored in browser local storage.
The larger graphs use white backgrounds for contrast, while the image remains visible around
panels. The layout adapts to smaller displays.

## Limitations and future scope

The source and generation process need verification. A high score could partly reflect an easy
synthetic generation rule. Historical ratings, rider assignment, road-distance estimates,
traffic and route signals may not all be available at initial order placement; the intended
estimation point must be documented for real deployment. Target start/end semantics also need
confirmation. The file's observed maximum of 180 minutes may reflect clipping, but its cause
cannot be established from the CSV alone.

Chronological validation reduces future-information leakage but does not prove robustness to
new cities, restaurants, riders or future traffic patterns. The dataset lacks explicit rider and
restaurant IDs for group-aware validation. A single test period has uncertainty. Unknown
categories, outliers, and orders outside observed ranges can yield poor estimates. Feature
importance reflects associations in one model and can be biased toward continuous variables.

Future work: verified operational data, richer time/group validation, prediction-interval
calibration, route-network estimates, prospective trials, model monitoring, and a documented
production serving approach. No production reliability or guaranteed arrival deadline is claimed.

## Verification performed

**20 automated tests passed.** They cover real-file loading, chronological separation, feature
leakage exclusions, missing values, unseen categories, exact metric calculations, all three
models, training-only selection, report round trips, input validation, the interactive dashboard, all 13 graph APIs,
weather filtering, real-score comparison traces, bundled assets, legacy URL redirects,
missing-file handling, simplified prediction, in-memory reuse and restart behavior.
The current suite completed with no test failures or warnings.

Test-only generated fixtures are isolated under pytest's temporary directory. They are not
project datasets and are never included in the actual reports or displayed as experiment results.

Real-data integration checks passed for all 13 interactive graph JSON endpoints using the
uploaded CSV. Chrome visual checks confirmed the generated background, enlarged white graph
surfaces, graph toolbars, expansion dialog, and both opacity controls. The panel-opacity value
was changed successfully in the browser. The running Flask server is available on port 5002.

## Viva questions

1. **Is this classification or regression?** Regression, because delivery duration is numerical.
2. **Why not report accuracy?** Classification accuracy compares labels; numerical duration needs error metrics.
3. **What is Linear Regression?** A linear combination of encoded inputs used to estimate a continuous target.
4. **What is a Decision Tree Regressor?** Feature-based splits whose leaves predict averages from training examples.
5. **What is Random Forest Regression?** An average of predictions from many randomized trees.
6. **Why use MAE for selection?** It expresses average error directly in minutes and is easy to explain.
7. **Why also use RMSE?** It emphasizes occasional large errors that MAE can obscure.
8. **What does R² mean?** Improvement in squared error compared with prediction by the test target mean.
9. **What is chronological validation?** Learn from earlier orders and validate on later orders.
10. **What is an expanding window?** Each successive fold adds more earlier training dates and validates on the next block.
11. **Why exclude average speed?** Realized trip speed may be known only after the target journey happens.
12. **Why exclude preparation duration?** Its availability as an advance estimate has not been established.
13. **Why use a Pipeline?** It ensures the same fitted imputer, encoder and scaler are used during prediction.
14. **Why one-hot encode?** It represents categories numerically without imposing an artificial ordered scale.
15. **Why scale numerical features?** It standardizes their magnitudes; it is useful for linear methods and harmless for these trees.
16. **What is the median baseline?** Predict the training target median for every future order, without using features.
17. **What is a residual?** Actual time minus predicted time; positive means the order took longer than estimated.
18. **Does feature importance establish causation?** No; it describes how one model uses associated predictors.
19. **How do models persist in the app?** Only in server memory; the user trains again after restarting.
20. **Are the data verified real-world records?** No; their original source and generation process remain unverified.
