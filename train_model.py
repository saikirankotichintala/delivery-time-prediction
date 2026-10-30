"""Fit and compare regressors without saving trained model objects."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from project import (DATA_PATH, REPORTS, TARGET, DATE, VERSION, EXCLUDED, fingerprint,
                     load_data, split_data, candidates, pipeline)


def metrics(actual, predicted):
    return {'mae_minutes': float(mean_absolute_error(actual, predicted)),
            'rmse_minutes': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted))}


def train(path=DATA_PATH, reports=REPORTS, return_models=False):
    frame, features = load_data(path)
    training, testing, features, removed, boundary = split_data(frame, features)
    print('First five rows:\n', frame.head().to_string())
    print('Shape:', frame.shape, '\nData types:\n', frame.dtypes.to_string())
    print('Missing values:\n', frame.isna().sum().to_string())
    print('Raw exact duplicates:', frame.duplicated().sum(), 'Non-ID duplicate rows removed:', removed)
    print('Target distribution:\n', frame[TARGET].describe().to_string())
    print('Chronological train/test:', len(training), len(testing), 'Test starts:', boundary)
    # Split unique TRAINING dates, so equal dates never cross any CV fold boundary.
    dates = pd.to_datetime(training[DATE]).dt.normalize()
    unique_dates = dates.drop_duplicates().sort_values().to_numpy()
    folds = []
    fold_ranges = []
    for train_days, validation_days in TimeSeriesSplit(n_splits=3).split(unique_dates):
        train_indices = np.flatnonzero(dates.isin(unique_dates[train_days]).to_numpy())
        validation_indices = np.flatnonzero(dates.isin(unique_dates[validation_days]).to_numpy())
        folds.append((train_indices, validation_indices))
        fold_ranges.append({'training_rows': len(train_indices), 'validation_rows': len(validation_indices),
                            'training_end': str(dates.iloc[train_indices].max().date()),
                            'validation_start': str(dates.iloc[validation_indices].min().date())})
    x, y = training[features], training[TARGET]
    rows, fitted = [], {}
    for name in candidates():
        print('Training-only time-series validation:', name, flush=True)
        model = pipeline(name, features)
        start = perf_counter()
        cv = cross_validate(model, x, y, cv=folds, scoring='neg_mean_absolute_error',
                            return_train_score=True, n_jobs=1, error_score='raise')
        cv_seconds = perf_counter() - start
        start = perf_counter(); model.fit(x, y); fit_seconds = perf_counter() - start
        rows.append({'model': name, 'cv_mae_mean': float(-cv['test_score'].mean()),
                     'cv_mae_std': float(cv['test_score'].std()),
                     'cv_training_mae': float(-cv['train_score'].mean()),
                     'cv_fold_mae': (-cv['test_score']).tolist(), 'cv_seconds': cv_seconds,
                     'fit_seconds': fit_seconds})
        fitted[name] = model
    selected = min(rows, key=lambda r: r['cv_mae_mean'])['model']
    print('Selected by lowest training-only CV MAE:', selected)
    output = Path(reports); output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        model = fitted[row['model']]
        row['training'] = metrics(y, model.predict(x))
        start = perf_counter(); prediction = model.predict(testing[features])
        row['prediction_seconds'] = perf_counter() - start
        row['test'] = metrics(testing[TARGET], prediction)
    predictions = fitted[selected].predict(testing[features])
    # A training-median predictor is a meaningful no-feature regression baseline.
    baseline = metrics(testing[TARGET], np.repeat(y.median(), len(testing)))
    selected_importance = fitted['Random Forest']
    importance = pd.DataFrame({'feature': selected_importance.named_steps['preprocess'].get_feature_names_out(),
                              'importance': selected_importance.named_steps['regressor'].feature_importances_})
    importance.sort_values('importance', ascending=False).to_csv(output / 'feature_importance.csv', index=False)
    pd.DataFrame({'Order_Date': testing[DATE].to_numpy(), 'actual_minutes': testing[TARGET].to_numpy(),
                  'predicted_minutes': predictions, 'residual_minutes': testing[TARGET].to_numpy() - predictions}).to_csv(
                      output / 'test_predictions.csv', index=False)
    metadata = {'run_id': str(uuid4()), 'version': VERSION, 'dataset_sha256': fingerprint(path),
                'source_file': Path(path).name, 'raw_rows': len(frame), 'raw_columns': len(frame.columns),
                'raw_duplicates': int(frame.duplicated().sum()), 'duplicates_removed': removed,
                'missing': {c: int(v) for c, v in frame.isna().sum().items()},
                'training_rows': len(training), 'testing_rows': len(testing), 'test_start': boundary,
                'features': features, 'excluded_columns': {c: EXCLUDED.get(c, 'Unsupported/unreviewed input') for c in frame if c not in features},
                'selected_model': selected, 'selection': 'Lowest mean 3-fold expanding-window training-only MAE',
                'cv_fold_ranges': fold_ranges, 'models': rows, 'median_baseline_test': baseline,
                'training_target_median': float(y.median()), 'target_min': float(y.min()), 'target_max': float(y.max()),
                'provenance': 'User-supplied CSV; real-world versus synthetic origin unverified.',
                'prediction_time': 'Order placement; excludes realized preparation duration and trip-average speed.'}
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    flat = []
    for row in rows:
        values = {k: v for k, v in row.items() if k not in ['test', 'training', 'cv_fold_mae']}
        values.update({'test_' + k: v for k, v in row['test'].items()})
        values.update({'train_' + k: v for k, v in row['training'].items()})
        flat.append(values)
    pd.DataFrame(flat).to_csv(output / 'model_comparison.csv', index=False)
    print(pd.DataFrame(flat).to_string(index=False))
    return (metadata, fitted) if return_models else metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=str(DATA_PATH))
    args = parser.parse_args()
    try:
        train(args.csv)
        return 0
    except Exception as exc:
        print(f'Training failed ({type(exc).__name__}): {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
