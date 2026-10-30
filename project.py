"""Shared real-data loading, chronological splitting, preprocessing and validation."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'Food_Delivery_Time_Prediction.csv'
REPORTS = ROOT / 'reports'
TARGET = 'Time_taken_min'
DATE = 'Order_Date'
VERSION = 'order-time-v1'
MODEL_LABELS = {'Decision Tree': 'Decision Tree Regression', 'Random Forest': 'Random Forest Regressor'}


def model_label(name):
    return MODEL_LABELS.get(name, name)

NUMERIC = ['Order_Hour', 'Is_Weekend', 'Is_Festival', 'Rider_Experience_Years',
           'Rider_Rating', 'Restaurant_Rating', 'Order_Items', 'Road_Distance_km', 'Number_of_Signals']
CATEGORICAL = ['Day_of_Week', 'Weather', 'Pickup_Zone', 'Dropoff_Zone', 'Vehicle_Type',
               'Cuisine_Type', 'Restaurant_Load', 'Traffic_Level', 'Delivery_Priority']
INTEGER = ['Order_Hour', 'Is_Weekend', 'Is_Festival', 'Order_Items', 'Number_of_Signals']
# Actual preparation duration and trip-average speed may only be known after the journey.
EXCLUDED = {'Order_ID': 'Identifier', DATE: 'Split boundary only; not an input',
            'Preparation_Time_Min': 'Availability at order placement is unverified',
            'Average_Speed_kmph': 'May be realized trip speed, unavailable at order placement',
            'Delivery_Distance_Category': 'Redundant with road distance', TARGET: 'Prediction target'}


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Dataset not found: {path}. Put Food_Delivery_Time_Prediction.csv in data/.')
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read dataset: {exc}') from exc
    if frame.empty or frame.dropna(how='all').empty:
        raise ValueError('The CSV contains no populated records.')
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    for column in [TARGET, DATE]:
        if column not in frame:
            raise ValueError(f'Required column missing: {column}. Actual columns: {frame.columns.tolist()}')
    features = [c for c in NUMERIC + CATEGORICAL if c in frame]
    if 'Road_Distance_km' not in features:
        raise ValueError('Road_Distance_km is required for this dataset schema.')
    for column in [TARGET] + [c for c in NUMERIC if c in features]:
        values = pd.to_numeric(frame[column], errors='coerce')
        if (frame[column].notna() & values.isna()).any() or np.isinf(values.dropna()).any():
            raise ValueError(f'{column} contains invalid numeric values.')
        if values.dropna().lt(0).any():
            raise ValueError(f'{column} cannot contain negative values.')
        frame[column] = values
    if frame[TARGET].isna().any() or frame[TARGET].le(0).any():
        raise ValueError('Delivery time must be populated, finite and greater than zero.')
    for column in [c for c in INTEGER if c in features]:
        if frame[column].dropna().mod(1).ne(0).any():
            raise ValueError(f'{column} must contain whole numbers.')
    for column, upper in [('Order_Hour', 23), ('Is_Weekend', 1), ('Is_Festival', 1),
                           ('Rider_Rating', 5), ('Restaurant_Rating', 5)]:
        if column in features and frame[column].dropna().gt(upper).any():
            raise ValueError(f'{column} must be at most {upper}.')
    for column in ['Road_Distance_km', 'Order_Items']:
        if column in features and frame[column].dropna().le(0).any():
            raise ValueError(f'{column} must be positive.')
    for column in [c for c in CATEGORICAL if c in features]:
        frame[column] = frame[column].map(lambda v: str(v) if pd.notna(v) else np.nan)
    dates = pd.to_datetime(frame[DATE], errors='coerce')
    if dates.isna().any():
        raise ValueError('Order_Date must contain valid dates for chronological evaluation.')
    return frame, features


def split_data(frame, features):
    # Drop duplicate non-ID records before splitting; preserve the source CSV on disk.
    subset = [DATE] + features + [TARGET]
    clean = frame.drop_duplicates(subset=subset).sort_values([DATE, 'Order_Hour'] if 'Order_Hour' in frame else [DATE]).reset_index(drop=True)
    dates = pd.to_datetime(clean[DATE]).dt.normalize()
    unique_dates = dates.drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 10:
        raise ValueError('At least 10 distinct order dates are needed for chronological training/validation/test.')
    boundary = unique_dates[int(len(unique_dates) * .8)]
    training = clean.loc[dates < boundary]
    testing = clean.loc[dates >= boundary]
    usable = [c for c in features if training[c].notna().any()]
    if not usable:
        raise ValueError('All prediction features are missing in the training period.')
    return training, testing, usable, int(len(frame) - len(clean)), str(boundary.date())


def preprocessor(features):
    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c in CATEGORICAL]
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)],
        verbose_feature_names_out=False)


def candidates():
    return {'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10,
                                                   n_jobs=2, random_state=42)}


def pipeline(name, features):
    return Pipeline([('preprocess', preprocessor(features)), ('regressor', candidates()[name])])


def schema_for(frame, features):
    schema = {}
    for c in features:
        if c in NUMERIC:
            schema[c] = {'kind': 'numeric', 'median': float(frame[c].median()),
                         'min': float(frame[c].min()), 'max': float(frame[c].max())}
        else:
            schema[c] = {'kind': 'categorical', 'options': sorted(frame[c].dropna().unique().tolist())}
    return schema


def validate(values, schema):
    if set(values) != set(schema):
        raise ValueError('Input fields must match the trained feature schema.')
    cleaned, warnings = {}, []
    for c, spec in schema.items():
        value = values[c]
        if value is None or value == '':
            cleaned[c] = np.nan
            continue
        if spec['kind'] == 'numeric':
            try:
                value = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{c} must be numeric.') from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{c} must be finite and nonnegative.')
            if c in INTEGER and value % 1:
                raise ValueError(f'{c} must be a whole number.')
            maximum = {'Order_Hour': 23, 'Is_Weekend': 1, 'Is_Festival': 1,
                       'Rider_Rating': 5, 'Restaurant_Rating': 5}.get(c)
            if maximum is not None and value > maximum:
                raise ValueError(f'{c} must be at most {maximum}.')
            if c in ['Road_Distance_km', 'Order_Items'] and value <= 0:
                raise ValueError(f'{c} must be greater than zero.')
            if value < spec['min'] or value > spec['max']:
                warnings.append(f'{c} is outside the observed training range.')
            cleaned[c] = value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{c} must be a nonempty category.')
            cleaned[c] = value
            if value not in spec['options']:
                warnings.append(f'{c}: unseen category; the trained encoder will ignore its indicator.')
    if all(pd.notna(cleaned.get(c)) for c in ['Day_of_Week', 'Is_Weekend']):
        day = cleaned['Day_of_Week']
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if int(cleaned['Is_Weekend']) != int(day in ['Saturday', 'Sunday']):
                raise ValueError('Day of week and weekend flag must agree.')
    return pd.DataFrame([cleaned], columns=list(schema)), warnings


def fit_selected(metadata, frame, features):
    training, _, features, _, _ = split_data(frame, features)
    if features != metadata['features']:
        raise ValueError('Features differ from the evaluated model. Run training again.')
    model = pipeline(metadata['selected_model'], features)
    model.fit(training[features], training[TARGET])
    return {'pipeline': model, 'schema': schema_for(training, features), 'run_id': metadata['run_id']}


def predict(bundle, values):
    row, warnings = validate(values, bundle['schema'])
    prediction = float(bundle['pipeline'].predict(row)[0])
    if not np.isfinite(prediction):
        raise ValueError('The model returned an invalid estimate.')
    # Do not silently clip regression output: expose an implausible estimate as an error.
    if prediction <= 0:
        raise ValueError('The model returned a nonpositive duration. Check the inputs and observed training ranges.')
    return prediction, warnings
