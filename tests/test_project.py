import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from project import (DATA_PATH, TARGET, load_data, split_data, pipeline, schema_for, validate,
                     fit_selected, predict)
from train_model import metrics


def test_actual_uploaded_csv():
    if not DATA_PATH.exists():
        pytest.skip('Actual delivery CSV is not present')
    frame, features = load_data()
    assert len(frame) > 0 and TARGET in frame and 'Road_Distance_km' in features


def test_chronological_split_and_leakage_exclusions(fixture_csv):
    frame, features = load_data(fixture_csv)
    train, test, usable, removed, boundary = split_data(frame, features)
    assert pd.to_datetime(train.Order_Date).max() < pd.to_datetime(test.Order_Date).min()
    assert len(train) + len(test) == len(frame)
    assert not set(usable) & {'Order_ID', 'Order_Date', TARGET, 'Preparation_Time_Min','Average_Speed_kmph'}
    assert removed == 0


def test_preprocessing_unseen_and_missing(fixture_csv):
    frame, features = load_data(fixture_csv)
    train, test, features, _, _ = split_data(frame, features)
    model = pipeline('Linear Regression', features).fit(train[features], train[TARGET])
    row = test[features].iloc[[0]].copy()
    row['Weather'] = 'Unseen snowstorm'; row['Rider_Rating'] = np.nan
    assert np.isfinite(model.predict(row)).all()
    encoder = model.named_steps['preprocess'].named_transformers_['category'].named_steps['encode']
    assert all('Unseen snowstorm' not in categories for categories in encoder.categories_)


def test_metric_calculation():
    actual = [10,20,30]; predicted = [12,18,31]
    result = metrics(actual, predicted)
    assert result['mae_minutes'] == pytest.approx(mean_absolute_error(actual,predicted))
    assert result['rmse_minutes'] == pytest.approx(np.sqrt(mean_squared_error(actual,predicted)))
    assert result['r2'] == pytest.approx(r2_score(actual,predicted))


def test_all_models_and_validation_selection(experiment):
    path, metadata, models = experiment
    frame, features = load_data(path); _, test, features, _, _ = split_data(frame,features)
    assert len(models) == 3
    for model in models.values():
        assert np.isfinite(model.predict(test[features])).all()
    assert metadata['selected_model'] == min(metadata['models'],key=lambda r:r['cv_mae_mean'])['model']
    for fold in metadata['cv_fold_ranges']:
        assert fold['training_end'] < fold['validation_start'] < metadata['test_start']
    restored = json.loads((path.parent/'reports/metrics.json').read_text())
    assert restored['run_id'] == metadata['run_id']
    assert not (path.parent/'models').exists()


def test_fitted_prediction_uses_same_training_pipeline(experiment):
    path, metadata, models = experiment
    frame, features = load_data(path)
    bundle = fit_selected(metadata,frame,features)
    row = frame[features].iloc[10].to_dict()
    value, warnings = predict(bundle,row)
    assert value == pytest.approx(models[metadata['selected_model']].predict(pd.DataFrame([row]))[0])


@pytest.mark.parametrize('column,value',[('Road_Distance_km',-1),('Order_Hour',24),('Order_Items',1.2),
                                         ('Rider_Rating',6),('Road_Distance_km','nan'),('Road_Distance_km',0)])
def test_invalid_inputs(fixture_csv,column,value):
    frame, features = load_data(fixture_csv)
    schema = schema_for(frame,features)
    row = frame[features].iloc[10].to_dict();row[column] = value
    with pytest.raises(ValueError):
        validate(row,schema)


def test_inconsistent_weekend(fixture_csv):
    frame, features = load_data(fixture_csv)
    row = frame[features].iloc[10].to_dict();row['Day_of_Week']='Monday';row['Is_Weekend']=1
    with pytest.raises(ValueError,match='agree'):
        validate(row,schema_for(frame,features))


def test_missing_csv_and_invalid_target(fixture_csv,tmp_path):
    with pytest.raises(ValueError,match='not found'):
        load_data(tmp_path/'absent.csv')
    frame=pd.read_csv(fixture_csv).drop(columns=TARGET)
    frame.to_csv(tmp_path/'bad.csv',index=False)
    with pytest.raises(ValueError,match='Required column'):
        load_data(tmp_path/'bad.csv')
