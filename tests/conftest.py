"""Only automated tests use small generated fixtures, isolated in temporary directories."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='session')
def fixture_csv(tmp_path_factory):
    folder = tmp_path_factory.mktemp('delivery-test-only')
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range('2024-01-01', periods=30).repeat(4)
    distance = rng.uniform(1, 30, n)
    frame = pd.DataFrame({'Order_ID': [f'TEST{i}' for i in range(n)], 'Order_Date': dates.strftime('%Y-%m-%d'),
                          'Order_Hour': rng.integers(0, 24, n), 'Is_Weekend': (dates.dayofweek >= 5).astype(int),
                          'Day_of_Week': dates.day_name(), 'Road_Distance_km': distance,
                          'Weather': rng.choice(['Clear', 'Rain'], n), 'Traffic_Level': rng.choice(['Low','High'], n),
                          'Order_Items': rng.integers(1, 5, n), 'Rider_Rating': rng.uniform(3, 5, n),
                          'Preparation_Time_Min': rng.integers(5, 30, n),
                          'Average_Speed_kmph': rng.uniform(10, 40, n),
                          'Time_taken_min': 10 + distance * 2 + rng.normal(0, 1, n)})
    frame.loc[0, 'Weather'] = np.nan
    frame.loc[2, 'Rider_Rating'] = np.nan
    path = folder / 'test_fixture.csv'
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope='session')
def experiment(fixture_csv):
    from train_model import train
    metadata, models = train(fixture_csv, reports=fixture_csv.parent / 'reports', return_models=True)
    return fixture_csv, metadata, models
