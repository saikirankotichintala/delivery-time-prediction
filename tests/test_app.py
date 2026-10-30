import pytest
from app import create_app, FORM_FIELDS
from project import load_data


@pytest.fixture
def client(experiment):
    path, _, _ = experiment
    return create_app({'TESTING':True,'DATA_PATH':path,'REPORTS_DIR':path.parent/'reports'}).test_client()


def test_complete_dashboard(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Total orders' in response.data and b'Compare the three regressors' in response.data
    assert b'Panel opacity' in response.data and b'Numerical correlations' in response.data
    assert b'Graph background' in response.data and b'plotly.min.js' in response.data
    assert response.data.count(b'/chart/') == 13
    assert b'delivery-background.png' in client.get('/static/style.css').data
    for route in ['/analysis','/models','/predict','/insights']:
        assert client.get(route).status_code == 302
    assert client.get('/health').json == {'status':'ok'}


def test_prediction_and_memory_reuse(client,experiment,monkeypatch):
    import app as module
    calls=[]; original=module.fit_selected
    def track(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'fit_selected',track)
    assert client.post('/train-prediction').status_code==302
    assert client.post('/train-prediction').status_code==302
    assert len(calls)==1
    path,_,_=experiment;frame,features=load_data(path)
    row={c:str(frame.iloc[10][c]) for c in FORM_FIELDS if c in features}
    response=client.post('/',data=row)
    assert response.status_code==200 and b'Estimated delivery time' in response.data
    assert b'Other details use training defaults' in response.data
    row['Order_Hour']='99'
    assert client.post('/',data=row).status_code==400
    assert client.post('/',data={}).status_code==400


def test_missing_reports_and_data(tmp_path,experiment):
    client=create_app({'TESTING':True,'DATA_PATH':tmp_path/'absent.csv','REPORTS_DIR':tmp_path}).test_client()
    assert b'Dataset missing' in client.get('/').data
    path,_,_=experiment
    client=create_app({'TESTING':True,'DATA_PATH':path,'REPORTS_DIR':tmp_path}).test_client()
    assert b'Run python train_model.py first' in client.get('/').data


def test_new_server_needs_training(experiment):
    path,_,_=experiment
    client=create_app({'TESTING':True,'DATA_PATH':path,'REPORTS_DIR':path.parent/'reports'}).test_client()
    assert b'Prepare model' in client.get('/').data
    assert client.post('/',data={}).status_code==400


def test_dashboard_graphs_and_filters(client):
    for name in ['mae','rmse','r2','generalization','actual','residual','importance',
                 'distribution','traffic','weather','hourly','distance','correlation']:
        response = client.get('/api/chart/' + name)
        assert response.status_code == 200 and response.is_json, name
        assert response.json['data'] and 'xaxis' in response.json['layout']
    assert client.get('/api/chart/traffic?weather=Rain').status_code == 200
    assert client.get('/api/chart/traffic?weather=Unknown').status_code == 400
    assert b'Rain weather' in client.get('/?weather=Rain').data
    assert b'Unknown weather filter' in client.get('/?weather=Unknown').data
    assert client.get('/api/chart/unknown').status_code == 400
    assert client.get('/static/delivery-background.png').status_code == 200


def test_interactive_comparison_uses_real_scores(client,experiment):
    _,metadata,_=experiment
    result=client.get('/api/chart/mae').json
    assert result['data'][0]['y'] == [row['test']['mae_minutes'] for row in metadata['models']]
    assert client.get('/static/plotly.min.js').status_code == 200
    assert b'expandGraph' in client.get('/static/dashboard.js').data
