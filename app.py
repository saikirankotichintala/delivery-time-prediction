"""One-page delivery dashboard. The prediction model stays in server memory."""
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from flask import Flask, render_template, request, redirect, url_for
from charts import charts
from project import (DATA_PATH, REPORTS, TARGET, VERSION,
                     load_data, fingerprint, fit_selected, predict, model_label)

# The remaining model inputs use training medians/modes from the fitted pipeline.
FORM_FIELDS = ['Road_Distance_km', 'Traffic_Level', 'Weather', 'Vehicle_Type', 'Order_Items', 'Order_Hour']


@lru_cache(maxsize=2)
def cached_data(path, stamp):
    frame, features = load_data(path)
    return frame, features, fingerprint(path)


def create_app(config=None):
    app = Flask(__name__)
    app.jinja_env.filters['model_label'] = model_label
    app.config.update(DATA_PATH=DATA_PATH, REPORTS_DIR=REPORTS, MAX_CONTENT_LENGTH=64 * 1024)
    if config:
        app.config.update(config)
    app.register_blueprint(charts)
    runtime = {'bundle': None}
    training_lock = RLock()

    def data():
        path = Path(app.config['DATA_PATH'])
        if not path.is_file():
            raise ValueError('Dataset missing. Place the delivery CSV in data/.')
        return cached_data(str(path), path.stat().st_mtime_ns)

    def report():
        path = Path(app.config['REPORTS_DIR']) / 'metrics.json'
        if not path.is_file():
            raise ValueError('Run python train_model.py first.')
        metadata = json.loads(path.read_text())
        if metadata['dataset_sha256'] != data()[2] or metadata['version'] != VERSION:
            raise ValueError('The data changed. Run python train_model.py again.')
        return metadata

    def memory_model(metadata):
        if runtime['bundle'] and runtime['bundle']['run_id'] != metadata['run_id']:
            runtime['bundle'] = None
        return runtime['bundle']

    @app.route('/', methods=['GET', 'POST'])
    def dashboard():
        context = {'estimate': None, 'error': None, 'ready': False, 'stats': None,
                   'metadata': None, 'fields': {}, 'values': {}, 'warnings': [], 'weather_options': [], 'weather': 'All'}
        try:
            frame, _, _ = data()
            options = sorted(frame.Weather.dropna().unique().tolist()) if 'Weather' in frame else []
            weather = request.args.get('weather', 'All')
            if weather != 'All' and weather not in options:
                raise ValueError('Unknown weather filter.')
            filtered = frame if weather == 'All' else frame[frame.Weather == weather]
            context.update(weather_options=options, weather=weather, filtered_orders=len(filtered))
            context['stats'] = {'orders': len(frame), 'median': float(frame[TARGET].median()),
                                'distance': float(frame['Road_Distance_km'].mean())}
            metadata = report()
            selected = next(row for row in metadata['models'] if row['model'] == metadata['selected_model'])
            context.update(metadata=metadata, mae=selected['test']['mae_minutes'])
            bundle = memory_model(metadata)
            if bundle:
                fields = {c: bundle['schema'][c] for c in FORM_FIELDS if c in bundle['schema']}
                context.update(ready=True, fields=fields)
                if request.method == 'POST':
                    submitted = request.form.to_dict()
                    context['values'] = submitted
                    if set(submitted) != set(fields):
                        raise ValueError('Complete the delivery details below.')
                    # Omitted fields are missing so the existing training-only imputer supplies
                    # medians/modes. No outcome information or newly fitted transforms are used.
                    values = {c: None for c in bundle['schema']}
                    values.update(submitted)
                    estimate, warnings = predict(bundle, values)
                    context.update(estimate=estimate, warnings=warnings)
            elif request.method == 'POST':
                raise ValueError('Prepare the model before predicting.')
        except (ValueError, OSError, KeyError, TypeError) as exc:
            context['error'] = str(exc)
        status = 400 if request.method == 'POST' and context['error'] else 200
        return render_template('page.html', **context), status

    @app.post('/train-prediction')
    def train_prediction():
        try:
            metadata = report()
            with training_lock:
                if memory_model(metadata) is None:
                    frame, features, _ = data()
                    runtime['bundle'] = fit_selected(metadata, frame, features)
            return redirect(url_for('dashboard'))
        except (ValueError, OSError, KeyError) as exc:
            return render_template('page.html', error=str(exc), stats=None, metadata=None, ready=False), 400

    # Old bookmarks open the same simple dashboard.
    @app.get('/<page>')
    def old_page(page):
        if page in ['analysis', 'models', 'predict', 'insights']:
            return redirect(url_for('dashboard'))
        return 'Not found', 404

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    return app


app = create_app()
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=False)
