"""Interactive chart specifications calculated from real rows and evaluation reports."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, abort, current_app, request
from project import VERSION, fingerprint, load_data, TARGET, NUMERIC, model_label

charts = Blueprint('charts', __name__)
COLORS = ['#248b83', '#6797c1', '#dba14c']


@lru_cache(maxsize=2)
def chart_data(path, stamp):
    frame, _ = load_data(path)
    return frame, fingerprint(path)


@charts.get('/api/chart/<name>')
def graph(name):
    try:
        path = Path(current_app.config['DATA_PATH'])
        data, digest = chart_data(str(path), path.stat().st_mtime_ns)
        directory = Path(current_app.config['REPORTS_DIR'])
        weather = request.args.get('weather', 'All')
        if weather != 'All':
            if 'Weather' not in data or weather not in data.Weather.unique():
                raise ValueError('Unknown weather filter')
            data = data[data.Weather == weather]
        traces = []
        layout = {'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
                  'font': {'family': 'system-ui, sans-serif', 'color': '#294651', 'size': 12},
                  'margin': {'l': 65, 'r': 25, 't': 28, 'b': 75},
                  'showlegend': False, 'hovermode': 'closest', 'dragmode': 'zoom',
                  'xaxis': {'automargin': True, 'gridcolor': '#e5ecee', 'zeroline': False},
                  'yaxis': {'automargin': True, 'gridcolor': '#e5ecee', 'zerolinecolor': '#d4e0e2'},
                  'legend': {'orientation': 'h', 'y': 1.15, 'x': 0}, 'bargap': .35}
        xlabel, ylabel = '', ''
        if name in ['mae','rmse','r2','generalization','actual','residual','importance']:
            metadata = json.loads((directory / 'metrics.json').read_text())
            if metadata['dataset_sha256'] != digest or metadata['version'] != VERSION:
                raise ValueError('Stale evaluation results')
            rows = metadata['models']
            model_names = [model_label(row['model']) for row in rows]
            if name in ['mae','rmse','r2']:
                metric = {'mae':'mae_minutes','rmse':'rmse_minutes','r2':'r2'}[name]
                values = [row['test'][metric] for row in rows]
                traces = [{'type':'bar','x':model_names,'y':values,'marker':{'color':COLORS},
                           'text':[f'{value:.3f}' if name == 'r2' else f'{value:.2f}' for value in values],
                           'textposition':'outside','cliponaxis':False,
                           'hovertemplate':'%{x}<br>' + ('R²' if name=='r2' else name.upper()+' (min)') + ': %{y:.3f}<extra></extra>'}]
                xlabel, ylabel = 'Regressor', 'R²' if name=='r2' else 'Error (minutes)'
                if min(values) >= 0:
                    layout['yaxis']['range'] = [0, max(values) * 1.2 or 1]
            elif name == 'generalization':
                for label, partition, color in [('Training','training',COLORS[1]),('Held-out test','test',COLORS[0])]:
                    traces.append({'type':'bar','x':model_names,'y':[row[partition]['mae_minutes'] for row in rows],
                                   'name':label,'marker':{'color':color},
                                   'hovertemplate':'%{x}<br>MAE: %{y:.2f} min<extra>'+label+'</extra>'})
                layout.update(barmode='group',showlegend=True)
                xlabel, ylabel = 'Regressor','MAE (minutes)'
            elif name in ['actual','residual']:
                predictions = pd.read_csv(directory/'test_predictions.csv')
                if name == 'actual':
                    sample = predictions.sample(min(len(predictions),1500),random_state=42)
                    traces = [{'type':'scatter','mode':'markers','x':sample.actual_minutes.tolist(),
                               'y':sample.predicted_minutes.tolist(),'name':'Held-out orders',
                               'marker':{'color':COLORS[0],'size':6,'opacity':.5},
                               'hovertemplate':'Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'}]
                    bounds = [float(min(sample.actual_minutes.min(),sample.predicted_minutes.min())),
                              float(max(sample.actual_minutes.max(),sample.predicted_minutes.max()))]
                    traces.append({'type':'scatter','mode':'lines','x':bounds,'y':bounds,'name':'Perfect estimate',
                                   'line':{'color':COLORS[2],'dash':'dash'},'hoverinfo':'skip'})
                    layout['showlegend']=True
                    xlabel,ylabel='Actual time (minutes)','Predicted time (minutes)'
                else:
                    traces = histogram(predictions.residual_minutes,35)
                    layout['shapes']=[{'type':'line','x0':0,'x1':0,'yref':'paper','y0':0,'y1':1,
                                       'line':{'dash':'dash','color':COLORS[2]}}]
                    xlabel,ylabel='Actual − predicted (minutes)','Orders'
            else:
                importance=pd.read_csv(directory/'feature_importance.csv').head(8).iloc[::-1]
                traces=[{'type':'bar','orientation':'h','x':importance.importance.tolist(),
                         'y':importance.feature.str.replace('_',' ').tolist(),'marker':{'color':COLORS[0]},
                         'hovertemplate':'%{y}<br>Importance: %{x:.4f}<extra></extra>'}]
                layout['margin']['l']=135
                xlabel,ylabel='Random Forest Regressor impurity importance',''
        elif name == 'distribution':
            traces=histogram(data[TARGET],30)
            xlabel,ylabel='Delivery duration (minutes)','Orders'
        elif name in ['traffic','weather']:
            column='Traffic_Level' if name=='traffic' else 'Weather'
            grouped=data.groupby(column)[TARGET].agg(['mean','count']).sort_values('mean')
            traces=[{'type':'bar','x':grouped.index.tolist(),'y':grouped['mean'].tolist(),
                     'customdata':grouped['count'].tolist(),'marker':{'color':COLORS[0]},
                     'text':[f'{v:.1f}' for v in grouped['mean']], 'textposition':'outside','cliponaxis':False,
                     'hovertemplate':'%{x}<br>Mean: %{y:.1f} min<br>Orders: %{customdata:,}<extra></extra>'}]
            layout['yaxis']['range']=[0,float(grouped['mean'].max())*1.2]
            xlabel,ylabel='Traffic level' if name=='traffic' else 'Weather','Mean delivery time (minutes)'
        elif name == 'hourly':
            grouped=data.groupby('Order_Hour')[TARGET].agg(['mean','count'])
            traces=[{'type':'scatter','mode':'lines+markers','x':grouped.index.tolist(),'y':grouped['mean'].tolist(),
                     'customdata':grouped['count'].tolist(),'line':{'color':COLORS[0]},'marker':{'size':7},
                     'hovertemplate':'Hour: %{x}:00<br>Mean: %{y:.1f} min<br>Orders: %{customdata:,}<extra></extra>'}]
            xlabel,ylabel='Order hour','Mean delivery time (minutes)'
        elif name == 'distance':
            sample=data.sample(min(len(data),1500),random_state=42)
            traces=[{'type':'scatter','mode':'markers','x':sample.Road_Distance_km.tolist(),'y':sample[TARGET].tolist(),
                     'marker':{'color':COLORS[0],'size':6,'opacity':.45},
                     'hovertemplate':'Distance: %{x:.2f} km<br>Duration: %{y:.1f} min<extra></extra>'}]
            xlabel,ylabel='Road distance (km)','Delivery duration (minutes)'
        elif name == 'correlation':
            columns=[c for c in NUMERIC+[TARGET] if c in data]
            corr=data[columns].corr()
            matrix=corr.astype(object).where(pd.notna(corr),None).values.tolist()
            labels=[c.replace('_',' ') for c in columns]
            traces=[{'type':'heatmap','x':labels,'y':labels,'z':matrix,'zmin':-1,'zmax':1,
                     'colorscale':'BrBG','colorbar':{'title':{'text':'r'},'thickness':12},
                     'hovertemplate':'%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'}]
            layout['margin'].update(l=160,b=140)
            layout['xaxis']['tickangle']=-40
        else:
            raise ValueError('Unknown chart')
        layout['xaxis']['title']={'text':xlabel,'standoff':15}
        layout['yaxis']['title']={'text':ylabel,'standoff':10}
        return jsonify({'data':traces,'layout':layout})
    except (ValueError,OSError,KeyError,TypeError):
        abort(400,'Chart unavailable. Check the dataset and evaluation reports.')


def histogram(series,bins):
    counts,edges=np.histogram(series.dropna().to_numpy(),bins=bins)
    return [{'type':'bar','x':((edges[:-1]+edges[1:])/2).tolist(),'y':counts.tolist(),
             'width':(np.diff(edges)*.94).tolist(),'marker':{'color':COLORS[0]},
             'customdata':np.column_stack([edges[:-1],edges[1:]]).tolist(),
             'hovertemplate':'%{customdata[0]:.1f}–%{customdata[1]:.1f} min<br>Orders: %{y:,}<extra></extra>'}]
