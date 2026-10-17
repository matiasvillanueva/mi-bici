"""Funciones reproducibles para estimar SARIMA y evaluar los puntos 5 y 6."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

CONFIG = [('Viajes MBTB', 'bicicletas_rosario.csv', 'viajes', 7),
          ('Temperatura', 'temperatura_rosario.csv', 't2m_c', 365),
          ('Humedad', 'humedad_rosario.csv', 'rh2m_pct', 365),
          ('Precipitación', 'precipitacion_rosario.csv', 'precip_mm', 0)]

def cargar(root):
    series = {}
    for nombre, archivo, col, periodo in CONFIG:
        df = pd.read_csv(Path(root) / 'data' / archivo, parse_dates=['fecha'])
        assert not df.fecha.duplicated().any()
        s = df.set_index('fecha')[col].sort_index().asfreq('D').astype(float)
        assert s.notna().all() and np.isfinite(s).all()
        series[nombre] = s
    assert all(s.index.equals(next(iter(series.values())).index) for s in series.values())
    return series

def candidatos(nombre):
    ordenes = [(0,0,0), (1,0,0), (0,0,1), (1,0,1)]
    if nombre == 'Viajes MBTB':
        return [dict(order=o, seasonal_order=se, anual=False)
                for o in [(1,0,0),(0,1,1),(1,1,1)]
                for se in [(1,0,0,7),(0,1,1,7),(1,0,1,7),(0,0,0,0)]]
    if nombre in ('Temperatura','Humedad'):
        return [dict(order=o, seasonal_order=(0,0,0,0), anual=a)
                for a in (False, True) for o in ordenes]
    return [dict(order=o, seasonal_order=(0,0,0,0), anual=False) for o in ordenes]

def etiqueta(c):
    se = (0,1,0,365) if c['anual'] else tuple(c['seasonal_order'])
    return f"SARIMA{tuple(c['order'])}×{se}"

def ajustar(y, c):
    # La diferencia anual externa evita un vector de estado de 365 componentes.
    z = y.diff(365).dropna() if c['anual'] else y
    tendencia = 'c' if c['order'][1] == 0 and c['seasonal_order'][1] == 0 and not c['anual'] else 'n'
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        r = SARIMAX(z, order=tuple(c['order']), seasonal_order=tuple(c['seasonal_order']),
                    trend=tendencia, enforce_stationarity=True,
                    enforce_invertibility=True).fit(disp=False, maxiter=300)
    if not r.mle_retvals.get('converged', False) or not np.isfinite(r.aic):
        raise RuntimeError('Ajuste sin convergencia o AIC no finito')
    return r, ' | '.join(dict.fromkeys(str(w.message) for w in avisos))

def pronosticar(r, historia, c, fechas):
    z = np.asarray(r.get_forecast(steps=len(fechas)).predicted_mean)
    if c['anual']:
        valores = list(historia.values)
        for delta in z:
            valores.append(float(delta + valores[-365]))
        z = np.array(valores[-len(fechas):])
    return pd.Series(z, index=fechas)

def ajustados(r, y, c):
    p = r.fittedvalues.copy()
    if c['anual']:
        p = p + y.shift(365).reindex(p.index)
    return p

def metricas(real, pred):
    assert real.index.equals(pred.index) and pred.notna().all()
    error = real - pred
    return {'n': len(real), 'MAE': np.abs(error).mean(),
            'RMSE': np.sqrt(np.mean(error ** 2)), 'sesgo': error.mean()}

def seleccionar(train, nombre):
    base, validacion = train.iloc[:-90], train.iloc[-90:]
    filas = []
    configs = candidatos(nombre)
    for i, c in enumerate(configs):
        fila = dict(id=i, modelo=etiqueta(c), estado='ok',
                    familia='diferencia anual' if c['anual'] else 'serie original')
        try:
            r, avisos = ajustar(base, c)
            pred = pronosticar(r, base, c, validacion.index)
            fila.update(MAE_validacion=metricas(validacion, pred)['MAE'],
                        AIC=r.aic, BIC=r.bic, n_ajuste=int(r.nobs), advertencias=avisos)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            fila.update(estado='descartado', advertencias=str(exc))
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    validos = tabla[tabla.estado.eq('ok')].sort_values('MAE_validacion')
    if validos.empty:
        raise RuntimeError(f'No hay candidatos válidos para {nombre}')
    # Si falla la reestimación, se intenta el siguiente candidato según validación.
    for idx in validos.id:
        try:
            c = configs[int(idx)]
            r, avisos = ajustar(train, c)
            tabla.loc[tabla.id.eq(idx), 'seleccionado'] = True
            return c, r, tabla, avisos
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            tabla.loc[tabla.id.eq(idx), 'advertencias'] += ' | Reestimación: ' + str(exc)
    raise RuntimeError(f'No convergió ninguna reestimación para {nombre}')
