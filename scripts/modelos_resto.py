"""Funciones compartidas para los puntos 7–12."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from modelos_puntos56 import ajustar, pronosticar


def calendario(index, semanal=True, tendencia=True):
    t=(index-pd.Timestamp('2023-01-01')).days.to_numpy(dtype=float)
    x={'constante':np.ones(len(t))}
    if tendencia: x['tendencia']=t/365.25
    for k in (1,2):
        x[f'seno_anual_{k}']=np.sin(2*np.pi*k*t/365.25)
        x[f'coseno_anual_{k}']=np.cos(2*np.pi*k*t/365.25)
    if semanal:
        for d in range(1,7): x[f'dia_{d}']=(index.dayofweek==d).astype(float)
    return pd.DataFrame(x,index=index)


def alternativa(train, fechas, tipo, nombre):
    if tipo=='Media': return pd.Series(train.mean(),index=fechas)
    if tipo=='Último valor': return pd.Series(train.iloc[-1],index=fechas)
    if tipo=='Ingenuo estacional':
        s=7 if nombre=='Viajes MBTB' else 365 if nombre in ('Temperatura','Humedad') else 1
        return pd.Series(np.resize(train.values[-s:],len(fechas)),index=fechas)
    if tipo=='ARIMA(1,1,1)':
        c={'order':(1,1,1),'seasonal_order':(0,0,0,0),'anual':False}
        r,_=ajustar(train,c)
        return pronosticar(r,train,c,fechas)
    if tipo=='Suavizado exponencial':
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r=ExponentialSmoothing(train,trend='add',damped_trend=True,
                seasonal='add' if nombre=='Viajes MBTB' else None,
                seasonal_periods=7 if nombre=='Viajes MBTB' else None,
                initialization_method='estimated').fit(optimized=True)
        if not r.mle_retvals.get('success',True): raise RuntimeError('Suavizado sin convergencia')
        return pd.Series(np.asarray(r.forecast(len(fechas))),index=fechas)
    raise ValueError(tipo)


def intervalo_sarima(train,c,h=30):
    assert h<=365, 'La reconstrucción de intervalos anuales usa rezagos observados.'
    r,avisos=ajustar(train,c)
    fechas=pd.date_range(train.index[-1]+pd.Timedelta(days=1),periods=h)
    fc=r.get_forecast(h)
    limites=np.asarray(fc.conf_int(alpha=.05))
    if c['anual']:
        base=train.reindex(fechas-pd.Timedelta(days=365)).values
        assert np.isfinite(base).all()
        limites=limites+base[:,None]
    return pd.DataFrame({'prediccion':pronosticar(r,train,c,fechas),
                         'inferior95':limites[:,0],'superior95':limites[:,1]}),r,avisos


def fit_var(panel,p=None):
    # El calendario se ajusta SOLO sobre los datos entregados a esta función.
    x=calendario(panel.index)
    beta=np.linalg.lstsq(x.values,panel.values,rcond=None)[0]
    resid=panel-pd.DataFrame(x.values@beta,index=panel.index,columns=panel.columns)
    transformado=resid.copy()
    transformado["Viajes MBTB"]=resid["Viajes MBTB"].diff()
    transformado=transformado.dropna()
    escala=transformado.std()
    z=transformado/escala
    modelo=VAR(z)
    criterios=modelo.select_order(maxlags=21,trend='c')
    tabla=pd.DataFrame(criterios.ics).rename_axis('rezago')
    # p>=1 permite estudiar relaciones dinámicas; se informa esta restricción.
    if p is None: p=int(tabla.loc[1:,'bic'].idxmin())
    r=modelo.fit(p,trend='c')
    return dict(resultado=r,beta=beta,escala=escala,residuos=transformado,nivel_residuos=resid,z=z,criterios=tabla)


def forecast_var(bundle,fechas):
    r=bundle['resultado']
    z=r.forecast(bundle['z'].values[-r.k_ar:],steps=len(fechas))
    pred=z*bundle['escala'].values
    j=list(bundle['z'].columns).index('Viajes MBTB')
    pred[:,j]=bundle['nivel_residuos']['Viajes MBTB'].iloc[-1]+np.cumsum(pred[:,j])
    pred=pred+calendario(fechas).values@bundle['beta']
    return pd.DataFrame(pred,index=fechas,columns=bundle['z'].columns)


def pruebas_estacionariedad(panel):
    filas=[]
    for nombre,s in panel.items():
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            a=adfuller(s,autolag='AIC'); k=kpss(s,regression='c',nlags='auto')
        filas.append({'serie':nombre,'ADF_p':a[1],'KPSS_p':k[1],
                      'compatible':a[1]<.05 and k[1]>=.05,
                      'advertencias':' | '.join(str(w.message) for w in avisos)})
    return pd.DataFrame(filas)


def fit_estacional(train,nombre,tipo):
    # Alternativa explícita: regresión de calendario con errores SARIMA (SARIMAX).
    x=calendario(train.index,semanal=False,tendencia=nombre=='Viajes MBTB').drop(columns='constante')
    if tipo=='SARIMA semanal + ciclo anual':
        order=(1,0,1); se=(1,0,0,7)
    else:
        order=(1,0,0); se=(0,0,0,0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r=SARIMAX(train,exog=x,order=order,seasonal_order=se,trend='c',
                   enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=500)
    if not r.mle_retvals.get('converged',False): raise RuntimeError('No convergió la alternativa estacional')
    return r,' | '.join(dict.fromkeys(str(a.message) for a in w))


def forecast_estacional(r,fechas,nombre):
    x=calendario(fechas,semanal=False,tendencia=nombre=='Viajes MBTB').drop(columns='constante')
    return pd.Series(np.asarray(r.get_forecast(len(fechas),exog=x).predicted_mean),index=fechas)
