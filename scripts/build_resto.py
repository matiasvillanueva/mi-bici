"""Genera los notebooks 7–12 sin modificar los puntos anteriores."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
LOAD='''from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data/bicicletas_rosario.csv').exists())
if str(ROOT/'scripts') not in sys.path: sys.path.insert(0,str(ROOT/'scripts'))
from modelos_puntos56 import cargar, ajustar, ajustados, pronosticar, metricas, etiqueta
from modelos_resto import *
series=cargar(ROOT)
seleccion=json.loads((ROOT/'puntos/punto5/seleccion.json').read_text())
plt.rcParams.update({'figure.figsize':(11,4),'figure.dpi':120})
'''
REF='''\n\n**Herramientas y referencias:** Pandas y NumPy para los datos; Matplotlib para gráficos; statsmodels para los modelos y pruebas. Statsmodels developers. (s. f.). [Documentación de series temporales](https://www.statsmodels.org/stable/tsa.html).'''
P={}
P[7]=[md('''# Punto 7 — Comparación con otros modelos

Se compara el SARIMA del punto 5 con alternativas más simples. Un modelo complejo resulta útil si mejora las predicciones de referencias sencillas.

- **Media:** predice siempre el promedio del entrenamiento.
- **Último valor:** repite el último dato observado.
- **Ingenuo estacional:** repite la última semana para viajes y los últimos 365 días para temperatura y humedad. En lluvia coincide con el último valor.
- **ARIMA(1,1,1):** usa una diferencia diaria, un término de valores anteriores y otro de errores anteriores.
- **Suavizado exponencial:** da más importancia a datos recientes. Incluye una tendencia que pierde fuerza con el tiempo y una parte semanal aditiva para viajes.

Se conservan las mismas fechas del punto 6: entrenamiento 2023–2024 y prueba 2025. Todas las predicciones parten del 31/12/2024, sin incorporar los datos reales de 2025. Se comparan MAE y RMSE anuales y de los primeros 30 días.

La clasificación por error de prueba describe lo ocurrido en 2025. **No se utiliza para volver a elegir el modelo del punto 5**. Si se tomara una decisión a partir de ella, haría falta otro período independiente para evaluarla.'''),code(LOAD),code('''OUT=ROOT/'puntos/punto7'; OUT.mkdir(exist_ok=True)
filas=[]; registros=[]
for nombre,s in series.items():
    train,test=s.loc[:'2024-12-31'],s.loc['2025-01-01':]
    for tipo in ['SARIMA punto 5','Media','Último valor','Ingenuo estacional','ARIMA(1,1,1)','Suavizado exponencial']:
        if tipo=='SARIMA punto 5':
            r,_=ajustar(train,seleccion[nombre]); pred=pronosticar(r,train,seleccion[nombre],test.index)
        else: pred=alternativa(train,test.index,tipo,nombre)
        for h in (30,365):
            filas.append(dict(serie=nombre,modelo=tipo,horizonte=h,**metricas(test.iloc[:h],pred.iloc[:h])))
        registros.append(pd.DataFrame({'fecha':test.index,'serie':nombre,'modelo':tipo,'observado':test.values,'prediccion':pred.values}))
comparacion=pd.DataFrame(filas)
comparacion.to_csv(OUT/'comparacion_modelos.csv',index=False)
pd.concat(registros).to_csv(OUT/'predicciones_modelos.csv',index=False)
for nombre in series:
    tabla=comparacion[(comparacion.serie==nombre)&(comparacion.horizonte==365)].sort_values('MAE')
    display(Markdown(f'### {nombre}'))
    display(tabla[['modelo','MAE','RMSE','sesgo']].round(3))
    mejor=tabla.iloc[0]
    sar=tabla[tabla.modelo=='SARIMA punto 5'].iloc[0]
    display(Markdown(f'El menor MAE anual corresponde a **{mejor.modelo}** ({mejor.MAE:.2f}). '
                     f'El SARIMA del punto 5 obtiene {sar.MAE:.2f}. Esta comparación no demuestra que el ganador sea siempre mejor.'))
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,nombre in zip(axes.flat,series):
    t=comparacion[(comparacion.serie==nombre)&(comparacion.horizonte==365)].sort_values('MAE')
    ax.barh(t.modelo,t.MAE,color='#2c6e91');ax.invert_yaxis();ax.set(title=nombre,xlabel='MAE anual')
fig.tight_layout();fig.savefig(OUT/'comparacion.png');plt.show();plt.close(fig)
'''),md('''## Conclusión

Las referencias simples permiten saber si el SARIMA aporta una mejora real para el período estudiado. Los resultados pueden cambiar con la duración del pronóstico y la época del año. El cuadro de 30 días se conserva en el CSV para esa comparación. En particular, los resultados anuales no representan una operación diaria en la que el modelo recibe observaciones nuevas.'''+REF)]
P[8]=[md('''# Punto 8 — Revisión de los errores del modelo

Un **residuo** es la diferencia entre un dato observado y el valor que el modelo esperaba para ese día. Un modelo adecuado debería dejar errores sin patrones claros.

Se revisan los modelos del punto 5 con el entrenamiento 2023–2024. Se descartan las primeras 30 predicciones disponibles, o más si lo requiere la inicialización. En temperatura, la diferencia anual también elimina los primeros 365 días.

Se presentan el gráfico de residuos, su autocorrelación (FAC), un histograma y un gráfico Q–Q. Este último compara la forma de los errores con una distribución normal.

**Ljung–Box** evalúa si queda relación entre errores separados por varios días. La hipótesis inicial es que no hay autocorrelación hasta el rezago indicado. Un p-valor menor a 0,05 sugiere que el modelo dejó un patrón sin explicar. Se descuentan los parámetros AR y MA, incluidos los estacionales.

También se usan **Jarque–Bera**, para revisar normalidad, y **ARCH-LM**, para revisar si el tamaño de los errores depende de errores anteriores. No rechazar una prueba no demuestra que su supuesto sea verdadero. Los p-valores son orientativos: se realizan varias pruebas sobre los mismos datos.'''),code(LOAD),code('''from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import jarque_bera
OUT=ROOT/'puntos/punto8';OUT.mkdir(exist_ok=True)
filas=[];residuos=[];otros=[]
for i,(nombre,s) in enumerate(series.items()):
    train=s.loc[:'2024-12-31'];c=seleccion[nombre];r,_=ajustar(train,c)
    pred=ajustados(r,train,c).iloc[max(30,int(r.loglikelihood_burn)):]
    e=train.reindex(pred.index)-pred
    dof=c['order'][0]+c['order'][2]+c['seasonal_order'][0]+c['seasonal_order'][2]
    lb=acorr_ljungbox(e,lags=[7,14,28],model_df=dof,return_df=True)
    lb['serie']=nombre;lb['rezago']=lb.index;lb['rechaza_5pct']=lb.lb_pvalue<.05
    filas.append(lb.reset_index(drop=True))
    jb=jarque_bera(e);arch=het_arch(e,nlags=7,ddof=dof)
    raices=np.concatenate([np.abs(r.arroots),np.abs(r.maroots)])
    otros.append({'serie':nombre,'n':len(e),'media_error':e.mean(),'JB_p':jb.pvalue,'ARCH_p':arch[1],
                  'raiz_minima':raices.min() if len(raices) else np.nan,
                  'parametro_cerca_limite':bool(len(raices) and raices.min()<1.01)})
    residuos.append(pd.DataFrame({'fecha':e.index,'serie':nombre,'residuo':e.values}))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(e,lw=.7);axes[0,0].axhline(0,color='black',lw=.5);axes[0,0].set_title('Residuos')
    plot_acf(e,lags=40,zero=False,ax=axes[0,1]);axes[0,1].set_title('FAC de residuos')
    axes[1,0].hist(e,bins=35,color='#2c6e91');axes[1,0].set_title('Distribución de errores')
    qqplot(e,line='s',ax=axes[1,1]);axes[1,1].set_title('Comparación con distribución normal')
    fig.suptitle(nombre);fig.tight_layout();fig.savefig(OUT/f'diagnostico_{i}.png');plt.show();plt.close(fig)
ljung=pd.concat(filas,ignore_index=True);diagnostico=pd.DataFrame(otros)
ljung.to_csv(OUT/'ljung_box.csv',index=False);diagnostico.to_csv(OUT/'diagnostico.csv',index=False)
pd.concat(residuos).to_csv(OUT/'residuos.csv',index=False)
display(ljung.round(5));display(diagnostico.round(5))
for row in diagnostico.itertuples():
    sub=ljung[ljung.serie==row.serie];rechazos=sub.loc[sub.rechaza_5pct,'rezago'].tolist()
    texto=f'Queda autocorrelación según Ljung–Box en los rezagos {rechazos}.' if rechazos else 'Ljung–Box no detecta autocorrelación al 5% en los rezagos evaluados.'
    display(Markdown(f'**{row.serie}:** {texto} '
        + ('Los parámetros están cerca del límite de estabilidad o invertibilidad; sus errores estándar requieren cautela. ' if row.parametro_cerca_limite else '')
        + ('Se rechaza normalidad. ' if row.JB_p<.05 else 'No se rechaza normalidad. ')
        + ('ARCH sugiere cambios predecibles en la variabilidad.' if row.ARCH_p<.05 else 'ARCH no detecta ese patrón al 5%.')))
'''),md('''## Alcance del diagnóstico

Un error promedio pequeño no basta para considerar correcto un modelo. Si quedan autocorrelaciones, puede faltar un ciclo o una relación temporal. Si la normalidad falla, los intervalos gaussianos pueden representar mal los extremos. El diagnóstico puede cuestionar el modelo aunque haya tenido el menor MAE de validación.

**Referencia:** Statsmodels developers. (s. f.). [Prueba de Ljung–Box](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html).''')]
P[9]=[md('''# Punto 9 — Pronóstico para los próximos 30 días

Se conservan los órdenes elegidos en el punto 5 y se recalculan sus parámetros con todos los datos disponibles, de 2023 a 2025. Se pronostica **del 1 al 30 de enero de 2026**. Son fechas posteriores al último dato del archivo, no al día en que se redactó el trabajo.

Se usa una ventana de 30 días porque permite observar varias semanas sin extender demasiado el plazo. Para clima y lluvia, estos resultados son una proyección estadística y no reemplazan un pronóstico meteorológico.

La línea central es el valor esperado. La banda del **95%** muestra la incertidumbre del modelo bajo sus supuestos. No es una garantía: puede ser poco confiable si los residuos no cumplen esos supuestos. La banda tampoco incluye toda la incertidumbre de haber elegido y estimado los parámetros.

En temperatura se suma el valor observado de 365 días antes tanto al pronóstico de la diferencia como a sus límites. Esto es válido para estos 30 días porque todos esos valores anteriores son conocidos. No se recortan resultados negativos ni valores de humedad fuera de 0–100%; se registran como limitaciones.'''),code(LOAD),code('''OUT=ROOT/'puntos/punto9';OUT.mkdir(exist_ok=True)
filas=[];alertas=[]
for i,(nombre,s) in enumerate(series.items()):
    fc,r,avisos=intervalo_sarima(s,seleccion[nombre],h=30)
    fc.index.name='fecha';fc['serie']=nombre;filas.append(fc.reset_index())
    invalidos=(fc.prediccion<0) if nombre in ('Viajes MBTB','Precipitación') else ((fc.prediccion<0)|(fc.prediccion>100)) if nombre=='Humedad' else pd.Series(False,index=fc.index)
    alertas.append({'serie':nombre,'predicciones_fuera_rango':int(invalidos.sum()),'advertencias':avisos})
    display(Markdown(f'### {nombre}'))
    display(fc.drop(columns='serie').round(2))
    fig,ax=plt.subplots();ax.plot(s.iloc[-60:],label='Observado',lw=1)
    ax.plot(fc.prediccion,label='Pronóstico',color='#bd5825');ax.fill_between(fc.index,fc.inferior95,fc.superior95,alpha=.2,color='#bd5825',label='Intervalo 95%')
    ax.set(title=nombre,xlabel='Fecha',ylabel=nombre);ax.legend();fig.tight_layout();fig.savefig(OUT/f'pronostico_{i}.png');plt.show();plt.close(fig)
pronosticos=pd.concat(filas,ignore_index=True)
pronosticos.to_csv(OUT/'pronostico_30_dias.csv',index=False)
pd.DataFrame(alertas).to_csv(OUT/'alertas.csv',index=False)
resumen=pronosticos.groupby('serie').agg(promedio=('prediccion','mean'),minimo=('prediccion','min'),maximo=('prediccion','max'))
display(resumen.round(2))
'''),md('''## Conclusión

El pronóstico aprovecha todos los datos disponibles, pero conserva las limitaciones del diagnóstico anterior. No hay valores reales de enero de 2026 en los archivos para calcular su error. El desempeño conocido sigue siendo el de la prueba de 2025; recalcular el modelo con más datos no garantiza una mejora.'''+REF)]
P[10]=[md(r'''# Punto 10 — Modelo conjunto VAR

Un **VAR** predice cada serie a partir de valores anteriores de todas las series. Así, los viajes pueden depender de viajes anteriores y también del clima de días previos. Se incluyen las cuatro variables.

Su forma es $z_t=c+A_1z_{t-1}+\cdots+A_pz_{t-p}+u_t$. El vector $z_t$ contiene las cuatro series; las matrices $A$ reúnen sus relaciones; **p** es la cantidad de días anteriores utilizados.

## Preparación

El VAR necesita series con un comportamiento relativamente estable. Antes de estimarlo, se retira de cada serie una parte de calendario: tendencia lineal, dos pares de senos y cosenos anuales y seis indicadores de día de la semana. Estos términos representan cambios suaves del año y diferencias entre días. Se estiman **solo con entrenamiento**. En viajes, KPSS todavía rechazó estacionariedad después de retirar el calendario (p≈0,024). Por eso se calcula una diferencia diaria adicional de sus residuos. Para las tres series climáticas se conservan los residuos sin diferencias. Luego se divide cada variable por su desvío para trabajar en escalas comparables.

Se aplican ADF y KPSS sobre esos residuos. ADF parte de la hipótesis de raíz unitaria; KPSS, de estacionariedad. La coincidencia de ambas pruebas respalda, pero no demuestra, la transformación. Sus p-valores son orientativos porque se aplican a residuos de un ajuste previo. No se aplica una diferencia anual automática: el punto 4 no la justificó para todas las series.

## Selección y evaluación

Se usa BIC para elegir entre **1 y 21 rezagos**. Se muestra también el rezago cero como referencia, pero se exige al menos uno para estudiar relaciones dinámicas. Los criterios se calculan con una ventana común. Luego se estima el orden elegido con todo el entrenamiento.

Se verifica estabilidad y se aplica una prueba conjunta de autocorrelación de residuos. La predicción de 2025 parte de diciembre de 2024; se vuelve a la escala original, se acumulan las diferencias de viajes desde el último residuo observado y se suma el calendario esperado. Ningún dato real de prueba entra en el pronóstico. También se ajusta el mismo orden con 2023–2025 para pronosticar enero de 2026.

Este procedimiento es una regresión de calendario seguida de un VAR de sus residuos. No es un VAR aplicado directamente a las series originales ni un modelo de cointegración.'''),code(LOAD),code('''OUT=ROOT/'puntos/punto10';OUT.mkdir(exist_ok=True)
orden=['Temperatura','Humedad','Precipitación','Viajes MBTB']
panel=pd.DataFrame(series)[orden];train=panel.loc[:'2024-12-31'];test=panel.loc['2025-01-01':]
bundle=fit_var(train);r=bundle['resultado']
pruebas=pruebas_estacionariedad(bundle['residuos']);pruebas.to_csv(OUT/'estacionariedad_var.csv',index=False)
bundle['criterios'].to_csv(OUT/'seleccion_rezagos.csv')
display(pruebas[['serie','ADF_p','KPSS_p','compatible']]);display(bundle['criterios'].round(3))
wh=r.test_whiteness(nlags=28,adjusted=True)
info={'rezagos':r.k_ar,'estable':bool(r.is_stable()),'blancura_p':float(wh.pvalue),
      'estacionariedad_compatible':bool(pruebas.compatible.all()),'n':int(r.nobs)}
(OUT/'resumen_var.json').write_text(json.dumps(info,indent=2))
(OUT/'resumen_estimacion.txt').write_text(str(r.summary()))
r.params.to_csv(OUT/'coeficientes_var.csv');r.pvalues.to_csv(OUT/'pvalores_var.csv')
pred=forecast_var(bundle,test.index);filas=[]
for nombre in orden:
    for h in (30,365): filas.append(dict(serie=nombre,horizonte=h,**metricas(test[nombre].iloc[:h],pred[nombre].iloc[:h])))
metricas_var=pd.DataFrame(filas);metricas_var.to_csv(OUT/'metricas_var.csv',index=False)
pred.rename_axis('fecha').to_csv(OUT/'predicciones_testing.csv')
comparacion=pd.read_csv(ROOT/'puntos/punto6/metricas_training_testing.csv')
comparacion=comparacion[comparacion.conjunto=='Testing (365 días)'][['serie','MAE','RMSE']]
comparacion=comparacion.merge(metricas_var[metricas_var.horizonte==365][['serie','MAE','RMSE']],on='serie',suffixes=('_SARIMA','_VAR'))
comparacion.to_csv(OUT/'comparacion_var_sarima.csv',index=False);display(comparacion.round(3))
display(Markdown(f'**Se eligió VAR({r.k_ar}).** Estable: **{r.is_stable()}**. '
                 f'La prueba conjunta de autocorrelación tiene p={wh.pvalue:.4g}. '
                 + ('Quedan relaciones entre residuos que el modelo no explicó.' if wh.pvalue<.05 else 'No se detecta autocorrelación conjunta al 5% en esta prueba.')))
fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
for ax,nombre in zip(axes,orden):
    ax.plot(test[nombre],lw=.7,label='Real');ax.plot(pred[nombre],lw=1,label='VAR');ax.set_ylabel(nombre);ax.legend()
fig.tight_layout();fig.savefig(OUT/'var_testing.png');plt.show();plt.close(fig)
final=fit_var(panel,p=r.k_ar)
fechas=pd.date_range('2026-01-01',periods=30)
futuro=forecast_var(final,fechas);futuro.rename_axis('fecha').to_csv(OUT/'var_30_dias.csv')
(OUT/'estabilidad_final.json').write_text(json.dumps({'estable':bool(final['resultado'].is_stable())}))
display(futuro.round(2))
'''),md('''## Interpretación

Un VAR puede aprovechar información compartida, pero tiene muchos más parámetros. Por eso no necesariamente supera a los modelos por separado. La tabla compara exactamente las mismas fechas y el mismo tipo de pronóstico. Si las pruebas rechazan los supuestos, sus resultados deben tratarse como exploratorios.

La parte de calendario permite representar ciclos sin usar el clima real futuro. Supone que esos patrones continúan; un cambio del servicio o del comportamiento de las personas podría volverlos poco adecuados.

**Referencia:** Statsmodels developers. (s. f.). [Vector autoregressions](https://www.statsmodels.org/stable/vector_ar.html).''')]
P[11]=[md('''# Punto 11 — Impulso-respuesta y relaciones predictivas

Se estudia el VAR del punto 10, ajustado con 2023–2024. El calendario ya fue retirado. En viajes se usa además la diferencia diaria de los residuos. Por eso, su respuesta muestra el cambio del componente no explicado por calendario; las variables climáticas representan desvíos respecto del calendario. Para obtener el efecto sobre el nivel de viajes se acumulan sus respuestas diarias.

## Impulso-respuesta

La función impulso-respuesta muestra cómo cambia una variable en los días posteriores a una sorpresa en otra. Se utiliza un shock de un desvío estándar, separado mediante la descomposición de **Cholesky**. Se adopta el orden temperatura, humedad, precipitación y viajes: el clima puede relacionarse con los viajes del mismo día, mientras que los viajes se ubican al final.

Este orden es una suposición, no una conclusión demostrada. El orden interno del clima también afecta los resultados. Se compara con el orden inverso como ejercicio de sensibilidad. Los gráficos muestran respuestas en unidades estandarizadas y bandas aproximadas del 95%, condicionadas al modelo. Si una banda incluye cero, no hay una respuesta claramente distinta de cero con ese criterio.

## Causalidad de Granger

Se evalúa si agregar valores anteriores de una serie ayuda a explicar otra después de tener en cuenta las demás. La hipótesis inicial es que esos valores no agregan información. Se realizan pruebas **F y Wald** para las 12 direcciones posibles, y una prueba conjunta del clima hacia viajes. Se ajustan los p-valores de cada familia con **Holm** para limitar los falsos hallazgos por probar muchas relaciones.

También se evalúa la relación contemporánea de cada variable con las demás mediante una prueba de causalidad instantánea. No indica una dirección del efecto.

**Una relación de Granger no demuestra causalidad real.** Puede reflejar variables omitidas, patrones compartidos o una especificación incompleta. Si el VAR deja autocorrelación en sus errores, las pruebas y bandas requieren todavía más cautela.'''),code(LOAD),code('''from statsmodels.stats.multitest import multipletests
OUT=ROOT/'puntos/punto11';OUT.mkdir(exist_ok=True)
orden=['Temperatura','Humedad','Precipitación','Viajes MBTB']
panel=pd.DataFrame(series)[orden].loc[:'2024-12-31']
p=json.loads((ROOT/'puntos/punto10/resumen_var.json').read_text())['rezagos']
bundle=fit_var(panel,p=p);r=bundle['resultado']
if not r.is_stable(): raise RuntimeError('El VAR no es estable: no se interpreta su impulso-respuesta.')
filas=[]
for destino in orden:
    for origen in orden:
        if origen==destino:continue
        f=r.test_causality(destino,[origen],kind='f');w=r.test_causality(destino,[origen],kind='wald')
        filas.append({'origen':origen,'destino':destino,'F':f.test_statistic,'F_p':f.pvalue,'Wald_p':w.pvalue})
f=r.test_causality('Viajes MBTB',orden[:3],kind='f');w=r.test_causality('Viajes MBTB',orden[:3],kind='wald')
filas.append({'origen':'Clima conjunto','destino':'Viajes MBTB','F':f.test_statistic,'F_p':f.pvalue,'Wald_p':w.pvalue})
g=pd.DataFrame(filas)
for col in ['F_p','Wald_p']:
    rechazo,ajustado,_,_=multipletests(g[col],method='holm');g[col+'_Holm']=ajustado;g[col+'_rechaza']=rechazo
g.to_csv(OUT/'granger.csv',index=False);display(g.round(5))
instant=[]
for nombre in orden:
    q=r.test_inst_causality([nombre]);instant.append({'serie':nombre,'p_valor':q.pvalue})
instant=pd.DataFrame(instant);instant['p_Holm']=multipletests(instant.p_valor,method='holm')[1]
instant.to_csv(OUT/'instantanea.csv',index=False);display(instant.round(5))
irf=r.irf(14)
fig=irf.plot(orth=True,plot_stderr=True,stderr_type='asym',figsize=(13,11))
fig.suptitle('Respuestas a shocks: escala estandarizada',y=1.01)
fig.savefig(OUT/'impulso_respuesta.png',bbox_inches='tight');plt.show();plt.close(fig)
filas_irf=[]
errores_irf=irf.stderr(orth=True)
for h in range(15):
    for i,destino in enumerate(orden):
        for j,origen in enumerate(orden):
            filas_irf.append({'dia':h,'origen':origen,'destino':destino,
                'respuesta_estandarizada':irf.orth_irfs[h,i,j],
                'inferior95':irf.orth_irfs[h,i,j]-1.96*errores_irf[h,i,j],
                'superior95':irf.orth_irfs[h,i,j]+1.96*errores_irf[h,i,j],
                'respuesta_unidad_original':irf.orth_irfs[h,i,j]*bundle['escala'][destino]})
tabla_irf=pd.DataFrame(filas_irf)
mask=tabla_irf.destino.eq('Viajes MBTB')
tabla_irf.loc[mask,'respuesta_nivel_viajes']=tabla_irf[mask].groupby('origen').respuesta_unidad_original.cumsum()
tabla_irf.to_csv(OUT/'impulso_respuesta.csv',index=False)
inverso=orden[::-1];br=fit_var(panel[inverso],p=p);ir2=br['resultado'].irf(14)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,clima in zip(axes,orden[:3]):
    a=irf.orth_irfs[:,orden.index('Viajes MBTB'),orden.index(clima)]*bundle['escala']['Viajes MBTB']
    b=ir2.orth_irfs[:,inverso.index('Viajes MBTB'),inverso.index(clima)]*br['escala']['Viajes MBTB']
    ax.plot(a,label='Clima primero');ax.plot(b,label='Orden inverso');ax.axhline(0,color='black',lw=.5)
    ax.set(title=clima,xlabel='Días después del shock',ylabel='Cambio del residuo de viajes');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'sensibilidad_orden.png');plt.show();plt.close(fig)
hallazgos=g[g.F_p_rechaza]
if hallazgos.empty: display(Markdown('Ninguna relación supera el ajuste de Holm al 5% en las pruebas F.'))
else:
    display(Markdown('Relaciones predictivas detectadas con F y ajuste de Holm: '+ '; '.join(hallazgos.origen+' → '+hallazgos.destino)+'.'))
# Resumen descriptivo; no convierte el máximo observado en una prueba adicional.
for clima in orden[:3]:
    sub=tabla_irf[(tabla_irf.origen==clima)&(tabla_irf.destino=='Viajes MBTB')]
    dias=sub.loc[(sub.inferior95>0)|(sub.superior95<0),'dia'].tolist()
    display(Markdown(f'Para **{clima}**, la banda puntual del 95% de la respuesta del cambio diario de viajes excluye cero en los días **{dias}**. Estas bandas no están ajustadas por mirar varios días a la vez.'))
    valores=np.cumsum(irf.orth_irfs[:,3,orden.index(clima)]*bundle['escala']['Viajes MBTB'])
    h=int(np.argmax(np.abs(valores)))
    display(Markdown(f'Para un shock de **{clima}**, la mayor respuesta acumulada absoluta del nivel de viajes dentro de 14 días '
                     f'es **{valores[h]:.2f} viajes**, en el día **{h}**, bajo el orden adoptado.'))
'''),md('''## Conclusión

Las pruebas describen información predictiva dentro del modelo. No permiten afirmar que intervenir sobre el clima produciría el cambio mostrado en los viajes. Las respuestas dependen del orden de Cholesky, de los rezagos elegidos y de la calidad del ajuste. Las bandas no incluyen la incertidumbre de estimar previamente el calendario. Las pruebas F y Wald contrastan la misma hipótesis mediante aproximaciones diferentes; no son dos confirmaciones independientes.

**Referencia:** Statsmodels developers. (s. f.). [Vector autoregressions: impulso-respuesta y pruebas](https://www.statsmodels.org/stable/vector_ar.html).''')]
P[12]=[md('''# Punto 12 — Revisión de la estacionalidad

Se revisa si representar mejor los ciclos mejora las predicciones. La referencia disponible es el modelo elegido en el punto 5. No hay otro trabajo anterior en las carpetas; por eso, la comparación se realiza con esa referencia.

En viajes se distinguen ciclos semanales y anuales. En temperatura y humedad predomina el ciclo anual. La precipitación se mantiene como referencia: no se impone una diferencia estacional sin respaldo.

Se agrega una alternativa con **dos pares de senos y cosenos anuales**, que dibujan un patrón suave durante el año. Para viajes se incluye además una tendencia y errores SARIMA(1,0,1)(1,0,0)₇. Para temperatura y humedad se usan errores AR(1). Estas alternativas son **regresiones de calendario con errores SARIMA (SARIMAX)**: utilizan datos de fechas, no valores climáticos futuros.

La comparación usa los mismos últimos 90 días de entrenamiento del punto 5. El modelo con menor MAE de validación se recalcula con 2023–2024 y se evalúa en 2025. La prueba no decide el ganador. La referencia ya fue elegida con esa validación, por lo que seguir probando alternativas en la misma ventana puede favorecer un ajuste excesivo a ella.

Esta comparación amplía el conjunto de opciones. No demuestra que el ganador sea la única representación correcta de la estacionalidad.'''),code(LOAD),code('''OUT=ROOT/'puntos/punto12';OUT.mkdir(exist_ok=True)
filas=[];elecciones={};pronosticos=[]
for nombre,s in series.items():
    train,test=s.loc[:'2024-12-31'],s.loc['2025-01-01':]
    base,val=train.iloc[:-90],train.iloc[-90:]
    opciones=['SARIMA punto 5']
    if nombre!='Precipitación': opciones.append('SARIMA semanal + ciclo anual' if nombre=='Viajes MBTB' else 'Ciclo anual + AR(1)')
    locales=[]
    for tipo in opciones:
        if tipo=='SARIMA punto 5':
            rv,_=ajustar(base,seleccion[nombre]);pv=pronosticar(rv,base,seleccion[nombre],val.index)
            rt,avisos=ajustar(train,seleccion[nombre]);pt=pronosticar(rt,train,seleccion[nombre],test.index)
        else:
            rv,_=fit_estacional(base,nombre,tipo);pv=forecast_estacional(rv,val.index,nombre)
            rt,avisos=fit_estacional(train,nombre,tipo);pt=forecast_estacional(rt,test.index,nombre)
        locales.append({'serie':nombre,'modelo':tipo,'MAE_validacion':metricas(val,pv)['MAE'],
                        'MAE_testing':metricas(test,pt)['MAE'],'RMSE_testing':metricas(test,pt)['RMSE'],'advertencias':avisos})
        pronosticos.append(pd.DataFrame({'fecha':test.index,'serie':nombre,'modelo':tipo,'prediccion':pt.values,'observado':test.values}))
    ganador=min(locales,key=lambda x:x['MAE_validacion'])['modelo'];elecciones[nombre]=ganador
    for row in locales:row['elegido_validacion']=row['modelo']==ganador
    filas+=locales
    display(Markdown(f'**{nombre}:** se elige **{ganador}** por validación.'))
tabla=pd.DataFrame(filas);tabla.to_csv(OUT/'comparacion_estacionalidad.csv',index=False)
(OUT/'seleccion_estacional.json').write_text(json.dumps(elecciones,ensure_ascii=False,indent=2))
pred=pd.concat(pronosticos,ignore_index=True);pred.to_csv(OUT/'predicciones_estacionales.csv',index=False)
display(tabla.round(3))
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
for ax,nombre in zip(axes,['Viajes MBTB','Temperatura','Humedad']):
    sub=pred[pred.serie==nombre]
    real=sub[sub.modelo=='SARIMA punto 5'];ax.plot(real.fecha,real.observado,color='gray',alpha=.5,lw=.7,label='Observado')
    for tipo,g in sub.groupby('modelo'):ax.plot(g.fecha,g.prediccion,lw=1,label=tipo)
    ax.set_ylabel(nombre);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'estacionalidad.png');plt.show();plt.close(fig)
for nombre,g in tabla.groupby('serie'):
    elegido=g[g.elegido_validacion].iloc[0];base=g[g.modelo=='SARIMA punto 5'].iloc[0]
    cambio=100*(base.MAE_testing-elegido.MAE_testing)/base.MAE_testing
    display(Markdown(f'**{nombre}:** el cambio de MAE anual respecto de la referencia es **{cambio:.1f}%** '
                     '(positivo significa mejora; negativo, empeoramiento). La elección se tomó con validación, antes de esta comparación.'))
'''),md('''## Conclusión

Una diferencia estacional y un patrón de calendario cumplen funciones distintas: la primera resta ciclos anteriores; el segundo representa una forma que se repite. Los resultados muestran cuál funcionó mejor dentro de las alternativas y fechas estudiadas.

Los pronósticos del punto 9 conservan los modelos originalmente seleccionados para mantener la continuidad del trabajo. Las nuevas alternativas se presentan aquí como revisión y necesitarían su propio diagnóstico completo antes de reemplazarlos. El trabajo no identifica relaciones causales ni prueba que los ciclos se mantengan sin cambios.'''+REF)]
for n,cells in P.items():
    path=ROOT/f'puntos/punto{n}/punto{n}.ipynb';path.parent.mkdir(exist_ok=True)
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(nb,path)
indice=ROOT/'puntos/00_indice.ipynb';texto=indice.read_text()
for n in range(7,13):texto=texto.replace(f'`punto{n}/`',f'[punto{n}/punto{n}.ipynb](punto{n}/punto{n}.ipynb)')
indice.write_text(texto)
