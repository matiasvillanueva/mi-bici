"""Genera los notebooks de los puntos 5 y 6. Ejecutar luego en orden."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
LOAD = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/bicicletas_rosario.csv').exists())
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from modelos_puntos56 import cargar, seleccionar, ajustar, ajustados, pronosticar, metricas, etiqueta
series = cargar(ROOT)
plt.rcParams.update({'figure.figsize': (12, 4), 'figure.dpi': 110})
'''
p5 = [md(r'''# Punto 5 — Comparación y elección de modelos SARIMA

En este punto se prueban distintos modelos para las cuatro series diarias de Rosario. Se busca elegir, para cada serie, el modelo que cometa menos errores al predecir datos que no utilizó para calcular sus parámetros.

## ¿Qué es un modelo SARIMA?

Un modelo **SARIMA** utiliza valores y errores anteriores de una serie para predecir sus próximos valores. También puede representar patrones que se repiten, como el ciclo semanal de los viajes en bicicleta.

Se escribe **SARIMA(p,d,q)(P,D,Q)ₛ**. Sus letras indican:

- **p:** cuántos valores anteriores se tienen en cuenta.
- **d:** cuántas veces se calcula la diferencia entre un valor y el anterior.
- **q:** cuántos errores anteriores se tienen en cuenta.
- **P, D y Q:** cumplen funciones similares, pero para el ciclo estacional.
- **s:** duración del ciclo. Por ejemplo, **s=7** representa una semana en datos diarios.

Un modelo sin parte estacional es un caso particular de esta familia y se denomina ARIMA. Si tampoco necesita diferencias, se denomina ARMA.

## ¿Cómo se elige el modelo?

Se separan los datos por fecha:

1. **Entrenamiento (Training Set): 2023–2024.** Se usa para calcular los parámetros y elegir el modelo.
2. **Prueba (Testing Set): 2025.** Se reserva para evaluar el modelo elegido en el punto 6.

Dentro del entrenamiento, se reservan los últimos **90 días** para comparar los modelos. Cada modelo se ajusta con los días anteriores y predice esos 90 días sin conocer sus valores reales. Este paso se llama **validación**.

Se elige el modelo con menor **MAE**, es decir, menor error absoluto promedio. Después se vuelven a calcular sus parámetros con todos los datos de 2023–2024.

Los puntos 2–4 ya analizaron la serie completa. Por eso, aunque los errores de 2025 no se usan para elegir el modelo, las decisiones iniciales pueden estar influidas por esa revisión previa.

## Modelos que se comparan

| Serie | Alternativas | Motivo |
|---|---|---|
| Viajes MBTB | 12 combinaciones, con y sin parte semanal | Evaluar el patrón de la semana y si conviene aplicar diferencias. |
| Temperatura y humedad | 4 modelos con los valores originales y otros 4 con diferencia anual | Comprobar si restar el valor de hace 365 días mejora la predicción. |
| Precipitación | 4 modelos con los valores originales | Las pruebas del punto 4 no mostraron una necesidad clara de aplicar diferencias. |

En temperatura y humedad, la **diferencia anual** consiste en restar a cada valor el de 365 días antes. Se modela esa diferencia y luego se vuelve a la unidad original para calcular el error. Estos modelos se escriben SARIMA(p,0,q)(0,1,0)₃₆₅. Las primeras 365 observaciones sirven como referencia y no entran en el ajuste de las diferencias. El período de 365 días es una aproximación, ya que el calendario incluye un año bisiesto.

Se prueba un conjunto limitado de opciones. No se incluyen términos que relacionen directamente los valores o errores de años anteriores, aparte de la diferencia anual. Para viajes tampoco se representan a la vez los ciclos semanal y anual. Por lo tanto, se elige **el mejor de los modelos probados**, no necesariamente el mejor de todos los posibles.

Los modelos sin diferencias incluyen un término constante. Los modelos con diferencias no incluyen un término adicional de crecimiento sostenido, llamado deriva.

## ¿Cómo se leen las tablas?

- **MAE de validación:** error absoluto promedio en los 90 días reservados. Es el criterio utilizado para elegir el modelo; cuanto menor, mejor.
- **AIC y BIC:** medidas que consideran tanto el ajuste como la cantidad de parámetros. Valores menores son preferibles cuando los modelos son comparables. No se usan para ordenar juntos modelos con distintas diferencias o distinta cantidad de datos utilizados.
- **Coeficiente:** valor calculado para cada parámetro del modelo.
- **Error estándar e intervalo de confianza:** indican la incertidumbre de esa estimación.
- **p-valor:** si es menor a 0,05, aporta evidencia de que el parámetro es distinto de cero, siempre que se cumplan los supuestos del modelo. Esto no asegura que el modelo prediga bien.

Si el cálculo de un modelo no logra converger, es decir, no alcanza una solución numérica estable, se descarta y se registra la advertencia.'''), code(LOAD), code('''OUT = ROOT / 'puntos/punto5'
OUT.mkdir(exist_ok=True)
seleccionados, comparaciones, parametros, resumen = {}, [], [], []
for nombre, s in series.items():
    train = s.loc[:'2024-12-31']
    c, r, tabla, avisos = seleccionar(train, nombre)
    seleccionados[nombre] = c
    tabla.insert(0, 'serie', nombre)
    comparaciones.append(tabla)
    ic = r.conf_int()
    par = pd.DataFrame({'coeficiente': r.params, 'error_estandar': r.bse,
                        'p_valor': r.pvalues, 'IC95_inferior': ic.iloc[:,0], 'IC95_superior': ic.iloc[:,1]})
    par.insert(0, 'serie', nombre)
    par.index.name = 'parametro'
    parametros.append(par.reset_index())
    elegida = tabla[tabla.seleccionado.eq(True)].iloc[0]
    resumen.append({'serie': nombre, 'modelo': etiqueta(c), 'n_train': len(train),
                    'n_estimacion': int(r.nobs), 'MAE_validacion': elegida.MAE_validacion,
                    'AIC_train': r.aic, 'BIC_train': r.bic, 'advertencias_train': avisos})
    display(Markdown(f'### {nombre}'))
    display(tabla.sort_values('MAE_validacion').round(3))
    display(Markdown(f'**Modelo seleccionado:** {etiqueta(c)}. Parámetros calculados con todos los datos de 2023–2024:'))
    display(par.round(4))
comparacion = pd.concat(comparaciones, ignore_index=True)
resumen = pd.DataFrame(resumen)
comparacion.to_csv(OUT / 'comparacion_sarima.csv', index=False)
pd.concat(parametros, ignore_index=True).to_csv(OUT / 'parametros_seleccionados.csv', index=False)
resumen.to_csv(OUT / 'modelos_seleccionados.csv', index=False)
(OUT / 'seleccion.json').write_text(json.dumps(seleccionados, ensure_ascii=False, indent=2))
display(resumen.round(3))
'''), md('''## Lectura de los resultados

A continuación se presenta el modelo elegido para cada serie. Los AIC y BIC de la tabla de comparación se calcularon antes de la validación. Los de la tabla resumen se calcularon después, con todo el entrenamiento. Como utilizan distintos datos, no deben compararse entre esas dos tablas.'''), code('''for fila in resumen.itertuples():
    c = seleccionados[fila.serie]
    lectura = ('Entre las opciones probadas, funcionó mejor restar el valor de hace 365 días.' if c['anual'] else
               'El modelo elegido no necesita restar el valor de hace 365 días.')
    display(Markdown(f'**{fila.serie}:** se seleccionó **{fila.modelo}**, con MAE de validación '
                     f'**{fila.MAE_validacion:.3f}** en las unidades de la serie. {lectura} '
                     'En el punto 6 se revisa cómo predice los datos de 2025.'))
'''), md('''## Limitaciones y herramientas

Se utiliza `SARIMAX` de statsmodels para calcular los parámetros mediante máxima verosimilitud: un método que busca valores compatibles con los datos observados. Cada serie se modela por separado, sin agregar otras variables explicativas. Pandas y NumPy se usan para organizar los datos.

El método supone errores con distribución normal. Esta idea puede resultar poco adecuada para la precipitación, que tiene muchos días sin lluvia y algunos valores muy altos. Además, estos modelos pueden producir valores negativos, que no tienen sentido para la lluvia o los viajes. Los errores se calculan con las predicciones originales, sin modificarlas.

Solo hay dos años de entrenamiento y se usa un único período de validación. Con otro período, podría elegirse un modelo diferente. En el punto 8 se revisarán los errores del modelo para evaluar si quedaron patrones sin explicar.

**Referencia:** Statsmodels developers. (s. f.). *statsmodels.tsa.statespace.sarimax.SARIMAX*. [Documentación oficial](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html).''')]
p6 = [md(r'''# Punto 6 — Evaluación en entrenamiento y prueba

En este punto se revisa qué tan bien predicen los modelos elegidos en el punto 5. Se utilizan **2023–2024 para entrenar** y **2025 para probar**. Los datos conservan su orden por fecha.

## ¿Cómo se mide el error?

El error es la diferencia entre el valor real y la predicción. Se calculan tres medidas:

- **MAE (error absoluto promedio):** indica cuánto se equivoca el modelo en promedio, sin considerar si predice de más o de menos. Por ejemplo, un MAE de 4 °C significa que las predicciones se alejan, en promedio, 4 °C de la temperatura real.
- **RMSE (raíz del error cuadrático medio):** da más peso a los errores grandes. Si es mucho mayor que el MAE, algunos errores grandes tienen una influencia importante.
- **Sesgo (error promedio con signo):** indica la dirección del error. Un valor positivo significa que el modelo predice menos de lo observado; uno negativo, que predice más.

Para MAE y RMSE, **un valor menor indica mayor precisión**. Las medidas se expresan en la unidad de cada serie: viajes, °C, puntos porcentuales de humedad o mm de lluvia. No se comparan directamente errores de variables con unidades distintas.

No se utiliza MAPE, una medida de error porcentual, porque requiere dividir por el valor real y la precipitación contiene ceros.

## ¿Cómo se hace la evaluación?

**En entrenamiento**, se predice cada día con los valores reales de los días anteriores. Los parámetros se calcularon con todo el período de entrenamiento, por lo que estos errores describen el ajuste a datos ya utilizados. Se dejan fuera las primeras 30 predicciones, o más si el modelo necesita un inicio más largo. Si se aplicó la diferencia anual, tampoco se evalúan los primeros 365 días, porque se usaron como referencia.

**En prueba**, el modelo parte del 31/12/2024 y predice todo 2025. No recibe los valores reales de 2025 ni se actualiza durante ese año. Se calcula el error de los 365 días y, por separado, el de los primeros 30 días. Este segundo resultado permite observar cómo le fue al comienzo; no se usa para cambiar el modelo elegido.

Las dos evaluaciones tienen distinta dificultad. En entrenamiento se conocen los valores reales anteriores a cada día. En prueba se predicen muchos días seguidos sin esa información nueva. Por eso, un error mayor en prueba no demuestra por sí solo que el modelo haya aprendido demasiado los datos de entrenamiento.

Además, el modelo se eligió por su error en una validación de 90 días. El que funciona mejor para ese plazo puede no ser el más adecuado para predecir un año completo.'''), code(LOAD), code('''archivo = ROOT / 'puntos/punto5/seleccion.json'
if not archivo.exists():
    raise FileNotFoundError('Ejecutar primero punto5.ipynb para guardar los modelos seleccionados.')
seleccionados = json.loads(archivo.read_text())
OUT = ROOT / 'puntos/punto6'
OUT.mkdir(exist_ok=True)
filas, predicciones = [], []
for nombre, s in series.items():
    train, test = s.loc[:'2024-12-31'], s.loc['2025-01-01':]
    assert train.index.max() < test.index.min() and len(test) == 365
    c = seleccionados[nombre]
    r, avisos = ajustar(train, c)
    pred_train = ajustados(r, train, c).iloc[max(30, int(r.loglikelihood_burn)):]
    pred_test = pronosticar(r, train, c, test.index)
    for conjunto, real, pred in [('Training (1 paso)', train.reindex(pred_train.index), pred_train),
                                  ('Testing (365 días)', test, pred_test),
                                  ('Testing (30 días)', test.iloc[:30], pred_test.iloc[:30])]:
        filas.append(dict(serie=nombre, modelo=etiqueta(c), conjunto=conjunto,
                          inicio=str(real.index.min().date()), fin=str(real.index.max().date()),
                          **metricas(real, pred)))
    for conjunto, real, pred in [('Training',train.reindex(pred_train.index),pred_train), ('Testing',test,pred_test)]:
        predicciones.append(pd.DataFrame({'fecha': real.index, 'serie': nombre, 'conjunto': conjunto,
                                         'observado': real.values, 'prediccion': pred.values,
                                         'error': (real-pred).values}))
    fig, ax = plt.subplots()
    ax.plot(train.iloc[-90:], color='gray', lw=.8, label='Últimos 90 días de entrenamiento')
    ax.plot(test, lw=.8, label='Observado en prueba')
    ax.plot(pred_test, lw=1.3, label='Pronóstico de origen fijo')
    ax.axvline(test.index[0], color='black', linestyle='--', lw=.8)
    ax.set(title=nombre, xlabel='Fecha', ylabel=nombre)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
resultados = pd.DataFrame(filas)
predicciones = pd.concat(predicciones, ignore_index=True)
resultados.to_csv(OUT / 'metricas_training_testing.csv', index=False)
predicciones.to_csv(OUT / 'predicciones_evaluacion.csv', index=False)
display(resultados.round(3))
'''), md('## Resultados por serie'), code('''unidades = {'Viajes MBTB':'viajes', 'Temperatura':'°C', 'Humedad':'puntos porcentuales', 'Precipitación':'mm'}
for nombre in series:
    tabla = resultados[resultados.serie.eq(nombre)].set_index('conjunto')
    tr, te, corto = (tabla.loc[k] for k in ['Training (1 paso)', 'Testing (365 días)', 'Testing (30 días)'])
    direccion = 'subestimación' if te.sesgo > 0 else 'sobreestimación'
    display(Markdown(f'**{nombre}.** El MAE de entrenamiento es **{tr.MAE:.2f}** y el de prueba anual '
                     f'**{te.MAE:.2f} {unidades[nombre]}**. El RMSE anual es **{te.RMSE:.2f}**; '
                     f'el sesgo de **{te.sesgo:.2f}** indica {direccion} promedio. '
                     f'En los primeros 30 días, el MAE es **{corto.MAE:.2f}**. '
                     'Estas predicciones se hicieron sin incorporar los valores reales de 2025.'))
    prueba = predicciones[(predicciones.serie == nombre) & (predicciones.conjunto == 'Testing')]
    if nombre in ('Viajes MBTB','Precipitación'):
        display(Markdown(f'Se registran **{int((prueba.prediccion < 0).sum())}** pronósticos negativos, '
                         'que no tienen sentido para esta variable.'))
'''), md('''## Conclusión

Los modelos se eligieron con los datos de entrenamiento. Después se compararon sus predicciones con los valores reales de 2025. Las tablas muestran cuánto se equivocó cada modelo y si tendió a predecir de más o de menos.

Los resultados muestran que:

- **Viajes:** el modelo predijo, en promedio, menos viajes de los que se observaron en 2025. El patrón semanal puede no alcanzar para explicar los cambios a lo largo del año.
- **Temperatura:** se eligió un modelo con diferencia anual. Hay solo dos años de entrenamiento, lo que limita la información disponible para representar ese ciclo.
- **Humedad:** el error promedio de los primeros 30 días fue mayor que el del año completo. Predecir un período más corto no asegura un error menor: también influye qué días se evalúan.
- **Precipitación:** el RMSE anual fue bastante mayor que el MAE. Esto indica que algunos errores grandes tuvieron un peso importante en el resultado.

Estos modelos son una primera elección. En el punto 7 se compararán con otras alternativas y en el punto 8 se revisará si sus errores todavía presentan patrones. Los resultados corresponden a los archivos CSV del proyecto; aquí no se comprueba su procedencia externa.

**Herramientas:** Pandas y NumPy para las tablas y los errores; Matplotlib para los gráficos; statsmodels para los modelos. Las funciones compartidas están en `scripts/modelos_puntos56.py`.''')]
for numero, cells in [(5,p5),(6,p6)]:
    path = ROOT / f'puntos/punto{numero}/punto{numero}.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(nb, path)
indice = ROOT / 'puntos/00_indice.ipynb'
texto = indice.read_text()
for numero in (5,6):
    texto = texto.replace(f'`punto{numero}/`', f'[punto{numero}/punto{numero}.ipynb](punto{numero}/punto{numero}.ipynb)')
indice.write_text(texto)
