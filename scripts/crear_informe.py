"""Construye el informe PDF y su versión Markdown a partir de resultados verificados."""
from pathlib import Path
import json, base64, re, html, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, acovf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from modelos_puntos56 import cargar
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entrega';OUT.mkdir(exist_ok=True)
FIG=OUT/'figuras';FIG.mkdir(exist_ok=True)
series=cargar(ROOT)

def csv(n,nombre): return pd.read_csv(ROOT/f'puntos/punto{n}'/nombre)
seleccion=csv(5,'modelos_seleccionados.csv');met=csv(6,'metricas_training_testing.csv')
comp=csv(7,'comparacion_modelos.csv');diag=csv(8,'diagnostico.csv');lb=csv(8,'ljung_box.csv')
fc=csv(9,'pronostico_30_dias.csv');var=csv(10,'comparacion_var_sarima.csv')
vi=json.loads((ROOT/'puntos/punto10/resumen_var.json').read_text())
g=csv(11,'granger.csv');est=csv(12,'comparacion_estacionalidad.csv')

# Figuras del informe: escalas y leyendas legibles, sin depender del estado de Jupyter.
fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
for ax,(nombre,s) in zip(axes,series.items()):
    ax.plot(s,lw=.55,color='#256282');ax.set_ylabel(nombre);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(FIG/'series.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(11,9))
for row,(nombre,s) in enumerate(series.items()):
    z=s.diff(7).dropna() if nombre=='Viajes MBTB' else s.diff(365).dropna() if nombre in ('Temperatura','Humedad') else s
    for col,(titulo,v) in enumerate([('Autocorrelación',acf(z,nlags=28)),('Autocorrelación parcial',pacf(z,nlags=28,method='ywm')),('Autocovarianza',acovf(z,nlag=28,fft=True))]):
        ax=axes[row,col];ax.stem(np.arange(1,29),v[1:],markerfmt=' ',basefmt=' ');ax.axhline(0,color='gray',lw=.6)
        if col<2:
            b=1.96/np.sqrt(len(z));ax.axhspan(-b,b,alpha=.12)
        ax.set_title(f'{nombre}: {titulo}',fontsize=9)
fig.tight_layout();fig.savefig(FIG/'correlaciones.png',dpi=150);plt.close(fig)
# Figura de diferencias, comparable con las decisiones iniciales del punto 2.
fig,axes=plt.subplots(4,1,figsize=(10,7),sharex=True)
for ax,(nombre,s) in zip(axes,series.items()):
    z=s.diff(7) if nombre=='Viajes MBTB' else s.diff(365) if nombre in ('Temperatura','Humedad') else s
    ax.plot(z,lw=.55);ax.set_ylabel(nombre);ax.axhline(0,color='gray',lw=.5)
fig.tight_layout();fig.savefig(FIG/'diferencias.png',dpi=150);plt.close(fig)

FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
if FONT.exists():
    pdfmetrics.registerFont(TTFont('Informe',str(FONT)))
    pdfmetrics.registerFont(TTFont('InformeBold',str(FONT.with_name('DejaVuSans-Bold.ttf'))))
    font,bold='Informe','InformeBold'
else:font,bold='Helvetica','Helvetica-Bold'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TextoTP',fontName=font,fontSize=9.5,leading=14,spaceAfter=9,textColor=colors.HexColor('#22313d')))
styles.add(ParagraphStyle(name='TituloTP',fontName=bold,fontSize=17,leading=22,spaceAfter=15,textColor=colors.HexColor('#18485f')))
styles.add(ParagraphStyle(name='SubTP',fontName=bold,fontSize=11,leading=15,spaceAfter=9,textColor=colors.HexColor('#18485f')))
styles.add(ParagraphStyle(name='TablaTP',fontName=font,fontSize=7.1,leading=10))
styles.add(ParagraphStyle(name='PortadaTP',fontName=bold,fontSize=24,leading=31,alignment=TA_CENTER,spaceAfter=20,textColor=colors.HexColor('#18485f')))
styles.add(ParagraphStyle(name='CentroTP',fontName=font,fontSize=12,leading=20,alignment=TA_CENTER,spaceAfter=15))
story=[];md=[]

def p(text):
    story.append(Paragraph(html.escape(str(text)).replace('\n','<br/>'),styles['TextoTP']));md.append(str(text)+'\n')
def title(text):
    story.append(Paragraph(html.escape(text),styles['TituloTP']));md.append('# '+text+'\n')
def sub(text):
    story.append(Paragraph(html.escape(text),styles['SubTP']));md.append('## '+text+'\n')
def page(text):
    if story:story.append(PageBreak())
    title(text)
def table(df,widths=None):
    df=df.copy()
    def fmt(x):
        if isinstance(x,(float,np.floating)):
            if not np.isfinite(x):return '—'
            return f'{x:.3g}' if abs(x)<.01 and x!=0 else f'{x:.3f}'
        return str(x)
    rows=[[str(x) for x in df.columns]]+[[fmt(x) for x in row] for row in df.itertuples(index=False,name=None)]
    vals=[[Paragraph(html.escape(v),styles['TablaTP']) for v in row] for row in rows]
    if widths is None:widths=[490/len(df.columns)]*len(df.columns)
    t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7a9fac')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,12)])
    md.append('| '+' | '.join(rows[0])+' |\n| '+' | '.join(['---']*len(rows[0]))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows[1:])+'\n')
def img(path,caption,height=390):
    from PIL import Image as PILImage
    path=Path(path);w,h=PILImage.open(path).size;scale=min(490/w,height/h)
    story.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,5),Paragraph(html.escape(caption),styles['TablaTP']),Spacer(1,10)])
    md.append(f'![{caption}]({Path(os.path.relpath(path, OUT)).as_posix()})\n')

story.extend([Spacer(1,70),Paragraph('Trabajo Práctico N.º 1<br/>Análisis de Series Temporales',styles['PortadaTP']),Spacer(1,25),Paragraph('Demanda de bicicletas públicas y clima en Rosario',styles['CentroTP']),Paragraph('Universidad Austral<br/>Maestría en Ciencia de Datos · Modalidad virtual',styles['CentroTP']),Spacer(1,30),Paragraph('Tomas del Bo<br/>Matias Villanueva<br/>Juan Ignacio Paberolis',styles['CentroTP']),Spacer(1,25),Paragraph('Docentes: Rodrigo Del Rosso y Braian Drago<br/>Datos analizados: enero de 2023 a diciembre de 2025',styles['CentroTP'])])
md.append('# Trabajo Práctico N.º 1 — Análisis de Series Temporales\n\nTomas del Bo · Matias Villanueva · Juan Ignacio Paberolis\n\nUniversidad Austral — Maestría en Ciencia de Datos\n')
page('Resumen ejecutivo')
p('Se analizan cuatro series diarias de Rosario: viajes del sistema de bicicletas públicas MBTB, temperatura, humedad y precipitación. El objetivo es describir sus patrones, comparar modelos de predicción y estudiar si la información climática previa se relaciona con la demanda. Los archivos contienen 1.096 días completos, entre el 1 de enero de 2023 y el 31 de diciembre de 2025.')
p('Los modelos univariados se eligieron con una validación de 90 días dentro de 2023–2024. El año 2025 se reservó para evaluar predicciones desde un único origen, sin incorporar los datos reales de ese año. Se compararon SARIMA, referencias simples, ARIMA y suavizado exponencial. También se estimó un modelo conjunto VAR y se revisó la representación de los ciclos de calendario.')
table(met[met.conjunto=='Testing (365 días)'][['serie','MAE','RMSE']].rename(columns={'serie':'Serie','MAE':'MAE en 2025','RMSE':'RMSE en 2025'}))
p('Los errores se expresan en las unidades de cada variable y no deben compararse entre variables. Los SARIMA de viajes, temperatura y humedad dejan alguna autocorrelación en sus residuos. En viajes aparece además un parámetro muy próximo al límite de invertibilidad. Por eso, los modelos son referencias de trabajo y no soluciones definitivas.')
p(f'El VAR elegido tiene {vi["rezagos"]} rezagos y es estable, pero su prueba conjunta de residuos rechaza ausencia de autocorrelación (p={vi["blancura_p"]:.3g}). Las pruebas de Granger y las respuestas a shocks se interpretan como resultados exploratorios. No demuestran causalidad real. Se presentan pronósticos de 30 días posteriores al último dato, con las limitaciones de cada modelo.')
page('Índice de contenido')
for item in ['1. Introducción y problema de interés — p. 4','2. Marco teórico — p. 5','3. Datos y exploración gráfica — puntos 1 y 2 — p. 6','4. Autocorrelación y raíces unitarias — puntos 3 y 4 — p. 8','5. Selección SARIMA y evaluación — puntos 5 y 6 — p. 10','6. Comparación con alternativas — punto 7 — p. 12','7. Diagnóstico de residuos — punto 8 — p. 13','8. Pronóstico de 30 días — punto 9 — p. 15','9. Modelo conjunto VAR — punto 10 — p. 17','10. Impulso-respuesta y Granger — punto 11 — p. 19','11. Revisión de estacionalidad — punto 12 — p. 22','12. Conclusiones — p. 24','13. Referencias — p. 25','Apéndices: parámetros, ejecución y archivos — pp. 26–27']:
    p(item)
p('El informe sigue la estructura solicitada. Los notebooks de cada punto contienen las tablas completas y el código. El archivo de entrega tp1_reproducible.py incluye los datos y las instrucciones para reconstruir los análisis.')
page('1. Introducción y problema de interés')
p('El uso de bicicletas públicas cambia según el día de la semana, la época del año y las condiciones climáticas. Estas variaciones importan para planificar la cantidad de bicicletas, su distribución entre estaciones y las tareas de mantenimiento. Un pronóstico puede apoyar esas decisiones, aunque no reemplaza el conocimiento operativo del servicio.')
p('Se eligió la cantidad diaria de viajes como variable principal. La temperatura, la humedad y la precipitación permiten estudiar condiciones que podrían acompañar los cambios de demanda. Las cuatro series tienen la misma frecuencia y el mismo período, lo que facilita compararlas y construir un modelo conjunto.')
p('La pregunta principal es qué modelos representan mejor los patrones temporales de cada serie y cuánto se equivocan al predecir datos posteriores. Una segunda pregunta es si el pasado de las variables climáticas agrega información para explicar los viajes, una vez incluidos sus propios valores anteriores y los patrones de calendario.')
p('El alcance es observacional. No se dispone de experimentos, información individual de usuarios ni controles completos sobre feriados, disponibilidad de bicicletas, cambios de tarifa u otras decisiones del servicio. Las asociaciones encontradas no se interpretan como efectos causales.')
sub('Datos y trazabilidad')
p('El punto 1 atribuye los viajes al portal Datos Abiertos Rosario. Los CSV climáticos identifican variables y coordenadas, pero el proyecto no incluye una descarga verificable, una URL exacta de consulta ni metadatos completos de su proveedor. Por eso se utilizan como archivos suministrados para el ejercicio, sin atribuirles una fuente no comprobada. Esta limitación de procedencia se conserva en el informe.')
p('Se mantienen los 1.096 días del proyecto. La consigna menciona una entrega en octubre de 2025, pero los archivos llegan hasta diciembre de ese año. El análisis respeta el contenido disponible; sus resultados no se presentan como una predicción realizada antes de aquella fecha.')
page('2. Marco teórico')
sub('Estacionariedad y diferencias')
p('Una serie es estacionaria en sentido débil si su media y su varianza no cambian con el tiempo y la relación entre dos observaciones depende de la distancia entre ellas. Un ciclo visible no demuestra por sí solo una raíz unitaria. La diferencia diaria es Δy(t)=y(t)−y(t−1); la estacional es Δs y(t)=y(t)−y(t−s). Diferenciar sin necesidad puede agregar ruido y dificultar el modelo.')
sub('Autocorrelación y modelos SARIMA')
p('La FAC mide la relación con valores anteriores. La FACP descuenta las relaciones explicadas por los rezagos intermedios. En un AR(p) estacionario, la FACP teórica se corta después de p; en un MA(q), la FAC teórica se corta después de q. En muestras reales, estos patrones solo orientan la elección. Aquí FAS y FAC se tratan como nombres de autocorrelación simple y se agrega la autocovarianza como complemento, porque la consigna no define esas siglas.')
p('El SARIMA se escribe: phi(B) Phi(B^s) (1-B)^d (1-B^s)^D y(t) = c + theta(B) Theta(B^s) epsilon(t). El operador B desplaza un dato hacia atrás. p y P indican términos de valores anteriores; q y Q, términos de errores anteriores; d y D, diferencias; s, duración del ciclo. Se usa la implementación de statsmodels, sin variables externas en el punto 5.')
sub('Error y evaluación')
p('Con e(t)=y(t)−ŷ(t), MAE=promedio de |e(t)|; RMSE=raíz del promedio de e(t)²; sesgo=promedio de e(t). Los dos primeros miden tamaño del error. El sesgo positivo indica que se predijo de menos. No se usa MAPE porque existen valores de precipitación iguales a cero. AIC y BIC equilibran ajuste y complejidad; no se comparan indiscriminadamente entre distintas diferencias o muestras.')
sub('Modelo conjunto')
p('Un VAR(p) tiene forma z(t) = c + A1 z(t-1) + ... + Ap z(t-p) + u(t). Cada ecuación utiliza rezagos de todas las variables. Se requiere un comportamiento estacionario y un sistema estable. Las respuestas a shocks dependen de cómo se separan las innovaciones. Granger evalúa información predictiva adicional y no demuestra una causa física (Statsmodels developers, s. f.-a).')
page('3. Datos y exploración gráfica — puntos 1 y 2')
res=pd.DataFrame({n:s.describe() for n,s in series.items()}).T.reset_index().rename(columns={'index':'Serie'})
table(res[['Serie','count','mean','std','min','max']].rename(columns={'count':'n','mean':'Media','std':'Desvío','min':'Mínimo','max':'Máximo'}))
p('Se comprobaron fechas duplicadas, días faltantes y valores no finitos. Las series tienen fechas coincidentes y no presentan faltantes. Las unidades son viajes diarios, °C, porcentaje de humedad relativa y mm diarios de precipitación.')
img(FIG/'series.png','Figura 1. Series originales. Fuente: elaboración propia con los CSV del proyecto.',height=430)
page('3.1. Diferencias y lectura inicial')
p('Viajes muestra un ciclo semanal y cambios de nivel. Temperatura y humedad presentan patrones anuales. Precipitación contiene muchos ceros y picos aislados. El punto 2 propuso diferencias como primera aproximación visual; el punto 4 mostró que no todas eran necesarias o suficientes. La diferencia anual de humedad, por ejemplo, no quedó respaldada por todas las pruebas.')
img(FIG/'diferencias.png','Figura 2. Diferencia semanal de viajes, diferencia anual de temperatura y humedad, y precipitación en niveles.',height=430)
p('Estos gráficos no constituyen una prueba de estacionariedad. El aumento o la reducción de dispersión después de diferenciar tampoco permite, por sí solo, elegir el mejor modelo.')
page('4. Autocorrelación — punto 3')
img(FIG/'correlaciones.png','Figura 3. FAC, FACP y autocovarianza de las transformaciones exploratorias. Bandas aproximadas del 95% en las correlaciones.',height=490)
p('La diferencia semanal de viajes puede introducir una relación negativa en el rezago 7. Por eso no se la toma como obligatoria. En temperatura y humedad se observa dependencia de corto plazo después de la diferencia anual. Precipitación muestra una dependencia más breve. Los órdenes se prueban luego con validación, en lugar de elegirlos solo por la forma de estas barras.')
page('4.1. Pruebas de raíces unitarias — punto 4')
pr=csv(4,'resultados_pruebas.csv');base=pr[(pr.transformacion=='Nivel')&(pr.especificacion=='c')]
t=base.pivot(index='serie',columns='prueba',values='p_reportado').reset_index()
table(t)
p('ADF y Phillips–Perron (PP) parten de la hipótesis de raíz unitaria; KPSS parte de estacionariedad. El nivel de referencia es 5%. En KPSS, los valores de los extremos se interpretan como límites del p-valor disponible. En las pruebas de raíz unitaria, un cero numérico indica un valor muy pequeño, no una probabilidad exactamente nula.')
p('Viajes presenta resultados mixtos: ADF no rechaza raíz unitaria, PP sí la rechaza y KPSS rechaza estacionariedad. Temperatura y humedad son compatibles con estacionariedad bajo una media fija, pero algunas conclusiones cambian al incluir tendencia. Precipitación muestra coincidencia a favor de conservar sus niveles. Estas diferencias explican por qué no se aplica la misma transformación a todas las series.')
p('La diferencia anual de humedad no resuelve por sí sola el problema: KPSS todavía rechaza estacionariedad en esa transformación. El análisis visual inicial se considera una propuesta, que luego se revisa con pruebas y predicción. Las pruebas completas, con y sin tendencia, están en resultados_pruebas.csv.')
page('5. Selección SARIMA — punto 5')
p('Se estiman 12 candidatos para viajes, 8 para temperatura, 8 para humedad y 4 para precipitación. Se entrenan con el inicio de 2023–2024 y se pronostican sus últimos 90 días. Gana el menor MAE en esa validación. Los parámetros se vuelven a calcular con todo 2023–2024. No se usa el error de 2025 para esta elección.')
table(seleccion[['serie','modelo','MAE_validacion','n_estimacion']].rename(columns={'serie':'Serie','modelo':'Modelo','MAE_validacion':'MAE validación','n_estimacion':'n efectivo'}),[85,235,85,85])
p('La diferencia anual se calcula antes de ajustar los modelos que la incluyen. Por ello se pierden 365 observaciones de estimación y se reconstruyen después las predicciones en unidades originales. Los AIC/BIC solo se interpretan dentro de conjuntos comparables; el MAE de validación permite comparar todas las alternativas en fechas comunes.')
p('Los p-valores individuales de los coeficientes se incluyen en el apéndice. En viajes, el término MA estacional está muy cerca de −1 y su error estándar es grande. Esto sugiere que la representación requiere revisión y coincide con la advertencia del diagnóstico posterior. Un buen resultado de validación no elimina este problema.')
p('La exploración de los puntos 2–4 utilizó todo el período. La separación de entrenamiento y prueba evita usar numéricamente el error de 2025 en la selección, pero no vuelve completamente independiente el conocimiento previo de esos datos.')
page('5.1. Entrenamiento y prueba — punto 6')
table(met[['serie','conjunto','n','MAE','RMSE','sesgo']].rename(columns={'serie':'Serie','conjunto':'Evaluación','sesgo':'Sesgo'}),[83,125,40,80,80,82])
p('El entrenamiento usa predicciones de un día con valores reales anteriores, aunque los parámetros se calcularon con todo el entrenamiento. La prueba predice 365 días seguidos desde diciembre de 2024. La brecha de errores no demuestra por sí sola sobreajuste: también cambia la información disponible y el plazo de predicción.')
p('El sesgo anual de viajes es positivo: el modelo tiende a subestimar la demanda. En humedad, enero tiene un error mayor que el promedio del año completo. Un plazo más corto no asegura menor error, porque también cambia el período evaluado.')
page('6. Comparación con alternativas — punto 7')
filas=[]
for nombre,subt in comp[comp.horizonte==365].groupby('serie'):
    best=subt.sort_values('MAE').iloc[0]
    sar=subt[subt.modelo=='SARIMA punto 5'].iloc[0]
    filas.append({'Serie':nombre,'Menor MAE observado':best.modelo,'MAE mejor':best.MAE,'MAE SARIMA':sar.MAE})
table(pd.DataFrame(filas),[85,210,95,100])
img(ROOT/'puntos/punto7/comparacion.png','Figura 4. Error absoluto promedio de los modelos en 2025.',height=350)
p('Todos los modelos usan las mismas fechas y predicen desde el mismo origen. La clasificación es descriptiva: no se reutiliza 2025 para seleccionar un ganador definitivo. La comparación completa de 30 y 365 días queda en comparacion_modelos.csv.')
page('7. Diagnóstico de residuos — punto 8')
table(lb.pivot(index='serie',columns='rezago',values='lb_pvalue').reset_index().rename(columns={'serie':'Serie',7:'LB p, 7 días',14:'LB p, 14 días',28:'LB p, 28 días'}))
table(diag[['serie','JB_p','ARCH_p','raiz_minima']].rename(columns={'serie':'Serie','JB_p':'Jarque–Bera p','ARCH_p':'ARCH p','raiz_minima':'Raíz mínima'}))
p('Ljung–Box detecta autocorrelación en viajes y temperatura en los tres rezagos revisados. En humedad rechaza en 7 y 28 días. En precipitación no rechaza en estos rezagos, pero sus errores se apartan mucho de una distribución normal. La prueba se ajusta por el número de términos AR y MA, incluidos los estacionales (Statsmodels developers, s. f.-b).')
p('Jarque–Bera rechaza normalidad en viajes, humedad y lluvia. ARCH detecta variación del tamaño de los errores en viajes y humedad. Estos resultados afectan especialmente la confianza en intervalos gaussianos. La raíz mínima de viajes está casi en uno: el modelo se aproxima al límite de invertibilidad.')
img(ROOT/'puntos/punto8/diagnostico_0.png','Figura 5. Diagnóstico de residuos de viajes.',height=300)
page('7.1. Diagnóstico de las series climáticas')
img(ROOT/'puntos/punto8/diagnostico_1.png','Figura 6. Temperatura: queda dependencia temporal en los residuos.',height=300)
img(ROOT/'puntos/punto8/diagnostico_3.png','Figura 7. Precipitación: fuerte asimetría y errores extremos.',height=300)
page('8. Pronóstico de 30 días — punto 9')
p('Se conservan los órdenes del punto 5 y se ajustan sus parámetros con 2023–2025. La ventana va del 1 al 30 de enero de 2026, después del último dato disponible. No se compara con observaciones reales de 2026, porque no forman parte de los archivos.')
table(fc.groupby('serie').agg(Promedio=('prediccion','mean'),Mínimo=('prediccion','min'),Máximo=('prediccion','max')).reset_index().rename(columns={'serie':'Serie'}))
p('Las bandas del 95% son intervalos predictivos aproximados bajo el modelo, condicionados a sus parámetros. No incluyen toda la incertidumbre de selección. En la diferencia anual, los valores de hace 365 días son conocidos para este horizonte, por lo que se suman al centro y a los límites. No se recortan intervalos físicamente imposibles.')
img(ROOT/'puntos/punto9/pronostico_0.png','Figura 8. Pronóstico de viajes e intervalo aproximado del 95%.',height=250)
img(ROOT/'puntos/punto9/pronostico_1.png','Figura 9. Pronóstico de temperatura.',height=230)
page('8.1. Pronósticos de humedad y precipitación')
img(ROOT/'puntos/punto9/pronostico_2.png','Figura 10. Pronóstico de humedad.',height=260)
img(ROOT/'puntos/punto9/pronostico_3.png','Figura 11. Pronóstico de precipitación.',height=260)
p('Un límite negativo de lluvia o humedad fuera de 0–100% expone una limitación del modelo lineal. Estas bandas se presentan sin alteraciones para no ocultarla. Los pronósticos climáticos son ejercicios estadísticos y no reemplazan un servicio meteorológico.')
page('9. Modelo conjunto VAR — punto 10')
p('Se retiró de cada serie una parte de calendario estimada solo con entrenamiento: tendencia lineal, dos pares de senos y cosenos anuales e indicadores de día de semana. En viajes, KPSS seguía rechazando estacionariedad (p≈0,024), por lo que se agregó una diferencia diaria de sus residuos. Las variables climáticas conservaron los residuos en niveles. Después se dividió cada variable por su desvío.')
table(csv(10,'estacionariedad_var.csv')[['serie','ADF_p','KPSS_p','compatible']].rename(columns={'serie':'Serie','compatible':'Coinciden al 5%'}))
p('BIC eligió dos rezagos entre 1 y 21, con una ventana común para comparar los órdenes. El VAR(2) resultó estable. Se mantuvo como modelo parsimonioso para el ejercicio, aunque la prueba conjunta de residuos rechazó ausencia de autocorrelación. La estabilidad no garantiza que el modelo esté bien especificado.')
table(var[['serie','MAE_SARIMA','MAE_VAR']].rename(columns={'serie':'Serie','MAE_SARIMA':'MAE SARIMA','MAE_VAR':'MAE VAR'}))
p('El VAR mejoró el error de temperatura y humedad, pero empeoró viajes y precipitación. Las diferencias de viajes se acumulan para recuperar el nivel antes de sumar el calendario. Esa acumulación puede amplificar el error a plazos largos. No se usaron los valores reales de clima de 2025 para pronosticar la demanda.')
p('Se recalculó el mismo orden con todos los datos y se guardaron 30 predicciones para enero de 2026. Son pronósticos puntuales; no se presentan intervalos que ignoren la acumulación de diferencias o la incertidumbre del calendario.')
page('9.1. Evaluación gráfica del VAR')
img(ROOT/'puntos/punto10/var_testing.png','Figura 12. Predicciones VAR y valores reales en 2025, desde un único origen.',height=560)
p('Las diferencias de desempeño muestran que agregar variables no asegura una mejora. El efecto depende de la variable, del plazo y de la forma elegida para retirar los patrones de calendario.')
page('10. Impulso-respuesta — punto 11')
p('Se estudian shocks de un desvío estándar en el VAR entrenado con 2023–2024. La separación de Cholesky usa el orden temperatura, humedad, precipitación y viajes. Ubicar viajes al final permite una respuesta del mismo día al clima bajo esta suposición. Cambiar el orden modifica la interpretación de los shocks.')
img(ROOT/'puntos/punto11/impulso_respuesta.png','Figura 13. Respuestas estandarizadas y bandas aproximadas del 95%. Viajes representa la diferencia diaria del residuo de calendario.',height=480)
p('En clima las respuestas corresponden a desvíos del calendario. En viajes corresponden a cambios diarios de esos desvíos. Para hablar del nivel de viajes se acumulan las respuestas. Las bandas son asintóticas y no incorporan la estimación previa del calendario ni resuelven los problemas de residuos del VAR.')
page('10.1. Sensibilidad al orden de los shocks')
img(ROOT/'puntos/punto11/sensibilidad_orden.png','Figura 14. Respuesta del cambio diario del residuo de viajes bajo dos órdenes de Cholesky.',height=250)
ir=csv(11,'impulso_respuesta.csv');filas=[]
for origen,t in ir[(ir.destino=='Viajes MBTB')&(ir.origen!='Viajes MBTB')].groupby('origen'):
    row=t.loc[t.respuesta_nivel_viajes.abs().idxmax()]
    filas.append({'Shock':origen,'Día de mayor respuesta absoluta':int(row.dia),'Respuesta acumulada (viajes)':row.respuesta_nivel_viajes})
table(pd.DataFrame(filas))
bandas=[]
for origen,t in ir[(ir.destino=='Viajes MBTB')&(ir.origen!='Viajes MBTB')].groupby('origen'):
    dias=t.loc[(t.inferior95>0)|(t.superior95<0),'dia'].astype(int).tolist()
    bandas.append(f'{origen}: {dias}')
p('Las bandas puntuales del 95% para el cambio diario del residuo de viajes excluyen cero en los siguientes días: '+ '; '.join(bandas)+'. No son bandas simultáneas: mirar muchos días aumenta el riesgo de hallazgos por azar.')
p('La tabla resume el mayor movimiento absoluto del nivel de viajes dentro de los 14 días, bajo el orden principal. Es una descripción de la curva, no una prueba adicional de significatividad. El shock no equivale a aumentar directamente un grado, un punto de humedad o un milímetro: su tamaño depende de la innovación estimada y de la separación de Cholesky.')
p('Las relaciones predictivas que van desde viajes hacia clima no significan que los viajes modifiquen el clima. Pueden reflejar dependencia estadística compartida, variables omitidas o un ajuste incompleto. Esta precaución también se aplica a la dirección clima hacia viajes.')
page('10.2. Pruebas de Granger y relación instantánea')
table(g[['origen','destino','F_p_Holm','Wald_p_Holm','F_p_rechaza']].rename(columns={'origen':'Origen','destino':'Destino','F_p_Holm':'F p, Holm','Wald_p_Holm':'Wald p, Holm','F_p_rechaza':'Rechaza F'}),[115,100,95,95,85])
p('Las pruebas F y Wald contrastan si todos los coeficientes de los rezagos de una variable son cero en otra ecuación, con las demás variables incluidas. Se ajustaron los p-valores de las 12 direcciones y la prueba conjunta mediante Holm, por separado para F y Wald. Ambas pruebas contrastan la misma hipótesis y no aportan evidencia independiente.')
clima=g[g.destino=='Viajes MBTB']
hall=', '.join(clima.loc[clima.F_p_rechaza,'origen'])
p(f'Para el destino viajes, las pruebas F con ajuste detectan información adicional de: {hall}. El resultado corresponde al cambio del residuo de viajes, no directamente al nivel original. Las pruebas instantáneas también detectan relación contemporánea; no identifican una dirección causal.')
p('Dado que el VAR conserva autocorrelación residual, estos p-valores se presentan como aproximaciones exploratorias. No se extrae una recomendación de intervención ni una conclusión causal.')
page('11. Revisión de estacionalidad — punto 12')
table(est[['serie','modelo','MAE_validacion','MAE_testing','elegido_validacion']].rename(columns={'serie':'Serie','modelo':'Alternativa','MAE_validacion':'MAE validación','MAE_testing':'MAE 2025','elegido_validacion':'Elegido'}),[78,175,78,79,80])
p('La alternativa anual usa dos pares de senos y cosenos. En viajes añade una tendencia y errores SARIMA semanales; en temperatura y humedad, errores AR(1). Técnicamente se trata de regresión de calendario con errores SARIMA, también llamada SARIMAX, porque incluye funciones conocidas de la fecha. No utiliza clima futuro observado.')
p('La validación mantuvo el modelo previo de viajes y favoreció el ciclo anual con AR(1) en temperatura y humedad. La precipitación no se modificó. En 2025, las alternativas seleccionadas redujeron el MAE climático, mientras que la opción adicional de viajes tuvo un desempeño mucho peor. Un patrón más elaborado no siempre ayuda.')
p('La misma validación ya había servido para elegir órdenes. Seguir probando opciones en ella aumenta el riesgo de adaptarse demasiado a esos 90 días. Además, los modelos nuevos necesitan su propio diagnóstico antes de reemplazar los del punto 5. El punto 9 conserva los pronósticos de los modelos originales por continuidad del análisis.')
p('No se encontró otro trabajo anterior en las carpetas. La referencia de comparación es, por lo tanto, el punto 5 de este proyecto. Esta elección se explicita para no atribuir resultados a un documento no disponible.')
page('11.1. Comparación de las representaciones estacionales')
img(ROOT/'puntos/punto12/estacionalidad.png','Figura 15. Modelos originales y alternativas de calendario frente a los datos reales.',height=540)
p('El calendario suave describe un patrón anual esperado. La diferencia anual, en cambio, usa el valor del ciclo anterior como referencia. Son supuestos distintos y sus resultados dependen de la estabilidad del patrón entre años.')
page('12. Conclusiones')
p('El trabajo muestra que las cuatro series requieren decisiones diferentes. Viajes tiene una estructura semanal y cambios de nivel; temperatura y humedad presentan ciclos anuales; lluvia concentra muchos ceros y episodios intensos. No resulta adecuado diferenciar todas las variables de la misma forma ni elegir un único tipo de modelo por conveniencia.')
p('La separación temporal permitió medir error posterior a la estimación. El SARIMA semanal fue la opción elegida inicialmente para viajes; la temperatura utilizó una diferencia anual; humedad y lluvia conservaron modelos sin diferencia estacional. La comparación con alternativas mostró que la complejidad no asegura el menor error.')
p('Los residuos impiden considerar definitivos los modelos univariados. Persisten autocorrelaciones en viajes, temperatura y humedad, y las distribuciones de errores se apartan de normalidad en varias series. Por ello, los pronósticos e intervalos deben interpretarse como aproximaciones y no como garantías de desempeño.')
p('El VAR integró la información de las cuatro variables. Mejoró las predicciones de algunas variables climáticas, pero no la demanda ni la lluvia. Las pruebas de Granger y las respuestas a shocks describen relaciones del modelo; la autocorrelación restante, el orden de Cholesky y las variables omitidas impiden una interpretación causal.')
p('La revisión anual mejoró los resultados de temperatura y humedad dentro del conjunto probado. Una continuación razonable sería evaluar varios cortes temporales, incorporar calendario operativo y disponibilidad del servicio, y estudiar modelos que respeten la naturaleza no negativa y con muchos ceros de la lluvia. Estas mejoras requieren nueva validación y no se presentan como resultados ya obtenidos.')
p('El aporte principal es un análisis reproducible con decisiones y límites visibles. Los archivos permiten revisar cómo se seleccionó cada modelo, qué información tuvo al predecir y dónde falla. La utilidad operativa depende de actualizar los datos, confirmar su procedencia y volver a medir el desempeño.')
page('13. Referencias')
refs=[
'Municipalidad de Rosario. (s. f.). Datos Abiertos Rosario. https://datos.rosario.gob.ar/ — Portal atribuido a viajes en el punto 1; el proyecto no conserva la consulta exacta de descarga.',
'Statsmodels developers. (s. f.-a). Vector autoregressions. https://www.statsmodels.org/stable/vector_ar.html',
'Statsmodels developers. (s. f.-b). statsmodels.stats.diagnostic.acorr_ljungbox. https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html',
'Statsmodels developers. (s. f.-c). statsmodels.tsa.statespace.sarimax.SARIMAX. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html',
'Statsmodels developers. (s. f.-d). Augmented Dickey–Fuller unit root test. https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html',
'Statsmodels developers. (s. f.-e). Kwiatkowski–Phillips–Schmidt–Shin test. https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html',
'Sheppard, K. (s. f.). Phillips–Perron unit root test. arch. https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.PhillipsPerron.html',
'Universidad Austral. (s. f.). Trabajo Práctico N.º 1: Análisis de Series Temporales [Consigna de cátedra]. Maestría en Ciencia de Datos. Documento suministrado en el proyecto.'
]
for ref in refs:p(ref)
p('Fuente de las figuras y tablas: elaboración propia con los archivos del proyecto. La documentación de software explica los métodos implementados; no valida la procedencia de los datos ni las conclusiones empíricas de este caso.')
page('Apéndice A. Parámetros de los modelos SARIMA')
pa=csv(5,'parametros_seleccionados.csv')
table(pa[['serie','parametro','coeficiente','error_estandar','p_valor']].rename(columns={'serie':'Serie','parametro':'Parámetro','coeficiente':'Estimación','error_estandar':'Error estándar','p_valor':'p-valor'}),[92,92,102,102,102])
p('Los p-valores suponen que la especificación y la aproximación de la inferencia son adecuadas. En particular, la cercanía del término estacional de viajes al límite de invertibilidad hace poco confiable una lectura mecánica de su error estándar. Un p-valor numéricamente cero representa un número muy pequeño.')
page('Apéndice B. Ejecución y organización de archivos')
p('La entrega se compone de TP1_informe.pdf y tp1_reproducible.py. El segundo archivo contiene una copia comprimida de los CSV, los notebooks sin salidas y las funciones necesarias. Se puede ejecutar fuera de este proyecto. No descarga datos nuevos ni modifica las carpetas originales.')
p('Requisitos: Python 3.12 o compatible, pandas, NumPy, SciPy, Matplotlib, seaborn, statsmodels, arch, nbformat, nbclient, ipykernel y reportlab. Las versiones del entorno utilizado se registran en versiones_entorno.txt. Se recomienda instalarlas en un entorno virtual.')
p('Ejemplo de ejecución: python tp1_reproducible.py --output reproduccion_tp1. El programa requiere una carpeta vacía, extrae los archivos, ejecuta los puntos 2 al 12 en orden y genera nuevamente el informe. La opción --solo-extraer permite revisar el código y los datos antes de ejecutar los cálculos. El punto 1 es un texto y los puntos 13–14 se materializan en este informe y su estructura.')
p('Los módulos modelos_puntos56.py y modelos_resto.py contienen los cálculos comunes. Los notebooks explican cada decisión y llaman a esas funciones. crear_informe.py reúne las tablas y gráficos calculados. Los scripts build_* reconstruyen las celdas, pero no son necesarios para leer los notebooks entregados.')
table(pd.DataFrame([['5–6','Selección, parámetros y errores de entrenamiento/prueba'],['7','Comparación y predicciones de alternativas'],['8','Residuos, Ljung–Box, normalidad y ARCH'],['9','Pronósticos e intervalos de 30 días'],['10','VAR, criterios, coeficientes, errores y pronósticos'],['11','Granger, relaciones instantáneas y respuestas a shocks'],['12','Comparación de estacionalidad y elección por validación']],columns=['Puntos','Archivos principales']))
p('La repetición numérica puede variar ligeramente entre versiones de bibliotecas y sistemas. Los índices temporales, reglas de selección y períodos de evaluación quedan explícitos para detectar diferencias sustantivas.')

def footer(canvas,doc):
    canvas.setFont(font,7);canvas.setFillColor(colors.HexColor('#617582'))
    canvas.drawString(48,27,'TP1 · Series temporales · Rosario')
    canvas.drawRightString(A4[0]-48,27,str(doc.page))
    canvas.setTitle('TP1 — Series temporales de Rosario')
    canvas.setAuthor('Tomas del Bo; Matias Villanueva; Juan Ignacio Paberolis')

doc=SimpleDocTemplate(str(OUT/'TP1_informe.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=44)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'TP1_informe.md').write_text('\n'.join(md))
print(OUT/'TP1_informe.pdf')
