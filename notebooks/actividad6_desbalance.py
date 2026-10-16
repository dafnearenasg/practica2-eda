# %% [markdown]
# # Actividad 6. Tratamiento del desbalance de clases 
#
# **Autor:** Jorge Cruz Pinzón · Práctica 2, EDA y preparación de datasets
#
# **Uso de IA generativa:** se utilizó asistencia de IA generativa (Claude, de Anthropic) para
# estructurar este notebook. El autor lo ejecutó, revisó los resultados y puede explicar y
# modificar el cálculo del desbalance y cada técnica de balanceo.
#
# **Objetivo:** confirmar si cada dataset está desbalanceado (razón de desbalance
# `IR = N_mayoritaria / N_minoritaria`, con `IR > 1.5` como criterio) y medir el efecto de cinco
# estrategias sobre el desempeño de un clasificador:
#
# | Estrategia | Descripción | Riesgo principal |
# |---|---|---|
# | Sin balanceo | Línea base con la distribución original | Ignorar la clase minoritaria |
# | Ponderación de clases | Aumenta el costo de los errores en la clase minoritaria | Requiere calibración |
# | Sobremuestreo aleatorio | Replica casos minoritarios | Sobreajuste |
# | Submuestreo aleatorio | Elimina casos mayoritarios | Pérdida de información |
# | SMOTE | Genera ejemplos sintéticos de la clase minoritaria | Casos artificiales poco plausibles |
#
# **Datos de entrada:** los conjuntos train/test que deja la Actividad 5
# (`data/processed/<dataset>_<train|test>_winsor.csv`): imputados en la Actividad 4 y winsorizados
# en la 5 (Adult con percentiles 1–99 en `age`, `hours-per-week`, `capital-gain` y
# `capital-loss`; Telco sin winsorizar). Mantienen el mismo split 80/20 estratificado
# (`random_state = 42`) y la columna `target`. Para repetir el análisis sobre los datos solo
# imputados, cambiar `FUENTE` a `'imputado'`.
#
# **Regla anti-fuga:** el balanceo se aplica **solo sobre train**, dentro de un `Pipeline` de
# `imbalanced-learn`. El conjunto de prueba nunca se remuestrea, y la validación cruzada
# remuestrea únicamente los pliegues de entrenamiento.
#
# **Cómo ejecutarlo:** el notebook debe estar en `notebooks/`. Instalar dependencias con
# `pip install -r requirements.txt` y usar *Restart & Run All*. Con `RANDOM_STATE = 42` los
# resultados son idénticos entre corridas.
#
# **Estructura**
# 1. Carga de los datos preparados
# 2. Diagnóstico del desbalance (cantidad, porcentaje e IR)
# 3. Estrategias de balanceo y evaluación
# 4. Resultados: métricas, compromiso precision/recall, matrices de confusión y curvas
#    precision-recall
# 5. Pipeline final del equipo (Actividades 4, 5 y 6)
# 6. Conclusiones

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import imblearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

try:
    from IPython.display import display
except ImportError:          # por si se ejecuta como script normal
    display = print

RANDOM_STATE = 42            # mismo valor para todo el equipo
FUENTE = 'winsor'            # 'winsor' (Actividad 5, final) o 'imputado' (Actividad 4)
UMBRAL_IR = 1.5              # criterio del enunciado: IR > 1.5 = desbalanceado
PROC_DIR = Path('../data/processed')
FIG_DIR = Path('figuras')    # aquí se guardan gráficas y tablas para el informe
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)

NOMBRES = {'telco': 'Telco', 'adult': 'Adult'}
CLASES = {'telco': ('No abandona', 'Churn'), 'adult': ('<=50K', '>50K')}   # (clase 0, clase 1)

print(f'Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | '
      f'scikit-learn {sklearn.__version__} | imbalanced-learn {imblearn.__version__} | '
      f'matplotlib {matplotlib.__version__} | seaborn {sns.__version__}')
print(f'Fuente de datos: {FUENTE}')

# %% [markdown]
# ## 1. Carga de los datos preparados
# Cada archivo trae los atributos ya imputados (sin nulos) y la columna `target`
# (1 = clase positiva: Churn en Telco, ingreso >50K en Adult).

# %%
datos = {}
for ds in NOMBRES:
    for parte in ['train', 'test']:
        ruta = PROC_DIR / f'{ds}_{parte}_{FUENTE}.csv'
        if not ruta.exists():
            raise FileNotFoundError(
                f'No se encontró {ruta}. Ejecuta primero la Actividad 4 y la 5 (que generan '
                'los archivos de data/processed/) y revisa que el notebook esté en notebooks/.')
    tr = pd.read_csv(PROC_DIR / f'{ds}_train_{FUENTE}.csv')
    te = pd.read_csv(PROC_DIR / f'{ds}_test_{FUENTE}.csv')
    assert tr.isna().sum().sum() == 0 and te.isna().sum().sum() == 0, 'hay valores nulos'
    datos[ds] = dict(Xtr=tr.drop(columns='target'), ytr=tr['target'],
                     Xte=te.drop(columns='target'), yte=te['target'])
    print(f'{NOMBRES[ds]}: train {tr.shape}, test {te.shape}')

# %% [markdown]
# ## 2. Diagnóstico del desbalance
# Para cada dataset se calcula la cantidad y el porcentaje de observaciones por clase y la razón
# de desbalance `IR = N_mayoritaria / N_minoritaria`. Si `IR > 1.5`, el dataset se considera
# desbalanceado. Se reporta sobre el total y sobre cada partición, para comprobar que el split
# estratificado conserva la proporción.

# %%
conteos = []
resumen = []
for ds, d in datos.items():
    y_total = pd.concat([d['ytr'], d['yte']])
    c = y_total.value_counts().reindex([0, 1])
    for k in [0, 1]:
        conteos.append({'dataset': NOMBRES[ds], 'clase': CLASES[ds][k], 'target': k,
                        'cantidad': int(c[k]), 'porcentaje': round(c[k] / c.sum() * 100, 2)})
    for etiqueta, y in [('Total', y_total), ('Train', d['ytr']), ('Test', d['yte'])]:
        cc = y.value_counts()
        ir = cc.max() / cc.min()
        resumen.append({'dataset': NOMBRES[ds], 'conjunto': etiqueta, 'n': len(y),
                        'clase mayoritaria (n)': int(cc.max()),
                        'clase minoritaria (n)': int(cc.min()),
                        '% minoritaria': round(cc.min() / len(y) * 100, 2),
                        'IR': round(ir, 3), f'IR > {UMBRAL_IR}': 'Sí' if ir > UMBRAL_IR else 'No'})

tabla_conteos = pd.DataFrame(conteos)
tabla_ir = pd.DataFrame(resumen)
display(tabla_conteos)
display(tabla_ir)
tabla_conteos.to_csv(FIG_DIR / f'balanceo_conteo_clases_{FUENTE}.csv', index=False)
tabla_ir.to_csv(FIG_DIR / f'balanceo_razon_desbalance_{FUENTE}.csv', index=False)

for ds in NOMBRES:
    fila = tabla_ir.query("dataset == @NOMBRES[@ds] and conjunto == 'Total'").iloc[0]
    veredicto = 'está desbalanceado' if fila['IR'] > UMBRAL_IR else 'no está desbalanceado'
    print(f"{fila['dataset']}: IR = {fila['IR']:.2f} -> {veredicto} (criterio IR > {UMBRAL_IR})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, ds in zip(axes, NOMBRES):
    t = tabla_conteos[tabla_conteos.dataset == NOMBRES[ds]]
    barras = ax.bar(t['clase'], t['cantidad'], color=sns.color_palette()[:2])
    for b, p in zip(barras, t['porcentaje']):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():,.0f}\n({p}%)',
                ha='center', va='bottom')
    ir = tabla_ir.query("dataset == @NOMBRES[@ds] and conjunto == 'Total'")['IR'].iloc[0]
    ax.set_title(f'{NOMBRES[ds]}: distribución de clases (IR = {ir:.2f})')
    ax.set_ylabel('Observaciones')
    ax.set_ylim(0, t['cantidad'].max() * 1.18)
plt.tight_layout()
plt.savefig(FIG_DIR / f'balanceo_distribucion_clases_{FUENTE}.png', dpi=150)
plt.show()

# %% [markdown]
# **Lectura:** ambos datasets superan el umbral de 1.5, por lo que requieren tratamiento. Además,
# un clasificador que siempre predijera la clase mayoritaria tendría una accuracy igual al
# porcentaje de esa clase (ver sección 4), sin detectar ningún caso positivo. Por eso el
# análisis se centra en precision, recall, F1 y PR-AUC de la clase positiva.

# %% [markdown]
# ## 3. Estrategias de balanceo y evaluación
#
# Todas las estrategias usan el **mismo preprocesamiento** (estandarizar numéricas, codificar
# categóricas con one-hot) y el **mismo clasificador** (regresión logística), de modo que la única
# diferencia es el tratamiento del desbalance. El remuestreo va entre el preprocesamiento y el
# modelo, dentro del `Pipeline`, y solo se aplica al ajustar con datos de entrenamiento.
#
# Métricas de la clase positiva sobre el conjunto de prueba: precision, recall, F1 y PR-AUC
# (average precision), con umbral de decisión 0.5. Se reporta también la accuracy, únicamente
# para mostrar por qué no es un buen criterio con clases desbalanceadas. Para dimensionar el
# ruido se agrega F1 y recall con validación cruzada estratificada de 5 pliegues sobre train
# (el remuestreo se hace dentro de cada pliegue de entrenamiento).

# %%
ESTRATEGIAS = ['Sin balanceo', 'Ponderación de clases', 'Sobremuestreo aleatorio',
               'Submuestreo aleatorio', 'SMOTE']


def construir_prep(X):
    num = X.select_dtypes('number').columns.tolist()
    cat = X.select_dtypes(exclude='number').columns.tolist()
    return ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])


def crear_muestreador(nombre):
    """Devuelve un remuestreador nuevo (o None). Se crea uno por pipeline para no compartir estado."""
    return {'Sobremuestreo aleatorio': RandomOverSampler(random_state=RANDOM_STATE),
            'Submuestreo aleatorio': RandomUnderSampler(random_state=RANDOM_STATE),
            'SMOTE': SMOTE(random_state=RANDOM_STATE)}.get(nombre)


def crear_pipeline(nombre, X):
    pasos = [('prep', construir_prep(X))]
    muestreador = crear_muestreador(nombre)
    if muestreador is not None:
        pasos.append(('remuestreo', muestreador))
    pesos = 'balanced' if nombre == 'Ponderación de clases' else None
    pasos.append(('modelo', LogisticRegression(max_iter=2000, class_weight=pesos)))
    return Pipeline(pasos)


def tamano_entrenamiento(nombre, Xtr, ytr):
    """Filas y % de positivos que ve el modelo al entrenar (tras remuestrear)."""
    muestreador = crear_muestreador(nombre)
    if muestreador is None:
        return len(ytr), ytr.mean() * 100
    Z = construir_prep(Xtr).fit_transform(Xtr)
    _, y_r = muestreador.fit_resample(Z, ytr)
    return len(y_r), float(np.mean(y_r)) * 100


def mas_menos(serie):
    return f'{serie.mean():.4f} ± {serie.std():.4f}'


# %%
filas, matrices, probas = [], {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

for ds, d in datos.items():
    for nombre in ESTRATEGIAS:
        pipe = crear_pipeline(nombre, d['Xtr'])
        pipe.fit(d['Xtr'], d['ytr'])
        pred = pipe.predict(d['Xte'])
        prob = pipe.predict_proba(d['Xte'])[:, 1]
        tn, fp, fn, tp = confusion_matrix(d['yte'], pred).ravel()
        matrices[(ds, nombre)] = np.array([[tn, fp], [fn, tp]])
        probas[(ds, nombre)] = prob

        n_filas, pct_pos = tamano_entrenamiento(nombre, d['Xtr'], d['ytr'])
        cvr = cross_validate(crear_pipeline(nombre, d['Xtr']), d['Xtr'], d['ytr'], cv=cv,
                             scoring={'f1': 'f1', 'recall': 'recall'})
        filas.append({
            'dataset': NOMBRES[ds], 'estrategia': nombre,
            'filas train': n_filas, '% positivos en train': round(pct_pos, 1),
            'accuracy': accuracy_score(d['yte'], pred),
            'precision': precision_score(d['yte'], pred),
            'recall': recall_score(d['yte'], pred),
            'f1': f1_score(d['yte'], pred),
            'pr_auc': average_precision_score(d['yte'], prob),
            'F1 CV (5 pliegues)': mas_menos(cvr['test_f1']),
            'Recall CV (5 pliegues)': mas_menos(cvr['test_recall']),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn})

resultados = pd.DataFrame(filas).set_index(['dataset', 'estrategia'])

# %% [markdown]
# ## 4. Resultados

# %% [markdown]
# ### 4.1 Métricas por estrategia (conjunto de prueba)

# %%
cols_metricas = ['filas train', '% positivos en train', 'accuracy', 'precision', 'recall', 'f1',
                 'pr_auc', 'F1 CV (5 pliegues)', 'Recall CV (5 pliegues)']
display(resultados[cols_metricas].round(4))
resultados[cols_metricas].round(4).to_csv(FIG_DIR / f'balanceo_resultados_{FUENTE}.csv')

print('Accuracy de un clasificador que siempre predice la clase mayoritaria (test):')
for ds, d in datos.items():
    print(f"  {NOMBRES[ds]}: {(1 - d['yte'].mean()):.4f}")

# %% [markdown]
# ### 4.2 Compromiso precision / recall frente a la línea base
# Cuánto cambian las métricas y cuántos positivos más se detectan (TP) a costa de cuántas falsas
# alarmas más (FP), en comparación con "Sin balanceo".

# %%
comp = []
for ds in NOMBRES:
    base = resultados.loc[(NOMBRES[ds], 'Sin balanceo')]
    for nombre in ESTRATEGIAS[1:]:
        r = resultados.loc[(NOMBRES[ds], nombre)]
        comp.append({'dataset': NOMBRES[ds], 'estrategia': nombre,
                     'Δ precision': r['precision'] - base['precision'],
                     'Δ recall': r['recall'] - base['recall'],
                     'Δ F1': r['f1'] - base['f1'],
                     'Δ PR-AUC': r['pr_auc'] - base['pr_auc'],
                     'positivos detectados de más (TP)': int(r['TP'] - base['TP']),
                     'falsas alarmas de más (FP)': int(r['FP'] - base['FP'])})
tabla_comp = pd.DataFrame(comp).set_index(['dataset', 'estrategia'])
display(tabla_comp.round(4))
tabla_comp.round(4).to_csv(FIG_DIR / f'balanceo_comparacion_vs_base_{FUENTE}.csv')

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, ds in zip(axes, NOMBRES):
    sub = resultados.loc[NOMBRES[ds], ['precision', 'recall', 'f1']].loc[ESTRATEGIAS]
    sub.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{NOMBRES[ds]}: métricas de la clase positiva por estrategia')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', ncol=3)
    ax.tick_params(axis='x', rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
axes[0].set_ylabel('Valor')
plt.tight_layout()
plt.savefig(FIG_DIR / f'balanceo_metricas_por_estrategia_{FUENTE}.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.3 Matrices de confusión (conjunto de prueba)

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for i, ds in enumerate(NOMBRES):
    for j, nombre in enumerate(ESTRATEGIAS):
        ax = axes[i, j]
        sns.heatmap(matrices[(ds, nombre)], annot=True, fmt='d', cbar=False, cmap='Blues', ax=ax,
                    xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        ax.set_title(f'{NOMBRES[ds]}: {nombre}', fontsize=10)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real' if j == 0 else '')
plt.tight_layout()
plt.savefig(FIG_DIR / f'balanceo_matrices_confusion_{FUENTE}.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.4 Curvas precision-recall: ¿el balanceo mejora al modelo o solo mueve el umbral?
# Cada curva recorre todos los umbrales de decisión posibles; el punto marcado es el umbral 0.5
# que usa `predict`. Si las curvas de dos estrategias se enciman, el modelo tiene la misma
# capacidad de separar las clases y la diferencia entre ellas es solo el punto de operación.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ds in zip(axes, NOMBRES):
    yte = datos[ds]['yte']
    for k, nombre in enumerate(ESTRATEGIAS):
        prec, rec, _ = precision_recall_curve(yte, probas[(ds, nombre)])
        color = sns.color_palette()[k]
        ax.plot(rec, prec, color=color, linewidth=1.6,
                label=f"{nombre} (PR-AUC = {resultados.loc[(NOMBRES[ds], nombre), 'pr_auc']:.3f})")
        r = resultados.loc[(NOMBRES[ds], nombre)]
        ax.scatter(r['recall'], r['precision'], color=color, s=60, edgecolor='black', zorder=3)
    ax.axhline(yte.mean(), color='gray', linestyle='--', linewidth=1,
               label=f'Azar (proporción de positivos = {yte.mean():.3f})')
    ax.set_title(f'{NOMBRES[ds]}: curvas precision-recall (punto = umbral 0.5)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.savefig(FIG_DIR / f'balanceo_curvas_pr_{FUENTE}.png', dpi=150)
plt.show()

# %% [markdown]
# ### 4.5 Resumen automático

# %%
for ds in NOMBRES:
    sub = resultados.loc[NOMBRES[ds]]
    print(f'{NOMBRES[ds]}')
    print(f"  Mejor F1 (test):      {sub['f1'].idxmax()} ({sub['f1'].max():.4f})")
    print(f"  Mejor recall (test):  {sub['recall'].idxmax()} ({sub['recall'].max():.4f})")
    print(f"  Mejor precision:      {sub['precision'].idxmax()} ({sub['precision'].max():.4f})")
    print(f"  Mejor PR-AUC:         {sub['pr_auc'].idxmax()} ({sub['pr_auc'].max():.4f})")

# %% [markdown]
# ## 5. Pipeline final del equipo (Actividades 4, 5 y 6)
# Se compara el efecto acumulado de las decisiones del equipo con el mismo modelo y el mismo
# conjunto de prueba: (1) solo imputación (Actividad 4), (2) imputación + winsorización
# (Actividad 5) y (3) imputación + winsorización + ponderación de clases (recomendación de esta
# actividad).

# %%
def cargar(ds, fuente):
    tr = pd.read_csv(PROC_DIR / f'{ds}_train_{fuente}.csv')
    te = pd.read_csv(PROC_DIR / f'{ds}_test_{fuente}.csv')
    return tr.drop(columns='target'), tr['target'], te.drop(columns='target'), te['target']


ETAPAS = [('1. Imputación (A4)', 'imputado', 'Sin balanceo'),
          ('2. + Winsorización (A5)', 'winsor', 'Sin balanceo'),
          ('3. + Ponderación de clases (A6)', 'winsor', 'Ponderación de clases')]

filas_final = []
for ds in NOMBRES:
    for etapa, fuente, estrategia in ETAPAS:
        Xtr, ytr, Xte, yte = cargar(ds, fuente)
        pipe = crear_pipeline(estrategia, Xtr).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:, 1]
        cvr = cross_validate(crear_pipeline(estrategia, Xtr), Xtr, ytr, cv=cv, scoring='f1')
        filas_final.append({'dataset': NOMBRES[ds], 'etapa': etapa,
                            'precision': precision_score(yte, pred),
                            'recall': recall_score(yte, pred),
                            'f1': f1_score(yte, pred),
                            'pr_auc': average_precision_score(yte, prob),
                            'F1 CV (5 pliegues)': mas_menos(cvr['test_score'])})

tabla_final = pd.DataFrame(filas_final).set_index(['dataset', 'etapa'])
display(tabla_final.round(4))
tabla_final.round(4).to_csv(FIG_DIR / 'pipeline_final_equipo.csv')

# %% [markdown]
# ## 6. Conclusiones de la Actividad 6
#
# (Valores de la corrida con `FUENTE = 'winsor'` y `RANDOM_STATE = 42`.)
#
# - **Desbalance:** Telco tiene IR = 2.77 (26.54% de Churn) y Adult tiene IR = 3.15 (24.09% de
#   ingreso >50K). Ambos superan el umbral de 1.5, y el split estratificado conserva la proporción
#   en train y en test.
# - **La accuracy engaña:** la línea base tiene accuracy de 0.805 (Telco) y 0.857 (Adult), pero un
#   clasificador que siempre dijera "no" tendría 0.735 y 0.759. El recall de la línea base es solo
#   0.556 (Telco) y 0.626 (Adult): deja sin detectar el 44% de los clientes que abandonan y el 37%
#   de las personas con ingreso >50K.
# - **Efecto del balanceo:** las cuatro técnicas suben el recall unos 0.23 (Telco: de 0.556 a
#   0.786–0.797; Adult: de 0.626 a 0.854–0.859) y bajan la precision unos 0.15–0.17 (Telco: de
#   0.656 a 0.499–0.510; Adult: de 0.739 a 0.571–0.576). El F1 mejora poco en prueba (+0.009 a
#   +0.017 en Telco; +0.008 a +0.011 en Adult) y algo más en validación cruzada (de 0.593 a
#   0.626–0.630 en Telco; de 0.658 a 0.679–0.681 en Adult). La accuracy baja entre 6 y 7 puntos
#   en Telco y entre 4 y 5 en Adult.
# - **Compromiso:** en Telco se detectan entre 86 y 90 clientes que abandonan más, a cambio de
#   entre 173 y 186 falsas alarmas más; en Adult, entre 358 y 366 positivos más a cambio de entre
#   642 y 666 falsas alarmas más.
# - **El balanceo mueve el umbral, no mejora al modelo:** la PR-AUC no sube (baja entre 0.001 y
#   0.007) y las curvas precision-recall de las cinco estrategias se enciman. Todas tienen la misma
#   capacidad de separar las clases; lo que cambia es el punto de operación sobre la misma curva.
# - **Entre técnicas no hay un ganador claro:** las diferencias de F1 y recall entre ponderación,
#   sobremuestreo, submuestreo y SMOTE son menores que la desviación de la validación cruzada
#   (±0.02 en Telco, ±0.005 en Adult). Lo que las distingue es el riesgo: el sobremuestreo y SMOTE
#   aumentan el train de 5,634 a 8,278 filas (Telco) y de 26,029 a 39,516 (Adult); el submuestreo
#   descarta casi la mitad (a 2,990 y 12,542 filas); y SMOTE crea puntos sintéticos.
# - **Pipeline final del equipo:** imputación (A4) + winsorización (A5) + ponderación de clases
#   (A6). Frente a la línea base, el F1 en validación cruzada sube de 0.593 a 0.629 en Telco y de
#   0.658 a 0.681 en Adult, y el recall de 0.556 a 0.786 y de 0.626 a 0.858. La winsorización por
#   sí sola no cambia el desempeño (Telco no se winsorizó; en Adult el F1 de prueba pasa de 0.675
#   a 0.678).
# - **Recomendación:** si perder un caso positivo cuesta más que una falsa alarma (por ejemplo, no
#   detectar a un cliente que va a abandonar), usar **ponderación de clases**: logra el mismo
#   efecto que las otras técnicas sin duplicar, eliminar ni inventar observaciones, y no cambia el
#   tamaño del entrenamiento. Si importa más la precision, conservar la línea base o elegir el
#   umbral con validación cruzada según el costo de cada error. En ambos casos, reportar precision,
#   recall, F1 y PR-AUC, no la accuracy.
#
# **Limitaciones:** un solo modelo (regresión logística) y un umbral fijo de 0.5; un solo split de
# prueba (la validación cruzada dimensiona el ruido); `SMOTE` se aplicó sobre columnas ya
# codificadas con one-hot, donde los puntos sintéticos pueden tomar valores no binarios (`SMOTENC`
# sería una alternativa); y las probabilidades de los modelos balanceados quedan sesgadas hacia la
# clase positiva, por lo que no deben interpretarse como probabilidades reales sin calibrarlas.
