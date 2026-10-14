# %% [markdown]
# # Actividad 4. Imputación de valores perdidos (Integrante 3)
#
# **Objetivo:** comparar dos estrategias de imputación para una variable numérica
# (`TotalCharges`, Telco Churn) y una categórica (`occupation`, Adult Income), y analizar
# su efecto en la distribución, la varianza, las concentraciones artificiales y el
# desempeño de un clasificador.
#
# | Estrategia | Numérica | Categórica |
# |---|---|---|
# | A | Media | Moda |
# | B | Mediana | Categoría "Desconocido" |
# | Opcional | KNN Imputer | — |
# | Extra (justificada) | Cero (faltantes estructurales de `TotalCharges`) | — |
#
# **Autor:** Jorge Cruz Pinzón (Integrante 3) · Práctica 2, EDA y preparación de datasets
#
# **Uso de IA generativa:** se utilizó asistencia de IA generativa (Claude, de Anthropic) para
# estructurar este notebook. El autor lo ejecutó, revisó los resultados y puede explicar y
# modificar el método de imputación y la decisión final.
#
# **Cómo ejecutarlo:** el notebook debe estar en `notebooks/` y los archivos `adult.csv` y
# `telco_churn.csv` en `data/`. Instalar dependencias con `pip install -r requirements.txt` y
# usar *Restart & Run All*. Con `RANDOM_STATE = 42` los resultados son idénticos entre corridas.
#
# **Estructura**
# 1. Carga y limpieza (igual que el perfilado de la Actividad 2)
# 2. Imputación numérica: `TotalCharges` (diagnóstico descriptivo + simulación)
# 3. Imputación categórica: `occupation`
# 4. Efecto en el clasificador (con split train/test, sin fuga de información)

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score)

try:
    from IPython.display import display
except ImportError:          # por si se ejecuta como script normal
    display = print

import sys
import sklearn
import matplotlib

RANDOM_STATE = 42            # mismo valor para todo el equipo
DATA_DIR = Path('../data')   # carpeta con adult.csv y telco_churn.csv
FIG_DIR = Path('figuras')    # aquí se guardan gráficas y tablas para el informe
FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)

for archivo in ['adult.csv', 'telco_churn.csv']:
    if not (DATA_DIR / archivo).exists():
        raise FileNotFoundError(
            f'No se encontró {DATA_DIR / archivo}. Revisa que el notebook esté en notebooks/ '
            'y que los CSV estén en data/.')

print(f'Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | '
      f'scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__} | '
      f'seaborn {sns.__version__}')

# %% [markdown]
# ## 1. Carga y limpieza
# Igual que la Actividad 2: `?` como valor ausente en Adult y `TotalCharges` convertida a
# numérica en Telco (las celdas vacías pasan a `NaN`). Se aplica la limpieza detectada en
# el perfilado: duplicados de Adult, `fnlwgt`, `education` (redundante con `education-num`)
# y `customerID`. En Telco se unifican "No internet service" y "No phone service" con "No".

# %%
nombres_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                 'income']

adult = pd.read_csv(DATA_DIR / 'adult.csv', skipinitialspace=True, na_values='?',
                    header=None, names=nombres_adult)
telco = pd.read_csv(DATA_DIR / 'telco_churn.csv')
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')

adult_clean = (adult.drop_duplicates()
                    .drop(columns=['fnlwgt', 'education'])
                    .reset_index(drop=True))
telco_clean = (telco.drop(columns=['customerID'])
                    .replace({'No internet service': 'No', 'No phone service': 'No'}))

# Variable objetivo (1 = clase positiva, que es la minoritaria) y atributos
y_adult = adult_clean['income'].eq('>50K').astype(int)
X_adult = adult_clean.drop(columns='income')
y_telco = telco_clean['Churn'].eq('Yes').astype(int)
X_telco = telco_clean.drop(columns='Churn')

print(f'Adult: {X_adult.shape} | Telco: {X_telco.shape}')
print('\nValores faltantes por columna (solo las que tienen):')
for nombre, X in [('Adult', X_adult), ('Telco', X_telco)]:
    falt = X.isna().sum()
    print(f'\n{nombre}')
    print(pd.DataFrame({'faltantes': falt, '%': (falt / len(X) * 100).round(2)})
            .query('faltantes > 0'))

# %% [markdown]
# ## 2. Imputación numérica: `TotalCharges` (Telco)
#
# **Por qué esta variable:** es la única variable numérica con faltantes entre los dos
# datasets (Adult solo tiene faltantes en categóricas).
#
# En esta sección la imputación se calcula sobre todo el dataset porque es un
# **diagnóstico descriptivo** (no se entrena ningún modelo). La evaluación con el
# clasificador se hace en la sección 4, con split y sin fuga de información.

# %% [markdown]
# ### 2.1 Diagnóstico de los faltantes
# Los faltantes de `TotalCharges` ¿son aleatorios? Se revisa su relación con `tenure`.

# %%
col = 'TotalCharges'
tc = X_telco[col]
falt = tc.isna()

print(f'Faltantes en {col}: {falt.sum()} de {len(tc)} ({falt.mean() * 100:.2f}%)')
print(f'Faltantes con tenure == 0: {(X_telco.loc[falt, "tenure"] == 0).sum()} de {falt.sum()}')
print(f'Clientes con tenure == 0 en total: {(X_telco["tenure"] == 0).sum()}')
print('Tasa de churn en los clientes con TotalCharges faltante:',
      round(y_telco[falt].mean(), 3), '| global:', round(y_telco.mean(), 3))

# %% [markdown]
# Si todos los faltantes corresponden a `tenure == 0`, son **faltantes estructurales**:
# el cliente aún no ha sido facturado, así que el valor lógico es 0. Por eso se agrega una
# estrategia extra (C: cero) además de A y B, y se discute en el informe.

# %% [markdown]
# ### 2.2 Estrategias de imputación

# %%
COLS_KNN = ['tenure', 'MonthlyCharges', 'TotalCharges']


def imputar_knn(df, objetivo='TotalCharges', cols=COLS_KNN, k=5):
    """KNN Imputer con variables escaladas (si no, MonthlyCharges y TotalCharges
    dominan la distancia). Devuelve la columna objetivo ya imputada."""
    esc = StandardScaler()
    Z = esc.fit_transform(df[cols])                  # ignora NaN al ajustar
    Z_imp = KNNImputer(n_neighbors=k).fit_transform(Z)
    return pd.Series(esc.inverse_transform(Z_imp)[:, cols.index(objetivo)],
                     index=df.index, name=objetivo)


tc_A = tc.fillna(tc.mean())                 # Estrategia A: media
tc_B = tc.fillna(tc.median())               # Estrategia B: mediana
tc_knn = imputar_knn(X_telco)               # Opcional: KNN
tc_C = tc.fillna(0)                         # Extra: cero (faltantes estructurales)

# Indicador binario de ausencia (informativo: coincide con tenure == 0)
X_telco['TotalCharges_faltante'] = falt.astype(int)

versiones = {
    'Original (sin imputar)': tc,
    'A: Media': tc_A,
    'B: Mediana': tc_B,
    'Opcional: KNN (k=5)': tc_knn,
    'Extra C: Cero': tc_C,
}


def resumen(s):
    return pd.Series({'n': s.count(), 'media': s.mean(), 'mediana': s.median(),
                      'varianza': s.var(), 'desv_est': s.std(),
                      'asimetria': s.skew(), 'min': s.min(), 'max': s.max()})


tabla_num = pd.DataFrame({k: resumen(v) for k, v in versiones.items()}).T
display(tabla_num.round(2))
tabla_num.round(4).to_csv(FIG_DIR / 'imputacion_numerica_estadisticos.csv')

# %% [markdown]
# ### 2.3 ¿Qué valor recibió cada cliente con faltante?

# %%
filas_imputadas = pd.DataFrame({
    'tenure': X_telco.loc[falt, 'tenure'],
    'MonthlyCharges': X_telco.loc[falt, 'MonthlyCharges'],
    'A_media': tc_A[falt],
    'B_mediana': tc_B[falt],
    'KNN': tc_knn[falt],
    'C_cero': tc_C[falt],
})
display(filas_imputadas.round(2))
filas_imputadas.round(2).to_csv(FIG_DIR / 'imputacion_numerica_filas_imputadas.csv')

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Distribución completa (con 11 faltantes de 7,043 las curvas casi se empalman)
for nombre, serie, estilo in [('Original (sin imputar)', tc.dropna(), '-'),
                              ('A: Media', tc_A, '--'),
                              ('B: Mediana', tc_B, ':'),
                              ('KNN (k=5)', tc_knn, '-.')]:
    sns.kdeplot(serie, ax=axes[0], linestyle=estilo, linewidth=2, label=nombre,
                clip=(0, None))
axes[0].set_title('Distribución de TotalCharges tras la imputación')
axes[0].set_xlabel('TotalCharges')
axes[0].legend()

# Valor imputado promedio en los clientes con tenure == 0
prom = pd.Series({'A: Media': tc_A[falt].mean(), 'B: Mediana': tc_B[falt].mean(),
                  'KNN': tc_knn[falt].mean(), 'C: Cero': tc_C[falt].mean()})
prom.plot(kind='bar', ax=axes[1], color=sns.color_palette()[:4])
axes[1].set_title('Valor imputado en los clientes con tenure = 0')
axes[1].set_ylabel('TotalCharges imputado (promedio)')
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(prom):
    axes[1].text(i, v, f'{v:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(FIG_DIR / 'imputacion_numerica_totalcharges.png', dpi=150)
plt.show()

# %% [markdown]
# **Lectura:** con tan pocos faltantes, ninguna estrategia cambia de forma apreciable la
# media, la varianza ni la forma de la distribución. El problema es otro: a clientes que aún
# no han pagado nada, la media y la mediana les asignan un cargo acumulado de cientos o miles
# de dólares. KNN les asigna aproximadamente su cargo mensual (los vecinos más cercanos son
# clientes con 1 mes de antigüedad), que es mucho más cercano a la realidad. Como son
# faltantes estructurales, el valor coherente es 0.

# %% [markdown]
# ### 2.4 Simulación: ¿cómo se comparan las estrategias con más faltantes?
# Con 11 faltantes no se aprecia el efecto de cada estrategia sobre la varianza. Para
# observarlo se **simula** ausencia aleatoria (MCAR) en el 15% de los valores conocidos de
# `TotalCharges`, se imputa y se compara contra el valor real. Esto es un experimento
# controlado y debe declararse como tal en el reporte (no son faltantes reales).

# %%
conocidos = X_telco.loc[tc.notna(), COLS_KNN].reset_index(drop=True)
var_real = conocidos[col].var()
FRAC, N_REP = 0.15, 10

registros = []
for r in range(N_REP):
    rng = np.random.default_rng(RANDOM_STATE + r)
    mascara = rng.random(len(conocidos)) < FRAC
    parcial = conocidos.copy()
    parcial.loc[mascara, col] = np.nan
    real = conocidos.loc[mascara, col]
    s = parcial[col]
    imputados = {
        'A: Media': s.fillna(s.mean()),
        'B: Mediana': s.fillna(s.median()),
        'KNN (k=5)': imputar_knn(parcial),
    }
    for nombre, imp in imputados.items():
        registros.append({
            'estrategia': nombre,
            'rmse': np.sqrt(((imp[mascara] - real) ** 2).mean()),
            'desv_est': imp.std(),
            'varianza': imp.var(),
            'varianza / varianza real': imp.var() / var_real,
        })

sim = pd.DataFrame(registros).groupby('estrategia').mean().round(3)
print(f'Varianza real de TotalCharges (sin faltantes): {var_real:,.0f}')
print(f'Promedio de {N_REP} repeticiones con {FRAC:.0%} de faltantes simulados')
display(sim)
sim.to_csv(FIG_DIR / 'imputacion_numerica_simulacion.csv')

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sim['rmse'].plot(kind='bar', ax=axes[0], color=sns.color_palette()[:3])
axes[0].set_title('Error de reconstrucción (RMSE, menor es mejor)')
axes[0].tick_params(axis='x', rotation=0)
sim['varianza / varianza real'].plot(kind='bar', ax=axes[1], color=sns.color_palette()[:3])
axes[1].axhline(1, color='k', linestyle='--', linewidth=1)
axes[1].set_title('Varianza tras imputar / varianza real')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / 'imputacion_numerica_simulacion.png', dpi=150)
plt.show()

# %% [markdown]
# **Lectura:** al rellenar con un valor único (media o mediana) se crea una concentración
# artificial en un punto, la varianza baja respecto a la real (razón menor que 1) y el error
# de reconstrucción es grande. KNN obtiene un error mucho menor y conserva la varianza,
# porque `TotalCharges` es casi `tenure × MonthlyCharges` y KNN usa esas dos variables.

# %% [markdown]
# ## 3. Imputación categórica: `occupation` (Adult)
#
# **Por qué esta variable:** tiene la mayor cantidad de faltantes entre las categóricas
# de Adult (`workclass` y `occupation` suelen faltar en las mismas filas).

# %% [markdown]
# ### 3.1 Diagnóstico: ¿la ausencia es informativa?

# %%
col_cat = 'occupation'
occ = X_adult[col_cat]
print(f'Faltantes en {col_cat}: {occ.isna().sum()} de {len(occ)} ({occ.isna().mean() * 100:.2f}%)')

print('\nCoincidencia de faltantes entre occupation y workclass:')
display(pd.crosstab(occ.isna(), X_adult['workclass'].isna(),
                    rownames=['occupation faltante'], colnames=['workclass faltante']))

tasa = y_adult.groupby(occ.isna().rename('occupation_faltante')).mean().round(3)
print('\nProporción de ingreso >50K según si occupation falta o no:')
display(tasa)

# %% [markdown]
# Si la proporción de ingreso alto es distinta entre quienes tienen y quienes no tienen
# `occupation`, la ausencia contiene información. En ese caso, imputar con la moda la
# borra, y conviene usar "Desconocido" o un indicador binario. En estos datos la proporción de
# ingreso >50K es de 24.9% cuando `occupation` está presente y de 10.4% cuando falta, así que
# la ausencia es informativa y el indicador está justificado.

# %% [markdown]
# ### 3.2 Estrategias de imputación

# %%
moda = occ.mode()[0]
occ_A = occ.fillna(moda)             # Estrategia A: moda
occ_B = occ.fillna('Desconocido')    # Estrategia B: categoría "Desconocido"
X_adult['occupation_faltante'] = occ.isna().astype(int)   # indicador binario

print('Moda de occupation:', moda)

orig = occ.fillna('Desconocido').value_counts(normalize=True) * 100
tabla_cat = pd.DataFrame({
    'Original (NaN como Desconocido, %)': orig,
    'A: Moda (%)': occ_A.value_counts(normalize=True) * 100,
    'B: Desconocido (%)': occ_B.value_counts(normalize=True) * 100,
}).fillna(0)
tabla_cat['Cambio de A vs original (pp)'] = (tabla_cat['A: Moda (%)']
                                              - tabla_cat['Original (NaN como Desconocido, %)'])
tabla_cat = tabla_cat.sort_values('Original (NaN como Desconocido, %)', ascending=False)
display(tabla_cat.round(2))
tabla_cat.round(2).to_csv(FIG_DIR / 'imputacion_categorica_frecuencias.csv')

# %%
tabla_cat.iloc[:, :3].plot(kind='barh', figsize=(10, 6), width=0.8)
plt.gca().invert_yaxis()
plt.title('Frecuencias de occupation según la estrategia de imputación')
plt.xlabel('Porcentaje (%)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'imputacion_categorica_occupation.png', dpi=150)
plt.show()

# %% [markdown]
# **Lectura:** la moda concentra todos los faltantes en una sola categoría real
# (`Prof-specialty`), que aumenta artificialmente su peso. "Desconocido" conserva ese grupo
# como categoría propia y no altera la frecuencia de las demás.

# %% [markdown]
# ## 4. Efecto en el desempeño del clasificador
#
# Ahora sí se hace el split (estratificado, `random_state=42`, el mismo para todo el equipo).
# La imputación va dentro de un `Pipeline`, por lo que la media, la mediana y la moda se
# calculan **solo con el conjunto de entrenamiento** (sin fuga de información).
#
# Métricas: precision, recall, F1 y PR-AUC de la clase positiva. No se usa accuracy
# como criterio principal por el desbalance de clases.
#
# **Nota:** en los modelos, las estrategias A y B se aplican a **todas** las columnas con
# faltantes (en Adult: `workclass`, `occupation` y `native-country`).

# %%
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    X_adult.drop(columns='occupation_faltante'), y_adult,
    test_size=0.2, stratify=y_adult, random_state=RANDOM_STATE)
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
    X_telco.drop(columns='TotalCharges_faltante'), y_telco,
    test_size=0.2, stratify=y_telco, random_state=RANDOM_STATE)

print('Adult:', Xa_tr.shape, Xa_te.shape, '| Telco:', Xt_tr.shape, Xt_te.shape)


def agregar_indicadores(X, columnas):
    """Agrega columnas binarias <col>_faltante. Depende solo de la fila, no hay fuga."""
    X = X.copy()
    for c in columnas:
        X[f'{c}_faltante'] = X[c].isna().astype(int)
    return X


def construir_prep(X, num_strategy, cat_strategy, cat_fill=None, num_fill=None):
    num = X.select_dtypes('number').columns.tolist()
    cat = X.select_dtypes(exclude='number').columns.tolist()
    num_pipe = Pipeline([('imp', SimpleImputer(strategy=num_strategy, fill_value=num_fill)),
                         ('esc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy=cat_strategy, fill_value=cat_fill)),
                         ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)])


def evaluar(pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    prob = pipe.predict_proba(Xte)[:, 1]
    return {'precision': precision_score(yte, pred),
            'recall': recall_score(yte, pred),
            'f1': f1_score(yte, pred),
            'pr_auc': average_precision_score(yte, prob)}


def f1_cv(pipe, Xtr, ytr):
    """F1 con validación cruzada de 5 pliegues sobre train (media ± desviación)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring='f1')['test_score']
    return f'{s.mean():.4f} ± {s.std():.4f}'


# %%
experimentos = {
    'A: media / moda': dict(num_strategy='mean', cat_strategy='most_frequent'),
    'B: mediana / Desconocido': dict(num_strategy='median', cat_strategy='constant',
                                     cat_fill='Desconocido'),
    'B + indicador de ausencia': dict(num_strategy='median', cat_strategy='constant',
                                      cat_fill='Desconocido', indicador=True),
    'C: cero / Desconocido (solo Telco)': dict(num_strategy='constant', num_fill=0,
                                               cat_strategy='constant',
                                               cat_fill='Desconocido'),
}

datasets = {
    'Telco': dict(Xtr=Xt_tr, ytr=yt_tr, Xte=Xt_te, yte=yt_te, flags=['TotalCharges']),
    'Adult': dict(Xtr=Xa_tr, ytr=ya_tr, Xte=Xa_te, yte=ya_te,
                  flags=['occupation', 'workclass', 'native-country']),
}

filas = []
for nombre, d in datasets.items():
    for exp, cfg in experimentos.items():
        if exp.startswith('C') and nombre == 'Adult':
            continue                       # Adult no tiene faltantes numéricos
        cfg = dict(cfg)
        con_ind = cfg.pop('indicador', False)
        Xtr = agregar_indicadores(d['Xtr'], d['flags']) if con_ind else d['Xtr']
        Xte = agregar_indicadores(d['Xte'], d['flags']) if con_ind else d['Xte']
        pipe = Pipeline([('prep', construir_prep(Xtr, **cfg)),
                         ('modelo', LogisticRegression(max_iter=2000))])
        res = evaluar(pipe, Xtr, d['ytr'], Xte, d['yte'])
        res['F1 CV (5 pliegues, train)'] = f1_cv(pipe, Xtr, d['ytr'])
        filas.append({'dataset': nombre, 'estrategia': exp, **res})

resultados = pd.DataFrame(filas).set_index(['dataset', 'estrategia'])
display(resultados.round(4))
resultados.round(4).to_csv(FIG_DIR / 'imputacion_desempeno_modelo.csv')

# %% [markdown]
# ## 5. Decisión final y entrega para la Actividad 5 (winsorización)
#
# **Decisión de imputación del equipo**
# - `TotalCharges` (Telco): se imputa con **0**, porque los 11 faltantes son estructurales
#   (clientes con `tenure = 0`). No se agrega indicador: sería idéntico a `tenure == 0`.
# - `workclass`, `occupation` y `native-country` (Adult): se imputan con la categoría
#   **"Desconocido"** y se agrega un indicador binario `<columna>_faltante`, porque la
#   ausencia es informativa (la proporción de ingreso >50K es mucho menor cuando falta
#   `occupation`) y la moda crearía una concentración artificial en `Prof-specialty`.
#
# Ambas reglas son **determinísticas** (un valor fijo, no se aprenden de los datos), así que
# aplicarlas no genera fuga de información. Se guardan los conjuntos train/test **ya
# imputados**, con el mismo split para todo el equipo, para que la Actividad 5 parta de ahí.
# Los límites de la winsorización sí deben calcularse **solo con train** y aplicarse a test.

# %%
PROC_DIR = DATA_DIR / 'processed'
PROC_DIR.mkdir(exist_ok=True)


def aplicar_decision_imputacion(X, dataset):
    X = X.copy()
    if dataset == 'telco':
        X['TotalCharges'] = X['TotalCharges'].fillna(0)
    else:
        cols = ['workclass', 'occupation', 'native-country']
        X = agregar_indicadores(X, cols)          # primero el indicador, luego el relleno
        X[cols] = X[cols].fillna('Desconocido')
    return X


particiones = {'telco': (Xt_tr, yt_tr, Xt_te, yt_te),
               'adult': (Xa_tr, ya_tr, Xa_te, ya_te)}
for nombre, (Xtr, ytr, Xte, yte) in particiones.items():
    tr = aplicar_decision_imputacion(Xtr, nombre).assign(target=ytr)
    te = aplicar_decision_imputacion(Xte, nombre).assign(target=yte)
    assert tr.isna().sum().sum() == 0 and te.isna().sum().sum() == 0
    tr.to_csv(PROC_DIR / f'{nombre}_train_imputado.csv', index=False)
    te.to_csv(PROC_DIR / f'{nombre}_test_imputado.csv', index=False)
    print(f'{nombre}: train {tr.shape}, test {te.shape}, '
          f'proporción de clase positiva train/test: {tr.target.mean():.3f} / {te.target.mean():.3f}')

# %% [markdown]
# **Cómo cargarlos en la Actividad 5** (`target` = 1 es la clase positiva: Churn o ingreso >50K):
#
# ```python
# tr = pd.read_csv('../data/processed/telco_train_imputado.csv')
# te = pd.read_csv('../data/processed/telco_test_imputado.csv')
# X_train, y_train = tr.drop(columns='target'), tr['target']
# X_test,  y_test  = te.drop(columns='target'), te['target']
# ```

# %% [markdown]
# ## 6. Conclusiones de la Actividad 4
#
# (Valores obtenidos con `RANDOM_STATE = 42`.)
#
# - **Numérica (`TotalCharges`):** los 11 faltantes son estructurales (todos tienen
#   `tenure = 0` y ninguno abandonó). Con solo 0.16% de faltantes, la media, la mediana y KNN
#   casi no cambian la media, la varianza (menos de 0.2%) ni la asimetría (0.96).
# - **Concentraciones artificiales:** la media y la mediana asignan 2,283.30 y 1,397.48 a los
#   11 clientes (un pico de valores idénticos), cuando el valor coherente es 0. En la
#   simulación con 15% de faltantes, la razón varianza imputada / real fue 0.847 (media), 0.866
#   (mediana) y 1.000 (KNN); el RMSE fue 2,275, 2,437 y 78, respectivamente.
# - **Categórica (`occupation`):** la moda aumentó `Prof-specialty` de 12.71% a 18.38%
#   (+5.66 puntos porcentuales); "Desconocido" no altera las demás categorías.
# - **Ausencia informativa:** el ingreso >50K es 24.9% con `occupation` y 10.4% sin ella;
#   `workclass` y `occupation` faltan casi siempre juntas (1,836 de 1,843). El indicador de
#   ausencia está justificado en Adult; en `TotalCharges` es redundante con `tenure == 0`.
# - **Desempeño del modelo:** las diferencias de F1 son menores a 0.003 en Telco y del orden de
#   0.004 a 0.005 en Adult (validación cruzada), dentro de una desviación estándar. No hay
#   evidencia de que una estrategia sea mejor que otra.
# - **Decisión final:** `TotalCharges` se imputa con 0; `workclass`, `occupation` y
#   `native-country` con "Desconocido" más indicador de ausencia. Se justifica por la
#   coherencia con el mecanismo de ausencia y por conservar la distribución, no por las métricas.
#
# **Limitaciones:** pocos faltantes en Telco (efecto casi nulo en el modelo), la simulación MCAR
# no representa el mecanismo real de ausencia y se usó un solo modelo (regresión logística) y un
# solo split.
