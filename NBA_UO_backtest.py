"""
============================================================
 BACKTEST — Calculadora Over/Under NBA
 Temporada 2025-26
============================================================
 Simula el algoritmo AVEDEV en cada partido de la temporada
 usando solo información anterior (walk-forward), mide la
 precisión y explora variantes para mejorar.

 Uso:
     python backtest_nba.py

 Dependencias:
     pip install nba_api pandas numpy
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
MIN_PARTIDOS_HISTORIAL = 3   # Mínimo de partidos antes de predecir


# ============================================================
# 2. FUNCIONES DE DISPERSIÓN
# ============================================================
def avedev(*valores):
    """Desviación media absoluta (AVEDEV de Excel)."""
    n = len(valores)
    if n == 0:
        return 0.0
    media = sum(valores) / n
    return sum(abs(v - media) for v in valores) / n


def std(*valores):
    """Desviación estándar muestral."""
    if len(valores) < 2:
        return 0.0
    return float(np.std(valores, ddof=1))


def mad(*valores):
    """Desviación absoluta mediana (robusta a outliers)."""
    if len(valores) == 0:
        return 0.0
    med = np.median(valores)
    return float(np.mean([abs(v - med) for v in valores]))


# ============================================================
# 3. CARGA DE PARTIDOS
# ============================================================
def cargar_partidos(season=SEASON, season_type=SEASON_TYPE):
    """
    Descarga todos los partidos y los convierte a formato
    'una fila por partido' (local, visitante, puntos).
    """
    print(f"[1/4] Descargando partidos de la temporada {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = finder.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    partidos = []
    for gid, grupo in df.groupby('GAME_ID'):
        if len(grupo) != 2:
            continue
        r1, r2 = grupo.iloc[0], grupo.iloc[1]
        # Determinar local: el que tiene ' vs. ' en MATCHUP
        if ' vs. ' in r1['MATCHUP']:
            local, visitante = r1, r2
        else:
            local, visitante = r2, r1

        partidos.append({
            'GAME_ID': gid,
            'FECHA': local['GAME_DATE'],
            'LOCAL': local['TEAM_NAME'],
            'VISITANTE': visitante['TEAM_NAME'],
            'PTS_LOCAL': int(local['PTS']),
            'PTS_VISITANTE': int(visitante['PTS']),
            'TOTAL': int(local['PTS']) + int(visitante['PTS']),
        })

    df_partidos = (pd.DataFrame(partidos)
                     .sort_values('FECHA')
                     .reset_index(drop=True))

    print(f"       → {len(df_partidos)} partidos cargados.")
    print(f"       → Rango de fechas: {df_partidos['FECHA'].min().date()} "
          f"a {df_partidos['FECHA'].max().date()}")
    return df_partidos


# ============================================================
# 4. WALK-FORWARD BACKTEST
# ============================================================
def backtest_walk_forward(df_partidos,
                          dispersion_fn=avedev,
                          multiplicador=1.0,
                          ventana=None,
                          ponderacion_reciente=False):
    """
    Simula el algoritmo partido a partido.

    Parámetros:
        dispersion_fn: función de dispersión (avedev, std, mad)
        multiplicador: factor que multiplica la dispersión
                       (1.0 = comportamiento original)
        ventana: nº de partidos recientes a usar (None = todos)
        ponderacion_reciente: si True, pondera más los últimos partidos
    """
    historial = {}   # equipo -> lista de dicts {pts_favor, pts_contra}
    resultados = []

    for _, partido in df_partidos.iterrows():
        local = partido['LOCAL']
        visitante = partido['VISITANTE']
        fecha = partido['FECHA']
        total_real = partido['TOTAL']

        hist_l = historial.get(local, [])
        hist_v = historial.get(visitante, [])

        # Solo predecir si ambos tienen historial suficiente
        if len(hist_l) >= MIN_PARTIDOS_HISTORIAL and len(hist_v) >= MIN_PARTIDOS_HISTORIAL:

            # Aplicar ventana si procede
            h_l = hist_l[-ventana:] if ventana else hist_l
            h_v = hist_v[-ventana:] if ventana else hist_v

            # Calcular pesos
            if ponderacion_reciente:
                n_l = len(h_l); n_v = len(h_v)
                pesos_l = np.array([np.exp(i / n_l) for i in range(n_l)])
                pesos_v = np.array([np.exp(i / n_v) for i in range(n_v)])
                pesos_l = pesos_l / pesos_l.sum()
                pesos_v = pesos_v / pesos_v.sum()
            else:
                pesos_l = None
                pesos_v = None

            ppg_l = float(np.average([x['favor'] for x in h_l], weights=pesos_l))
            oppg_l = float(np.average([x['contra'] for x in h_l], weights=pesos_l))
            ppg_v = float(np.average([x['favor'] for x in h_v], weights=pesos_v))
            oppg_v = float(np.average([x['contra'] for x in h_v], weights=pesos_v))

            ult_l = h_l[-1]
            ult_v = h_v[-1]

            # Aplicar multiplicador a la dispersión
            d_fav_l = multiplicador * dispersion_fn(ppg_l, ult_l['favor'], oppg_v)
            d_con_l = multiplicador * dispersion_fn(oppg_l, ult_l['contra'], ppg_v)
            tmin_l = (ppg_l - d_fav_l) + (oppg_l - d_con_l)
            tmax_l = (ppg_l + d_fav_l) + (oppg_l + d_con_l)

            d_fav_v = multiplicador * dispersion_fn(ppg_v, ult_v['favor'], oppg_l)
            d_con_v = multiplicador * dispersion_fn(oppg_v, ult_v['contra'], ppg_l)
            tmin_v = (ppg_v - d_fav_v) + (oppg_v - d_con_v)
            tmax_v = (ppg_v + d_fav_v) + (oppg_v + d_con_v)

            min_over = min(tmin_l, tmin_v)
            max_under = max(tmax_l, tmax_v)
            centro = (min_over + max_under) / 2
            ancho = max_under - min_over

            resultados.append({
                'FECHA': fecha,
                'LOCAL': local,
                'VISITANTE': visitante,
                'TOTAL_REAL': total_real,
                'MIN_OVER': round(min_over, 2),
                'MAX_UNDER': round(max_under, 2),
                'CENTRO': round(centro, 2),
                'ANCHO': round(ancho, 2),
                'ERROR': round(total_real - centro, 2),
                'ABS_ERROR': round(abs(total_real - centro), 2),
                'DENTRO': min_over <= total_real <= max_under,
            })

        # Actualizar historial SIEMPRE (incluso si no se predijo)
        historial.setdefault(local, []).append({
            'favor': partido['PTS_LOCAL'],
            'contra': partido['PTS_VISITANTE'],
        })
        historial.setdefault(visitante, []).append({
            'favor': partido['PTS_VISITANTE'],
            'contra': partido['PTS_LOCAL'],
        })

    return pd.DataFrame(resultados)


# ============================================================
# 5. MÉTRICAS
# ============================================================
def calcular_metricas(df_bt, etiqueta=""):
    """Calcula las métricas principales del backtest."""
    if df_bt.empty:
        return {}

    hit_rate = df_bt['DENTRO'].mean()
    mae = df_bt['ABS_ERROR'].mean()
    bias = df_bt['ERROR'].mean()
    rmse = np.sqrt((df_bt['ERROR'] ** 2).mean())
    ancho_medio = df_bt['ANCHO'].mean()
    total_medio = df_bt['TOTAL_REAL'].mean()

    return {
        'ETIQUETA': etiqueta,
        'N_PARTIDOS': len(df_bt),
        'HIT_RATE': round(hit_rate * 100, 2),
        'MAE': round(mae, 2),
        'BIAS': round(bias, 2),
        'RMSE': round(rmse, 2),
        'ANCHO_MEDIO': round(ancho_medio, 2),
        'TOTAL_MEDIO_REAL': round(total_medio, 2),
    }


# ============================================================
# 6. EXPLORACIÓN DE VARIANTES
# ============================================================
def explorar_multiplicadores(df_partidos, dispersion_fn=avedev,
                             multiplicadores=(0.6, 0.8, 1.0, 1.2, 1.5, 2.0)):
    """Prueba distintos multiplicadores de dispersión."""
    print("\n[Explorando multiplicadores...]")
    resumen = []
    for k in multiplicadores:
        df_bt = backtest_walk_forward(
            df_partidos, dispersion_fn=dispersion_fn, multiplicador=k
        )
        m = calcular_metricas(df_bt, etiqueta=f"k={k}")
        resumen.append(m)
    return pd.DataFrame(resumen)


def explorar_ventanas(df_partidos, dispersion_fn=avedev, multiplicador=1.0,
                      ventanas=(5, 10, 15, 20, None)):
    """Prueba distintas ventanas de partidos recientes."""
    print("\n[Explorando ventanas...]")
    resumen = []
    for v in ventanas:
        df_bt = backtest_walk_forward(
            df_partidos, dispersion_fn=dispersion_fn,
            multiplicador=multiplicador, ventana=v
        )
        m = calcular_metricas(df_bt, etiqueta=f"ventana={v}")
        resumen.append(m)
    return pd.DataFrame(resumen)


def explorar_dispersion(df_partidos, multiplicador=1.0):
    """Compara AVEDEV, STD y MAD."""
    print("\n[Explorando funciones de dispersión...]")
    resumen = []
    for nombre, fn in [('AVEDEV', avedev), ('STD', std), ('MAD', mad)]:
        df_bt = backtest_walk_forward(
            df_partidos, dispersion_fn=fn, multiplicador=multiplicador
        )
        m = calcular_metricas(df_bt, etiqueta=nombre)
        resumen.append(m)
    return pd.DataFrame(resumen)


# ============================================================
# 7. ANÁLISIS DEL ERROR
# ============================================================
def analizar_error(df_bt):
    """Analiza la distribución del error y rangos de cobertura."""
    print("\n" + "=" * 70)
    print("  ANÁLISIS DEL ERROR")
    print("=" * 70)

    # Error en función del total real
    print("\nError medio por rango de total real:")
    bins = pd.cut(df_bt['TOTAL_REAL'],
                  bins=[150, 190, 200, 210, 220, 230, 240, 300],
                  labels=['<190', '190-200', '200-210', '210-220',
                          '220-230', '230-240', '>240'])
    for rango, grupo in df_bt.groupby(bins, observed=True):
        if len(grupo) == 0:
            continue
        err = grupo['ERROR'].mean()
        print(f"  {rango:10s}  n={len(grupo):4d}  bias={err:+.2f} pts")

    # Cobertura por ancho
    print("\nCobertura según ancho del rango:")
    cuartiles = pd.qcut(df_bt['ANCHO'], q=4, labels=['Q1 (estrecho)',
                                                      'Q2', 'Q3', 'Q4 (amplio)'])
    for q, grupo in df_bt.groupby(cuartiles, observed=True):
        print(f"  {q:15s}  n={len(grupo):4d}  "
              f"ancho_medio={grupo['ANCHO'].mean():.1f}  "
              f"hit_rate={grupo['DENTRO'].mean()*100:.1f}%")

    # Percentiles del error absoluto
    print("\nPercentiles del error absoluto:")
    for p in [50, 75, 90, 95]:
        v = np.percentile(df_bt['ABS_ERROR'], p)
        print(f"  P{p}: {v:.2f} pts")


# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"  BACKTEST — NBA {SEASON}")
    print("=" * 70)

    # 1. Cargar partidos
    df_partidos = cargar_partidos()

    # 2. Backtest con parámetros originales
    print(f"\n[2/4] Ejecutando backtest con parámetros originales...")
    df_bt = backtest_walk_forward(df_partidos)
    metricas = calcular_metricas(df_bt, etiqueta="Original")

    print(f"       → {metricas['N_PARTIDOS']} partidos evaluados")
    print(f"       → Hit rate:     {metricas['HIT_RATE']}%")
    print(f"       → MAE:          {metricas['MAE']} pts")
    print(f"       → Bias:         {metricas['BIAS']:+.2f} pts")
    print(f"       → RMSE:         {metricas['RMSE']} pts")
    print(f"       → Ancho medio:  {metricas['ANCHO_MEDIO']} pts")

    # 3. Explorar mejoras
    print(f"\n[3/4] Explorando variantes...")
    df_mult = explorar_multiplicadores(df_partidos)
    df_vent = explorar_ventanas(df_partidos)
    df_disp = explorar_dispersion(df_partidos)

    print("\n--- Multiplicadores de dispersión ---")
    print(df_mult.to_string(index=False))

    print("\n--- Ventanas de partidos ---")
    print(df_vent.to_string(index=False))

    print("\n--- Funciones de dispersión ---")
    print(df_disp.to_string(index=False))

    # 4. Análisis del error
    print(f"\n[4/4] Analizando distribución del error...")
    analizar_error(df_bt)

    # 5. Guardar resultados
    df_bt.to_csv('backtest_nba_partidos.csv', index=False)
    df_mult.to_csv('backtest_nba_multiplicadores.csv', index=False)
    df_vent.to_csv('backtest_nba_ventanas.csv', index=False)
    df_disp.to_csv('backtest_nba_dispersion.csv', index=False)
    print("\n✅ Resultados guardados en CSV")

    # 6. Recomendación
    print("\n" + "=" * 70)
    print("  RECOMENDACIÓN")
    print("=" * 70)
    mejor = df_mult.loc[df_mult['MAE'].idxmin()]
    print(f"  Mejor multiplicador (menor MAE): {mejor['ETIQUETA']}")
    print(f"    → MAE: {mejor['MAE']} pts | Hit rate: {mejor['HIT_RATE']}%")

    mejor_hit = df_mult.loc[(df_mult['HIT_RATE'] - 80).abs().idxmin()]
    print(f"  Multiplicador más cercano a 80% de cobertura: {mejor_hit['ETIQUETA']}")
    print(f"    → MAE: {mejor_hit['MAE']} pts | Hit rate: {mejor_hit['HIT_RATE']}%")


if __name__ == "__main__":
    main()