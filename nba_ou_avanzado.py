"""
============================================================
 CALCULADORA OVER/UNDER — NBA (Modelo Avanzado v2)
 Temporada 2025-26
============================================================
 Modelo con Pace ajustado + eficiencias + local/visitante,
 con dispersión corregida basada en el historial de totales.

 Incluye:
   - Cálculo del rango MIN_OVER / MAX_UNDER con AVEDEV corregido
   - Backtest walk-forward comparativo
   - Exploración de multiplicadores de dispersión

 Uso:
     python nba_ou_avanzado.py

 Dependencias:
     pip install nba_api pandas numpy
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import scoreboardv3
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
EXPORTAR_CSV = True

# Parámetros del backtest
MULTIPLICADOR_DISPERSION = 1.3   # Factor que multiplica la dispersión
MIN_PARTIDOS_HISTORIAL = 3       # Mínimo de partidos antes de predecir
VENTANA_TOTALES = 10             # Nº de totales recientes por equipo para la dispersión
BIAS_CORRECCION = 0.0            # Offset al centro (ajustar tras el backtest)


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
# 3. MAPA DE EQUIPOS
# ============================================================
def construir_mapa_equipos():
    equipos = nba_teams.get_teams()
    mapa = {}
    for eq in equipos:
        nombre_oficial = eq['full_name']
        for clave in [eq['full_name'], eq['nickname'], eq['city'],
                      eq['abbreviation'], f"{eq['city']} {eq['nickname']}"]:
            mapa[clave.lower().strip()] = nombre_oficial
    return mapa


MAPA_EQUIPOS = construir_mapa_equipos()


def resolver_nombre(nombre_api):
    if not nombre_api:
        return None
    return MAPA_EQUIPOS.get(nombre_api.lower().strip(), nombre_api)


# ============================================================
# 4. ESTADÍSTICAS AVANZADAS POR UBICACIÓN
# ============================================================
def descargar_stats_avanzadas(season=SEASON, season_type=SEASON_TYPE):
    """
    Descarga ORtg, DRtg y PACE para Home y Road, y los une por TEAM_ID.
    """
    print(f"[1/5] Descargando estadísticas avanzadas NBA {season}...")

    stats_home = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Advanced',
        location_nullable='Home',
        per_mode_detailed='PerGame'
    )
    df_home = stats_home.get_data_frames()[0]

    stats_road = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense='Advanced',
        location_nullable='Road',
        per_mode_detailed='PerGame'
    )
    df_road = stats_road.get_data_frames()[0]

    cols_home = ['TEAM_ID', 'TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'PACE']
    cols_road = ['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'PACE']

    for col in cols_home:
        if col not in df_home.columns:
            raise ValueError(f"Falta '{col}' en Home. Disponibles: {df_home.columns.tolist()}")
    for col in cols_road:
        if col not in df_road.columns:
            raise ValueError(f"Falta '{col}' en Road. Disponibles: {df_road.columns.tolist()}")

    df_home = df_home[cols_home].rename(columns={
        'OFF_RATING': 'ORtg_Casa',
        'DEF_RATING': 'DRtg_Casa',
        'PACE': 'Pace_Casa'
    })

    df_road = df_road[cols_road].rename(columns={
        'OFF_RATING': 'ORtg_Fuera',
        'DEF_RATING': 'DRtg_Fuera',
        'PACE': 'Pace_Fuera'
    })

    df = pd.merge(df_home, df_road, on='TEAM_ID', how='inner')

    for col in ['ORtg_Casa', 'DRtg_Casa', 'Pace_Casa',
                'ORtg_Fuera', 'DRtg_Fuera', 'Pace_Fuera']:
        df[col] = df[col].round(2)

    print(f"       → {len(df)} equipos con stats avanzadas.")
    return df


# ============================================================
# 5. ESTADÍSTICAS TRADICIONALES (PPG/oPPG)
# ============================================================
def descargar_stats_tradicionales(season=SEASON, season_type=SEASON_TYPE):
    """Descarga PPG y oPPG brutos para el modelo original."""
    print(f"[2/5] Descargando stats tradicionales (PPG/oPPG)...")

    stats_base = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Base',
    )
    df_base = stats_base.get_data_frames()[0]

    stats_opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Opponent',
    )
    df_opp = stats_opp.get_data_frames()[0]

    df = pd.merge(
        df_base[['TEAM_ID', 'TEAM_NAME', 'GP', 'PTS']],
        df_opp[['TEAM_ID', 'OPP_PTS']],
        on='TEAM_ID', how='inner'
    )
    df = df.rename(columns={'PTS': 'PPG', 'OPP_PTS': 'oPPG'})
    df['PPG'] = df['PPG'].round(1)
    df['oPPG'] = df['oPPG'].round(1)

    print(f"       → {len(df)} equipos con stats tradicionales.")
    return df


# ============================================================
# 6. ÚLTIMOS RESULTADOS
# ============================================================
def descargar_ultimos_resultados(season=SEASON, season_type=SEASON_TYPE):
    """Último marcador de cada equipo."""
    print(f"[3/5] Descargando últimos resultados...")

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = finder.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

    partidos = {}
    for _, fila in df.iterrows():
        gid = fila['GAME_ID']
        eq = resolver_nombre(fila['TEAM_NAME'])
        if gid not in partidos:
            partidos[gid] = {}
        partidos[gid][eq] = int(fila['PTS'])

    ultimos = {}
    for _, fila in df.iterrows():
        eq = resolver_nombre(fila['TEAM_NAME'])
        if eq in ultimos:
            continue
        gid = fila['GAME_ID']
        favor = int(fila['PTS'])
        contra = None
        for otro_eq, pts in partidos.get(gid, {}).items():
            if otro_eq != eq:
                contra = pts
                break
        if contra is None:
            continue
        ultimos[eq] = {'favor': favor, 'contra': contra}

    print(f"       → {len(ultimos)} equipos con último resultado.")
    return ultimos


# ============================================================
# 7. CÁLCULO DEL TOTAL ESPERADO (MODELO AVANZADO)
# ============================================================
def calcular_total_esperado(local, visitante, df_avanzado):
    """
    Calcula el total esperado usando Pace ajustado y eficiencias.
    """
    s_l = df_avanzado[df_avanzado['TEAM_NAME'] == local]
    s_v = df_avanzado[df_avanzado['TEAM_NAME'] == visitante]

    if s_l.empty or s_v.empty:
        return None

    pace_l_casa = s_l.iloc[0]['Pace_Casa']
    ortg_l_casa = s_l.iloc[0]['ORtg_Casa']
    drtg_l_casa = s_l.iloc[0]['DRtg_Casa']

    pace_v_fuera = s_v.iloc[0]['Pace_Fuera']
    ortg_v_fuera = s_v.iloc[0]['ORtg_Fuera']
    drtg_v_fuera = s_v.iloc[0]['DRtg_Fuera']

    pace_esperado = (pace_l_casa + pace_v_fuera) / 2
    ortg_esperado_local = (ortg_l_casa + drtg_v_fuera) / 2
    ortg_esperado_visitante = (ortg_v_fuera + drtg_l_casa) / 2

    total_estimado = (ortg_esperado_local + ortg_esperado_visitante) * (pace_esperado / 100)

    return {
        'total_estimado': round(total_estimado, 2),
        'pace_esperado': round(pace_esperado, 1),
        'ortg_local': round(ortg_esperado_local, 1),
        'ortg_visitante': round(ortg_esperado_visitante, 1),
    }


# ============================================================
# 8. CÁLCULO DEL RANGO — MODELO AVANZADO (v2 con dispersión corregida)
# ============================================================
def calcular_rango_avanzado(local, visitante, df_avanzado, ultimos,
                            historial_totales=None,
                            dispersion_fn=avedev,
                            multiplicador=MULTIPLICADOR_DISPERSION):
    """
    Calcula MIN_OVER y MAX_UNDER usando el modelo pace-ajustado
    con dispersión basada en la variabilidad histórica de totales.
    """
    info = calcular_total_esperado(local, visitante, df_avanzado)
    if info is None:
        return None

    centro = info['total_estimado'] + BIAS_CORRECCION

    if local not in ultimos or visitante not in ultimos:
        return None

    ult_l = ultimos[local]
    ult_v = ultimos[visitante]

    # --- Dispersión corregida ---
    totales_recientes = []

    if historial_totales is not None:
        totales_recientes += historial_totales.get(local, [])[-VENTANA_TOTALES:]
        totales_recientes += historial_totales.get(visitante, [])[-VENTANA_TOTALES:]

    # Siempre incluir los últimos totales conocidos
    totales_recientes.append(ult_l['favor'] + ult_l['contra'])
    totales_recientes.append(ult_v['favor'] + ult_v['contra'])
    totales_recientes.append(centro)

    # Dispersión sobre todos los totales relevantes
    if len(totales_recientes) >= 2:
        desviacion = dispersion_fn(*totales_recientes) * multiplicador
    else:
        desviacion = centro * 0.08 * multiplicador

    min_over = centro - desviacion
    max_under = centro + desviacion

    return {
        'MIN_OVER': round(min_over, 2),
        'MAX_UNDER': round(max_under, 2),
        'CENTRO': round(centro, 2),
        'PACE_ESPERADO': info['pace_esperado'],
        'DESVIACION': round(desviacion, 2),
    }


# ============================================================
# 9. CÁLCULO DEL RANGO — MODELO ORIGINAL
# ============================================================
def calcular_rango_original(local, visitante, df_tradicional, ultimos,
                            dispersion_fn=avedev, multiplicador=1.0):
    """Modelo original con PPG/oPPG brutos."""
    s_l = df_tradicional[df_tradicional['TEAM_NAME'] == local]
    s_v = df_tradicional[df_tradicional['TEAM_NAME'] == visitante]

    if s_l.empty or s_v.empty or local not in ultimos or visitante not in ultimos:
        return None

    ppg_l = s_l.iloc[0]['PPG']
    oppg_l = s_l.iloc[0]['oPPG']
    ppg_v = s_v.iloc[0]['PPG']
    oppg_v = s_v.iloc[0]['oPPG']
    ult_l = ultimos[local]
    ult_v = ultimos[visitante]

    d_fav_l = dispersion_fn(ppg_l, ult_l['favor'], oppg_v) * multiplicador
    d_con_l = dispersion_fn(oppg_l, ult_l['contra'], ppg_v) * multiplicador
    tmin_l = (ppg_l - d_fav_l) + (oppg_l - d_con_l)
    tmax_l = (ppg_l + d_fav_l) + (oppg_l + d_con_l)

    d_fav_v = dispersion_fn(ppg_v, ult_v['favor'], oppg_l) * multiplicador
    d_con_v = dispersion_fn(oppg_v, ult_v['contra'], ppg_l) * multiplicador
    tmin_v = (ppg_v - d_fav_v) + (oppg_v - d_con_v)
    tmax_v = (ppg_v + d_fav_v) + (oppg_v + d_con_v)

    return {
        'MIN_OVER': round(min(tmin_l, tmin_v), 2),
        'MAX_UNDER': round(max(tmax_l, tmax_v), 2),
    }


# ============================================================
# 10. JORNADA ACTUAL
# ============================================================
def descargar_jornada_actual():
    """Partidos de ayer y hoy en formato NBA."""
    print(f"[4/5] Descargando jornada actual...")
    hoy = datetime.now()
    fixtures = []
    fechas_info = []

    for offset in [-1, 0]:
        fecha = (hoy + timedelta(days=offset)).strftime('%Y-%m-%d')
        try:
            board = scoreboardv3.ScoreboardV3(game_date=fecha, league_id='00')
            data = board.get_dict()
            games = data.get('scoreboard', {}).get('games', [])

            if not games:
                time.sleep(0.6)
                continue

            for g in games:
                fixtures.append({
                    'FECHA_NBA': fecha,
                    'LOCAL': resolver_nombre(g['homeTeam']['teamName']),
                    'VISITANTE': resolver_nombre(g['awayTeam']['teamName']),
                    'ESTADO': g.get('gameStatusText', '').strip(),
                    'PTS_LOCAL': g.get('homeTeam', {}).get('score'),
                    'PTS_VISITANTE': g.get('awayTeam', {}).get('score'),
                })

            fechas_info.append(f"{fecha} ({len(games)} partidos)")
            time.sleep(0.6)
        except Exception:
            time.sleep(0.6)
            continue

    print(f"       → {len(fixtures)} partidos en la jornada actual.")
    return fechas_info, fixtures


# ============================================================
# 11. GENERACIÓN DE TABLA
# ============================================================
def generar_tabla(fixtures, df_avanzado, df_tradicional, ultimos,
                  historial_totales=None):
    """Genera la tabla con ambos modelos para comparación."""
    filas = []
    for f in fixtures:
        local = f['LOCAL']
        visitante = f['VISITANTE']

        rango_av = calcular_rango_avanzado(
            local, visitante, df_avanzado, ultimos,
            historial_totales=historial_totales
        )
        rango_or = calcular_rango_original(local, visitante, df_tradicional, ultimos)

        fila = {
            'FECHA_NBA': f['FECHA_NBA'],
            'LOCAL': local,
            'VISITANTE': visitante,
            'ESTADO': f['ESTADO'],
        }

        if f['PTS_LOCAL'] is not None and f['PTS_VISITANTE'] is not None:
            try:
                pts_l, pts_v = int(f['PTS_LOCAL']), int(f['PTS_VISITANTE'])
                fila['MARCADOR'] = f"{pts_v}-{pts_l}"
                fila['TOTAL_REAL'] = pts_l + pts_v
            except (ValueError, TypeError):
                fila['MARCADOR'] = ''
                fila['TOTAL_REAL'] = None
        else:
            fila['MARCADOR'] = ''
            fila['TOTAL_REAL'] = None

        if rango_av:
            fila['AV_MIN_OVER'] = rango_av['MIN_OVER']
            fila['AV_MAX_UNDER'] = rango_av['MAX_UNDER']
            fila['AV_CENTRO'] = rango_av['CENTRO']
            fila['AV_DESVIACION'] = rango_av['DESVIACION']
        else:
            fila['AV_MIN_OVER'] = 'N/D'
            fila['AV_MAX_UNDER'] = 'N/D'
            fila['AV_CENTRO'] = 'N/D'
            fila['AV_DESVIACION'] = 'N/D'

        if rango_or:
            fila['OR_MIN_OVER'] = rango_or['MIN_OVER']
            fila['OR_MAX_UNDER'] = rango_or['MAX_UNDER']
        else:
            fila['OR_MIN_OVER'] = 'N/D'
            fila['OR_MAX_UNDER'] = 'N/D'

        filas.append(fila)

    return pd.DataFrame(filas)


# ============================================================
# 12. IMPRESIÓN
# ============================================================
def imprimir_tabla(df_tabla, fechas_info):
    print("\n" + "=" * 140)
    print(f"  TABLA COMPARATIVA — NBA | Jornada actual")
    print(f"  Fechas NBA: {', '.join(fechas_info)}")
    print("=" * 140)

    if df_tabla.empty:
        print("  No hay partidos.")
        return

    columnas = ['FECHA_NBA', 'LOCAL', 'VISITANTE', 'ESTADO', 'MARCADOR',
                'AV_MIN_OVER', 'AV_MAX_UNDER', 'AV_CENTRO',
                'OR_MIN_OVER', 'OR_MAX_UNDER']
    print(df_tabla[columnas].to_string(index=False))
    print("=" * 140)
    print("\n  AV = Modelo Avanzado (Pace + Eficiencias + dispersión corregida)")
    print("  OR = Modelo Original (PPG/oPPG brutos)")


# ============================================================
# 13. CARGA DE PARTIDOS HISTÓRICOS
# ============================================================
def cargar_partidos_historicos(season=SEASON, season_type=SEASON_TYPE):
    """Carga todos los partidos en formato una fila por partido."""
    print(f"[5/5] Cargando partidos históricos para backtest...")
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
        if ' vs. ' in r1['MATCHUP']:
            local, visitante = r1, r2
        else:
            local, visitante = r2, r1

        partidos.append({
            'GAME_ID': gid,
            'FECHA': local['GAME_DATE'],
            'LOCAL': resolver_nombre(local['TEAM_NAME']),
            'VISITANTE': resolver_nombre(visitante['TEAM_NAME']),
            'PTS_LOCAL': int(local['PTS']),
            'PTS_VISITANTE': int(visitante['PTS']),
            'TOTAL': int(local['PTS']) + int(visitante['PTS']),
        })

    df_partidos = pd.DataFrame(partidos).sort_values('FECHA').reset_index(drop=True)
    print(f"       → {len(df_partidos)} partidos históricos cargados.")
    return df_partidos


# ============================================================
# 14. BACKTEST WALK-FORWARD
# ============================================================
def backtest_walk_forward(df_partidos, df_avanzado, df_tradicional,
                          dispersion_fn=avedev, multiplicador=1.0):
    """
    Backtest walk-forward comparando el modelo avanzado (v2) con el original.
    """
    historial = {}          # equipo -> lista de {favor, contra}
    historial_totales = {}  # equipo -> lista de totales
    resultados = []

    for _, partido in df_partidos.iterrows():
        local = partido['LOCAL']
        visitante = partido['VISITANTE']
        total_real = partido['TOTAL']

        hist_l = historial.get(local, [])
        hist_v = historial.get(visitante, [])

        if len(hist_l) >= MIN_PARTIDOS_HISTORIAL and len(hist_v) >= MIN_PARTIDOS_HISTORIAL:

            # --- Modelo AVANZADO (v2) ---
            rango_av = calcular_rango_avanzado(
                local, visitante, df_avanzado,
                {local: hist_l[-1], visitante: hist_v[-1]},
                historial_totales=historial_totales,
                dispersion_fn=dispersion_fn,
                multiplicador=multiplicador
            )

            # --- Modelo ORIGINAL ---
            rango_or = calcular_rango_original(
                local, visitante, df_tradicional,
                {local: hist_l[-1], visitante: hist_v[-1]},
                dispersion_fn=dispersion_fn,
                multiplicador=multiplicador
            )

            if rango_av and rango_or:
                resultados.append({
                    'FECHA': partido['FECHA'],
                    'LOCAL': local,
                    'VISITANTE': visitante,
                    'TOTAL_REAL': total_real,
                    # Avanzado
                    'AV_MIN': rango_av['MIN_OVER'],
                    'AV_MAX': rango_av['MAX_UNDER'],
                    'AV_CENTRO': rango_av['CENTRO'],
                    'AV_DESV': rango_av['DESVIACION'],
                    'AV_ANCHO': round(rango_av['MAX_UNDER'] - rango_av['MIN_OVER'], 2),
                    'AV_ERROR': round(total_real - rango_av['CENTRO'], 2),
                    'AV_ABS_ERROR': round(abs(total_real - rango_av['CENTRO']), 2),
                    'AV_DENTRO': rango_av['MIN_OVER'] <= total_real <= rango_av['MAX_UNDER'],
                    # Original
                    'OR_MIN': rango_or['MIN_OVER'],
                    'OR_MAX': rango_or['MAX_UNDER'],
                    'OR_CENTRO': round((rango_or['MIN_OVER'] + rango_or['MAX_UNDER']) / 2, 2),
                    'OR_ANCHO': round(rango_or['MAX_UNDER'] - rango_or['MIN_OVER'], 2),
                    'OR_ERROR': round(total_real - (rango_or['MIN_OVER'] + rango_or['MAX_UNDER']) / 2, 2),
                    'OR_ABS_ERROR': round(abs(total_real - (rango_or['MIN_OVER'] + rango_or['MAX_UNDER']) / 2), 2),
                    'OR_DENTRO': rango_or['MIN_OVER'] <= total_real <= rango_or['MAX_UNDER'],
                })

        # Actualizar historial
        historial.setdefault(local, []).append({
            'favor': partido['PTS_LOCAL'],
            'contra': partido['PTS_VISITANTE'],
        })
        historial.setdefault(visitante, []).append({
            'favor': partido['PTS_VISITANTE'],
            'contra': partido['PTS_LOCAL'],
        })
        historial_totales.setdefault(local, []).append(total_real)
        historial_totales.setdefault(visitante, []).append(total_real)

    return pd.DataFrame(resultados)


# ============================================================
# 15. MÉTRICAS
# ============================================================
def calcular_metricas(df_bt, etiqueta=""):
    if df_bt.empty:
        return {}
    return {
        'ETIQUETA': etiqueta,
        'N_PARTIDOS': len(df_bt),
        'AV_HIT_RATE': round(df_bt['AV_DENTRO'].mean() * 100, 2),
        'AV_MAE': round(df_bt['AV_ABS_ERROR'].mean(), 2),
        'AV_BIAS': round(df_bt['AV_ERROR'].mean(), 2),
        'AV_RMSE': round(np.sqrt((df_bt['AV_ERROR'] ** 2).mean()), 2),
        'AV_ANCHO': round(df_bt['AV_ANCHO'].mean(), 2),
        'OR_HIT_RATE': round(df_bt['OR_DENTRO'].mean() * 100, 2),
        'OR_MAE': round(df_bt['OR_ABS_ERROR'].mean(), 2),
        'OR_BIAS': round(df_bt['OR_ERROR'].mean(), 2),
        'OR_ANCHO': round(df_bt['OR_ANCHO'].mean(), 2),
    }


def imprimir_metricas(m):
    print(f"\n  --- MODELO AVANZADO (Pace + Eficiencias + dispersión corregida) ---")
    print(f"    Hit rate:     {m['AV_HIT_RATE']}%")
    print(f"    MAE:          {m['AV_MAE']} pts")
    print(f"    Bias:         {m['AV_BIAS']:+.2f} pts")
    print(f"    RMSE:         {m['AV_RMSE']} pts")
    print(f"    Ancho medio:  {m['AV_ANCHO']} pts")

    print(f"\n  --- MODELO ORIGINAL (PPG/oPPG) ---")
    print(f"    Hit rate:     {m['OR_HIT_RATE']}%")
    print(f"    MAE:          {m['OR_MAE']} pts")
    print(f"    Bias:         {m['OR_BIAS']:+.2f} pts")
    print(f"    Ancho medio:  {m['OR_ANCHO']} pts")

    print(f"\n  --- MEJORA ---")
    print(f"    Δ MAE:  {m['OR_MAE'] - m['AV_MAE']:+.2f} pts (positivo = mejor)")
    print(f"    Δ Hit:  {m['AV_HIT_RATE'] - m['OR_HIT_RATE']:+.2f}%")


# ============================================================
# 16. EXPLORACIÓN DE MULTIPLICADORES
# ============================================================
def explorar_multiplicadores(df_partidos, df_avanzado, df_tradicional,
                             multiplicadores=(0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8)):
    print("\n[Explorando multiplicadores con dispersión corregida...]")
    resumen = []
    for k in multiplicadores:
        df_bt = backtest_walk_forward(
            df_partidos, df_avanzado, df_tradicional,
            dispersion_fn=avedev, multiplicador=k
        )
        m = calcular_metricas(df_bt, etiqueta=f"k={k}")
        resumen.append(m)
    return pd.DataFrame(resumen)


# ============================================================
# 17. MAIN
# ============================================================
def main():
    print("=" * 140)
    print("  CALCULADORA OVER/UNDER — NBA | MODELO AVANZADO v2")
    print(f"  Temporada {SEASON}")
    print("=" * 140)

    try:
        # 1. Descargar stats
        df_avanzado = descargar_stats_avanzadas()
        df_tradicional = descargar_stats_tradicionales()
        ultimos = descargar_ultimos_resultados()

        # 2. Jornada actual
        fechas_info, fixtures = descargar_jornada_actual()

        if fixtures:
            print(f"[5/5] Generando tabla comparativa...")
            df_tabla = generar_tabla(fixtures, df_avanzado, df_tradicional, ultimos)
            imprimir_tabla(df_tabla, fechas_info)

            if EXPORTAR_CSV:
                nombre = f"nba_over_under_{datetime.now().strftime('%Y%m%d')}.csv"
                df_tabla.to_csv(nombre, index=False)
                print(f"\n✅ Tabla guardada en {nombre}")
        else:
            print("\n⚠️  No hay partidos hoy. Saltando tabla.")

        # 3. Backtest
        print("\n" + "=" * 140)
        print("  BACKTEST WALK-FORWARD")
        print("=" * 140)

        df_historicos = cargar_partidos_historicos()

        if len(df_historicos) > 100:
            print(f"  Partidos históricos: {len(df_historicos)}")
            print(f"  Multiplicador de dispersión: {MULTIPLICADOR_DISPERSION}")
            print("  Ejecutando backtest... (puede tardar unos segundos)")

            df_bt = backtest_walk_forward(
                df_historicos, df_avanzado, df_tradicional,
                dispersion_fn=avedev, multiplicador=MULTIPLICADOR_DISPERSION
            )

            if not df_bt.empty:
                metricas = calcular_metricas(df_bt)
                imprimir_metricas(metricas)

                df_bt.to_csv('backtest_comparativo.csv', index=False)
                print(f"\n✅ Backtest guardado en backtest_comparativo.csv")

                # 4. Exploración de multiplicadores
                df_mult = explorar_multiplicadores(
                    df_historicos, df_avanzado, df_tradicional
                )
                print("\n--- Métricas por multiplicador ---")
                print(df_mult[['ETIQUETA', 'AV_HIT_RATE', 'AV_MAE', 'AV_ANCHO',
                               'OR_HIT_RATE', 'OR_MAE', 'OR_ANCHO']].to_string(index=False))

                df_mult.to_csv('backtest_multiplicadores.csv', index=False)

                # Recomendación
                print("\n  --- RECOMENDACIÓN ---")
                mejor_mae = df_mult.loc[df_mult['AV_MAE'].idxmin()]
                print(f"  Mejor MAE:   {mejor_mae['ETIQUETA']}  "
                      f"(MAE={mejor_mae['AV_MAE']} | Hit={mejor_mae['AV_HIT_RATE']}%)")
                mejor_hit = df_mult.loc[(df_mult['AV_HIT_RATE'] - 70).abs().idxmin()]
                print(f"  Hit ~70%:    {mejor_hit['ETIQUETA']}  "
                      f"(MAE={mejor_hit['AV_MAE']} | Hit={mejor_hit['AV_HIT_RATE']}%)")

            else:
                print("  ⚠️  No hay suficientes partidos para el backtest.")
        else:
            print(f"  ⚠️  Solo {len(df_historicos)} partidos. Se necesitan más de 100.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()