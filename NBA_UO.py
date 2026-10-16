"""
============================================================
 CALCULADORA OVER/UNDER — NBA
 Temporada 2025-26 (o la que indiques en SEASON)
============================================================
 Detecta automáticamente los partidos del próximo día con
 partidos programados y calcula MIN_OVER / MAX_UNDER usando
 el algoritmo AVEDEV.

 Uso:
     python NBA_UO.py

 Dependencias:
     pip install nba_api pandas
============================================================
"""

import pandas as pd
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


# ============================================================
# 2. AVEDEV
# ============================================================
def avedev(*valores):
    n = len(valores)
    if n == 0:
        return 0.0
    media = sum(valores) / n
    return sum(abs(v - media) for v in valores) / n


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
# 4. ESTADÍSTICAS DE EQUIPOS (PPG y oPPG)
# ============================================================
def descargar_estadisticas_equipos(season=SEASON, season_type=SEASON_TYPE):
    """
    Dos llamadas: Base (PTS) + Opponent (OPP_PTS). Se cruzan por TEAM_ID.
    """
    print(f"[1/4] Descargando estadísticas de equipos NBA {season}...")

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

    for col in ['TEAM_ID', 'TEAM_NAME', 'GP', 'PTS']:
        if col not in df_base.columns:
            raise ValueError(f"Falta '{col}' en Base. Disponibles: {df_base.columns.tolist()}")
    if 'OPP_PTS' not in df_opp.columns:
        raise ValueError(f"Falta 'OPP_PTS' en Opponent. Disponibles: {df_opp.columns.tolist()}")

    df = pd.merge(
        df_base[['TEAM_ID', 'TEAM_NAME', 'GP', 'PTS']],
        df_opp[['TEAM_ID', 'OPP_PTS']],
        on='TEAM_ID', how='inner'
    )

    df = df.rename(columns={
        'TEAM_NAME': 'TEAM',
        'PTS': 'PPG',
        'OPP_PTS': 'oPPG',
    })
    df['PPG'] = df['PPG'].round(1)
    df['oPPG'] = df['oPPG'].round(1)

    df = df[['TEAM', 'PPG', 'oPPG', 'GP']].sort_values('TEAM').reset_index(drop=True)
    print(f"       → {len(df)} equipos cargados.")
    return df


# ============================================================
# 5. ÚLTIMOS RESULTADOS (con LeagueGameFinder)
# ============================================================
def descargar_ultimos_resultados(season=SEASON, season_type=SEASON_TYPE):
    """
    Usa LeagueGameFinder para obtener todos los partidos y extrae
    el último marcador de cada equipo.
    """
    print(f"[2/4] Descargando últimos resultados...")

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = finder.get_data_frames()[0]

    if df.empty:
        raise ValueError(f"LeagueGameFinder no devolvió partidos para {season}.")

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

    # Mapa GAME_ID → {equipo: pts}
    partidos = {}
    for _, fila in df.iterrows():
        gid = fila['GAME_ID']
        eq = resolver_nombre(fila['TEAM_NAME'])
        if gid not in partidos:
            partidos[gid] = {}
        partidos[gid][eq] = int(fila['PTS'])

    # Último partido de cada equipo
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
# 6. PRÓXIMA JORNADA
# ============================================================
def descargar_proxima_jornada(dias_a_buscar=3):
    """
    Busca partidos en los próximos 'dias_a_buscar' días (por defecto 3).
    Combina todos los partidos programados en una sola lista.
    """
    print(f"[3/4] Detectando próxima jornada (rango de {dias_a_buscar} días)...")
    hoy = datetime.now()
    todos_fixtures = []
    fechas_encontradas = []

    for offset in range(0, dias_a_buscar):
        fecha = (hoy + timedelta(days=offset)).strftime('%Y-%m-%d')
        try:
            board = scoreboardv3.ScoreboardV3(game_date=fecha, league_id='00')
            data = board.get_dict()
            games = data.get('scoreboard', {}).get('games', [])

            if not games:
                time.sleep(0.6)
                continue

            for g in games:
                local = resolver_nombre(g['homeTeam']['teamName'])
                visitante = resolver_nombre(g['awayTeam']['teamName'])
                estado = g.get('gameStatusText', '')

                todos_fixtures.append({
                    'FECHA_NBA': fecha,
                    'LOCAL': local,
                    'VISITANTE': visitante,
                    'ESTADO': estado,
                })

            fechas_encontradas.append(f"{fecha} ({len(games)} partidos)")
            time.sleep(0.6)

        except Exception as e:
            print(f"       → Error en {fecha}: {e}")
            time.sleep(0.6)
            continue

    if not todos_fixtures:
        print("       → No se encontraron partidos en los próximos días.")
        return None, []

    print(f"       → Fechas consultadas: {', '.join(fechas_encontradas)}")
    print(f"       → Total partidos encontrados: {len(todos_fixtures)}")

    return fechas_encontradas, todos_fixtures

# ============================================================
# 7. CÁLCULO DEL RANGO
# ============================================================
def calcular_rango(local, visitante, df_equipos, ultimos):
    s_l = df_equipos[df_equipos['TEAM'] == local]
    s_v = df_equipos[df_equipos['TEAM'] == visitante]

    if s_l.empty or s_v.empty or local not in ultimos or visitante not in ultimos:
        return None

    ppg_l = s_l.iloc[0]['PPG']; oppg_l = s_l.iloc[0]['oPPG']
    ppg_v = s_v.iloc[0]['PPG']; oppg_v = s_v.iloc[0]['oPPG']
    ult_l = ultimos[local]; ult_v = ultimos[visitante]

    d_fav_l = avedev(ppg_l, ult_l['favor'], oppg_v)
    d_con_l = avedev(oppg_l, ult_l['contra'], ppg_v)
    tmin_l = (ppg_l - d_fav_l) + (oppg_l - d_con_l)
    tmax_l = (ppg_l + d_fav_l) + (oppg_l + d_con_l)

    d_fav_v = avedev(ppg_v, ult_v['favor'], oppg_l)
    d_con_v = avedev(oppg_v, ult_v['contra'], ppg_l)
    tmin_v = (ppg_v - d_fav_v) + (oppg_v - d_con_v)
    tmax_v = (ppg_v + d_fav_v) + (oppg_v + d_con_v)

    return {
        'MIN_OVER': round(min(tmin_l, tmin_v), 2),
        'MAX_UNDER': round(max(tmax_l, tmax_v), 2),
    }


# ============================================================
# 8. GENERAR TABLA
# ============================================================
def generar_tabla(fixtures, fecha, df_equipos, ultimos):
    filas = []
    for local, visitante in fixtures:
        rango = calcular_rango(local, visitante, df_equipos, ultimos)
        if rango is None:
            filas.append({
                'FECHA': fecha, 'LOCAL': local, 'VISITANTE': visitante,
                'MIN_OVER': 'N/D', 'MAX_UNDER': 'N/D', 'RANGO': 'N/D',
            })
        else:
            filas.append({
                'FECHA': fecha, 'LOCAL': local, 'VISITANTE': visitante,
                'MIN_OVER': rango['MIN_OVER'],
                'MAX_UNDER': rango['MAX_UNDER'],
                'RANGO': round(rango['MAX_UNDER'] - rango['MIN_OVER'], 2),
            })
    return pd.DataFrame(filas)


# ============================================================
# 9. IMPRESIÓN
# ============================================================
def imprimir_tabla(df_tabla, fecha):
    print("\n" + "=" * 110)
    print(f"  TABLA OVER/UNDER — NBA | {fecha}")
    print("=" * 110)
    if df_tabla.empty:
        print("  No hay partidos para mostrar.")
        print("=" * 110)
        return
    print(df_tabla[['LOCAL', 'VISITANTE', 'MIN_OVER', 'MAX_UNDER', 'RANGO']].to_string(index=False))
    print("=" * 110)
    print("\n  Interpretación:")
    print("    🟢 OVER con valor  → línea de apuestas < MIN_OVER")
    print("    🔴 UNDER con valor → línea de apuestas > MAX_UNDER")
    print("    RANGO = MAX_UNDER − MIN_OVER")


# ============================================================
# 10. MAIN
# ============================================================
def main():
    print("=" * 110)
    print("  CALCULADORA OVER/UNDER — NBA")
    print(f"  Temporada {SEASON}")
    print("=" * 110)

    try:
        df_equipos = descargar_estadisticas_equipos()
        ultimos = descargar_ultimos_resultados()

        # Buscar en un rango de 3 días
        fechas, fixtures_info = descargar_proxima_jornada(dias_a_buscar=3)

        if not fixtures_info:
            print("\n❌ No se encontraron partidos para la próxima jornada.")
            return

        # Filtrar solo partidos no jugados (opcional)
        # Si quieres ver todos, comenta este filtro
        fixtures_info = [f for f in fixtures_info if 'Final' not in f['ESTADO']]

        # Convertir a lista de tuplas (local, visitante)
        fixtures = [(f['LOCAL'], f['VISITANTE']) for f in fixtures_info]

        print(f"[4/4] Generando tabla over/under...")
        df_tabla = generar_tabla(fixtures, fechas, df_equipos, ultimos)
        imprimir_tabla(df_tabla, " | ".join(fechas) if isinstance(fechas, list) else fechas)

        if EXPORTAR_CSV:
            nombre = f"nba_over_under_{datetime.now().strftime('%Y%m%d')}.csv"
            df_tabla.to_csv(nombre, index=False)
            print(f"\n✅ Guardado en {nombre}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()