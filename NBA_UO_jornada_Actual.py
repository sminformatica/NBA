"""
============================================================
 CALCULADORA OVER/UNDER — NBA
 Jornada actual (madrugada + noche, huso español)
============================================================
 Consulta el scoreboard de los últimos 2 días NBA para cubrir
 la jornada completa tal como se percibe desde España:
   - Partidos de "ayer" en EE.UU. → madrugada de hoy en España
   - Partidos de "hoy" en EE.UU.  → noche de hoy en España

 Uso:
     python NBA_UO.py

 Dependencias:
     pip install nba_api pandas
============================================================
"""

import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import scoreboardv3
import time
import warnings

# Silenciar solo warnings concretos (no todo)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='nba_api')


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
EXPORTAR_CSV = True

TZ_ES = ZoneInfo("Europe/Madrid")

# Estados que consideramos "partido finalizado"
ESTADOS_FINALES = {"Final", "Final/OT", "Final/2OT", "Final/3OT", "Final/4OT"}

# Reintentos ligeros para llamadas a nba_api (no cambian la salida,
# solo evitan fallos transitorios de red / rate-limit puntual)
RETRIES = 3
RETRY_SLEEP = 2.0


def _con_reintentos(fn, *args, **kwargs):
    """Ejecuta fn(*args, **kwargs) con reintentos sencillos."""
    ultimo_error = None
    for intento in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            ultimo_error = e
            if intento < RETRIES:
                time.sleep(RETRY_SLEEP * intento)
    raise ultimo_error


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

# Alias conocidos: nombre devuelto por la API → nombre en el listado estático.
# (La API de stats.nba.com usa abreviaturas coloquiales en varios endpoints.)
ALIASES_API = {
    'la clippers':        'Los Angeles Clippers',
    'la lakers':          'Los Angeles Lakers',
    'ny knicks':          'New York Knicks',
    'gs warriors':        'Golden State Warriors',
    'no pelicans':        'New Orleans Pelicans',
    'sa spurs':           'San Antonio Spurs',
    'okc thunder':        'Oklahoma City Thunder',
    'phx suns':           'Phoenix Suns',
    'wsh wizards':        'Washington Wizards',
    'cha hornets':        'Charlotte Hornets',
    'min timberwolves':   'Minnesota Timberwolves',
    'bkn nets':           'Brooklyn Nets',
    'nj nets':            'Brooklyn Nets',
    'nohornets':          'New Orleans Pelicans',
    'no hornets':         'New Orleans Pelicans',
    'charlotte bobcats':  'Charlotte Hornets',
    'seattle supersonics':'Oklahoma City Thunder',
}

_EQUIPOS_NBA = nba_teams.get_teams()


def construir_mapa_equipos():
    mapa = {}
    for eq in _EQUIPOS_NBA:
        nombre_oficial = eq['full_name']
        for clave in [eq['full_name'], eq['nickname'], eq['city'],
                      eq['abbreviation'], f"{eq['city']} {eq['nickname']}"]:
            mapa[clave.lower().strip()] = nombre_oficial
    # Añadimos los alias de la API
    mapa.update(ALIASES_API)
    return mapa


MAPA_EQUIPOS = construir_mapa_equipos()

# Nicknames para el fallback por sufijo ("LA Clippers" termina en "clippers")
_NICKNAMES = {eq['nickname'].lower().strip(): eq['full_name']
              for eq in _EQUIPOS_NBA}

# Cache de avisos: evita repetir el mismo warning cientos de veces
_AVISOS_EMITIDOS = set()


def resolver_nombre(nombre_api):
    if not nombre_api:
        return None
    clave = nombre_api.lower().strip()

    # 1) Coincidencia directa (nombre oficial, nickname, city, abbr o alias)
    if clave in MAPA_EQUIPOS:
        return MAPA_EQUIPOS[clave]

    # 2) Fallback: ¿el string termina en algún nickname conocido?
    #    Cubre variantes del tipo "LA Clippers", "GS Warriors", etc.
    for nick, full in _NICKNAMES.items():
        if clave.endswith(nick):
            return full

    # 3) No mapeado: avisar una sola vez por nombre único
    if clave not in _AVISOS_EMITIDOS:
        _AVISOS_EMITIDOS.add(clave)
        print(f"       ⚠ Equipo no mapeado: {nombre_api!r}")
    return nombre_api


# ============================================================
# 4. ESTADÍSTICAS DE EQUIPOS
# ============================================================
def descargar_estadisticas_equipos(season=SEASON, season_type=SEASON_TYPE):
    print(f"[1/4] Descargando estadísticas de equipos NBA {season}...")

    df_base = _con_reintentos(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
        ).get_data_frames()[0]
    )

    df_opp = _con_reintentos(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Opponent',
        ).get_data_frames()[0]
    )

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
# 5. ÚLTIMOS RESULTADOS
# ============================================================
def descargar_ultimos_resultados(season=SEASON, season_type=SEASON_TYPE):
    print(f"[2/4] Descargando últimos resultados...")

    df = _con_reintentos(
        lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
        ).get_data_frames()[0]
    )

    if df.empty:
        raise ValueError(f"LeagueGameFinder no devolvió partidos para {season}.")

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
# 6. JORNADA ACTUAL (ayer + hoy en formato NBA)
# ============================================================
def descargar_jornada_actual():
    """
    Consulta el scoreboard de ayer y hoy (formato NBA) para cubrir
    la jornada completa tal como se percibe desde España.
    """
    print(f"[3/4] Descargando jornada actual (ayer + hoy NBA)...")
    hoy = datetime.now(TZ_ES)

    fixtures = []
    fechas_info = []

    # Ayer y hoy en formato NBA
    for offset in [-1, 0]:
        fecha = (hoy + timedelta(days=offset)).strftime('%Y-%m-%d')
        try:
            data = _con_reintentos(
                lambda f=fecha: scoreboardv3.ScoreboardV3(
                    game_date=f, league_id='00'
                ).get_dict()
            )
            games = data.get('scoreboard', {}).get('games', [])

            if not games:
                print(f"       → {fecha}: sin partidos")
                time.sleep(0.6)
                continue

            for g in games:
                local = resolver_nombre(g['homeTeam']['teamName'])
                visitante = resolver_nombre(g['awayTeam']['teamName'])
                estado = (g.get('gameStatusText') or '').strip()

                fixtures.append({
                    'FECHA_NBA': fecha,
                    'LOCAL': local,
                    'VISITANTE': visitante,
                    'ESTADO': estado,
                    'PTS_LOCAL': g.get('homeTeam', {}).get('score'),
                    'PTS_VISITANTE': g.get('awayTeam', {}).get('score'),
                })

            fechas_info.append(f"{fecha} ({len(games)} partidos)")
            print(f"       → {fecha}: {len(games)} partidos")
            time.sleep(0.6)

        except Exception as e:
            print(f"       → Error en {fecha}: {e}")
            time.sleep(0.6)
            continue

    if not fixtures:
        print("       → No se encontraron partidos en la jornada actual.")
        return [], []

    print(f"       → Total partidos: {len(fixtures)}")
    return fechas_info, fixtures


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
def _esta_finalizado(f):
    """
    Un partido se considera finalizado si su ESTADO es uno de los
    estados finales conocidos o, como respaldo, si ambos marcadores
    son enteros positivos.
    """
    estado = (f.get('ESTADO') or '').strip()
    if estado in ESTADOS_FINALES:
        return True

    pl, pv = f.get('PTS_LOCAL'), f.get('PTS_VISITANTE')
    try:
        pl_i = int(pl)
        pv_i = int(pv)
    except (TypeError, ValueError):
        return False
    return pl_i > 0 and pv_i > 0


def generar_tabla(fixtures, df_equipos, ultimos):
    filas = []
    for f in fixtures:
        local = f['LOCAL']
        visitante = f['VISITANTE']
        rango = calcular_rango(local, visitante, df_equipos, ultimos)

        fila = {
            'FECHA_NBA': f['FECHA_NBA'],
            'LOCAL': local,
            'VISITANTE': visitante,
            'ESTADO': f['ESTADO'],
            'MARCADOR': '',
            'MARCADOR_TOTAL': '',
        }

        # Si el partido ya se ha jugado, mostrar marcador y total
        if _esta_finalizado(f):
            try:
                pts_l = int(f['PTS_LOCAL'])
                pts_v = int(f['PTS_VISITANTE'])
                fila['MARCADOR'] = f"{pts_v}-{pts_l}"      # visitante-local
                fila['MARCADOR_TOTAL'] = pts_v + pts_l
            except (ValueError, TypeError):
                pass

        if rango is None:
            fila['MIN_OVER'] = 'N/D'
            fila['MAX_UNDER'] = 'N/D'
            fila['RANGO'] = 'N/D'
        else:
            fila['MIN_OVER'] = rango['MIN_OVER']
            fila['MAX_UNDER'] = rango['MAX_UNDER']
            fila['RANGO'] = round(rango['MAX_UNDER'] - rango['MIN_OVER'], 2)

        filas.append(fila)

    return pd.DataFrame(filas)


# ============================================================
# 9. IMPRESIÓN
# ============================================================
def imprimir_tabla(df_tabla, fechas_info):
    print("\n" + "=" * 130)
    print(f"  TABLA OVER/UNDER — NBA | Jornada actual")
    print(f"  Fechas NBA consultadas: {', '.join(fechas_info)}")
    print("=" * 130)

    if df_tabla.empty:
        print("  No hay partidos para mostrar.")
        print("=" * 130)
        return

    columnas = ['FECHA_NBA', 'LOCAL', 'VISITANTE', 'ESTADO', 'MARCADOR',
                'MARCADOR_TOTAL', 'MIN_OVER', 'MAX_UNDER', 'RANGO']
    print(df_tabla[columnas].to_string(index=False))
    print("=" * 130)
    print("\n  Interpretación:")
    print("    🟢 OVER con valor  → línea de apuestas < MIN_OVER")
    print("    🔴 UNDER con valor → línea de apuestas > MAX_UNDER")
    print("    RANGO = MAX_UNDER − MIN_OVER")
    print("    MARCADOR_TOTAL = puntos totales del partido (solo si ya se jugó)")
    print("\n  Nota: 'FECHA_NBA' indica el día NBA del partido.")
    print("        Los partidos del día anterior se vieron en la madrugada española.")


# ============================================================
# 10. MAIN
# ============================================================
def main():
    print("=" * 120)
    print("  CALCULADORA OVER/UNDER — NBA")
    print(f"  Temporada {SEASON}")
    print("=" * 120)

    try:
        df_equipos = descargar_estadisticas_equipos()
        ultimos = descargar_ultimos_resultados()
        fechas_info, fixtures = descargar_jornada_actual()

        if not fixtures:
            print("\n❌ No se encontraron partidos para la jornada actual.")
            return

        print(f"[4/4] Generando tabla over/under...")
        df_tabla = generar_tabla(fixtures, df_equipos, ultimos)
        imprimir_tabla(df_tabla, fechas_info)

        if EXPORTAR_CSV:
            nombre = f"nba_over_under_{datetime.now(TZ_ES).strftime('%Y%m%d')}.csv"
            df_tabla.to_csv(nombre, index=False)
            print(f"\n✅ Guardado en {nombre}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()