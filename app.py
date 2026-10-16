"""
============================================================
 DASHBOARD STREAMLIT — NBA Over/Under
============================================================
 Dashboard interactivo para:
   - Ver la jornada actual con predicciones del modelo
   - Ejecutar backtest walk-forward
   - Comparar equipos manualmente
   - Analizar calibración y distribución de errores

 Uso:
     streamlit run app.py

 Dependencias:
     pip install streamlit plotly pandas numpy nba_api
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar funciones del script principal
from nba_ou_avanzado import (
    descargar_stats_avanzadas,
    descargar_stats_tradicionales,
    descargar_ultimos_resultados,
    descargar_jornada_actual,
    generar_tabla,
    cargar_partidos_historicos,
    backtest_walk_forward,
    calcular_metricas,
    calcular_rango_avanzado,
    calcular_rango_original,
    avedev,
)


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="NBA Over/Under",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stDataFrame {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHÉ DE DATOS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_stats_avanzadas(season):
    return descargar_stats_avanzadas(season=season)


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_stats_tradicionales(season):
    return descargar_stats_tradicionales(season=season)


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_ultimos(season):
    return descargar_ultimos_resultados(season=season)


@st.cache_data(ttl=600, show_spinner=False)
def cargar_jornada():
    return descargar_jornada_actual()


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_historicos(season):
    return cargar_partidos_historicos(season=season)


# ============================================================
# SIDEBAR — CONFIGURACIÓN
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    season = st.selectbox(
        "Temporada",
        ["2025-26", "2024-25", "2023-24"],
        index=0
    )

    multiplicador = st.slider(
        "Multiplicador de dispersión",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Factor que amplía o reduce el ancho del rango. Mayor = más cobertura."
    )

    ventana = st.slider(
        "Ventana de totales",
        min_value=5, max_value=30, value=10, step=1,
        help="Nº de partidos recientes usados para calcular la dispersión."
    )

    modelo = st.radio(
        "Modelo a mostrar",
        ["Avanzado (Pace + Eficiencias)", "Original (PPG/oPPG)", "Ambos"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 Estado")
    st.markdown(f"**Temporada:** {season}")
    st.markdown(f"**Multiplicador:** {multiplicador}")
    st.markdown(f"**Ventana:** {ventana}")


# ============================================================
# CABECERA
# ============================================================
st.markdown('<div class="main-header">🏀 NBA Over/Under Calculator</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelo avanzado con Pace + Eficiencias + ajuste local/visitante</div>',
            unsafe_allow_html=True)


# ============================================================
# CARGA DE DATOS
# ============================================================
with st.spinner("Cargando datos de la NBA..."):
    try:
        df_avanzado = cargar_stats_avanzadas(season)
        df_tradicional = cargar_stats_tradicionales(season)
        ultimos = cargar_ultimos(season)
        fechas_jornada, fixtures = cargar_jornada()
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Jornada actual",
    "📊 Backtest",
    "🔍 Comparador",
    "📈 Análisis histórico"
])


# ============================================================
# TAB 1 — JORNADA ACTUAL
# ============================================================
with tab1:
    st.markdown(f"### Partidos de la jornada")
    st.caption(f"Fechas NBA consultadas: {', '.join(fechas_jornada)}")

    if not fixtures:
        st.warning("⚠️ No hay partidos en la jornada actual.")
    else:
        with st.spinner("Calculando rangos..."):
            df_tabla = generar_tabla(fixtures, df_avanzado, df_tradicional, ultimos)

        # Métricas destacadas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Partidos", len(df_tabla))

        # Ancho medio (solo si hay datos)
        anchos = []
        for _, r in df_tabla.iterrows():
            if r['AV_MIN_OVER'] != 'N/D' and r['AV_MAX_UNDER'] != 'N/D':
                anchos.append(r['AV_MAX_UNDER'] - r['AV_MIN_OVER'])

        with col2:
            if anchos:
                st.metric("Ancho medio", f"{np.mean(anchos):.1f} pts")
            else:
                st.metric("Ancho medio", "N/D")

        with col3:
            partidos_jugados = df_tabla['MARCADOR'].notna().sum() if 'MARCADOR' in df_tabla else 0
            st.metric("Ya jugados", partidos_jugados)

        with col4:
            st.metric("Modelo", "Avanzado v2")

        # Tabla interactiva
        st.markdown("#### Tabla de predicciones")

        # Preparar columnas según el modelo elegido
        if modelo == "Avanzado (Pace + Eficiencias)":
            columnas_tabla = ['FECHA_NBA', 'LOCAL', 'VISITANTE', 'ESTADO', 'MARCADOR',
                              'AV_MIN_OVER', 'AV_MAX_UNDER', 'AV_CENTRO']
        elif modelo == "Original (PPG/oPPG)":
            columnas_tabla = ['FECHA_NBA', 'LOCAL', 'VISITANTE', 'ESTADO', 'MARCADOR',
                              'OR_MIN_OVER', 'OR_MAX_UNDER']
        else:
            columnas_tabla = ['FECHA_NBA', 'LOCAL', 'VISITANTE', 'ESTADO', 'MARCADOR',
                              'AV_MIN_OVER', 'AV_MAX_UNDER',
                              'OR_MIN_OVER', 'OR_MAX_UNDER']

        columnas_tabla = [c for c in columnas_tabla if c in df_tabla.columns]

        st.dataframe(
            df_tabla[columnas_tabla],
            use_container_width=True,
            height=min(400, 50 + len(df_tabla) * 35)
        )

        # Descarga
        csv = df_tabla.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar CSV",
            csv,
            file_name=f"nba_over_under_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # Gráfico de barras con rangos
        st.markdown("#### Rangos por partido")
        df_plot = df_tabla.copy()
        df_plot = df_plot[df_plot['AV_MIN_OVER'] != 'N/D'].copy()

        if not df_plot.empty:
            df_plot['PARTIDO'] = df_plot['LOCAL'].str.split().str[-1] + " vs " + \
                                  df_plot['VISITANTE'].str.split().str[-1]
            df_plot['AV_MIN_OVER'] = pd.to_numeric(df_plot['AV_MIN_OVER'], errors='coerce')
            df_plot['AV_MAX_UNDER'] = pd.to_numeric(df_plot['AV_MAX_UNDER'], errors='coerce')

            fig = go.Figure()

            # Rango como barra horizontal
            for _, row in df_plot.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['AV_MIN_OVER'], row['AV_MAX_UNDER']],
                    y=[row['PARTIDO'], row['PARTIDO']],
                    mode='lines',
                    line=dict(color='#667eea', width=20),
                    showlegend=False,
                    hovertemplate=f"<b>{row['PARTIDO']}</b><br>" +
                                  f"MIN: {row['AV_MIN_OVER']:.1f}<br>" +
                                  f"MAX: {row['AV_MAX_UNDER']:.1f}<extra></extra>"
                ))
                # Centro
                fig.add_trace(go.Scatter(
                    x=[row['AV_CENTRO']],
                    y=[row['PARTIDO']],
                    mode='markers',
                    marker=dict(color='white', size=10, line=dict(color='#1a1a2e', width=2)),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Resultado real si existe
            df_jugados = df_plot[df_plot['TOTAL_REAL'].notna()] if 'TOTAL_REAL' in df_plot else pd.DataFrame()
            if not df_jugados.empty:
                for _, row in df_jugados.iterrows():
                    color = 'green' if row['AV_MIN_OVER'] <= row['TOTAL_REAL'] <= row['AV_MAX_UNDER'] else 'red'
                    fig.add_trace(go.Scatter(
                        x=[row['TOTAL_REAL']],
                        y=[row['PARTIDO']],
                        mode='markers',
                        marker=dict(color=color, size=14, symbol='diamond'),
                        showlegend=False,
                        hovertemplate=f"Real: {row['TOTAL_REAL']}<extra></extra>"
                    ))

            fig.update_layout(
                height=max(300, len(df_plot) * 60),
                xaxis_title="Puntos totales",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 — BACKTEST
# ============================================================
with tab2:
    st.markdown("### Backtest walk-forward")
    st.caption(f"Simula el algoritmo en cada partido de la temporada {season} "
               "usando solo información anterior.")

    col1, col2 = st.columns([1, 3])
    with col1:
        ejecutar = st.button("🚀 Ejecutar backtest", type="primary",
                             use_container_width=True)

    if ejecutar or 'df_bt' in st.session_state:
        if ejecutar:
            with st.spinner("Descargando partidos históricos..."):
                df_historicos = cargar_historicos(season)

            with st.spinner(f"Ejecutando backtest ({len(df_historicos)} partidos)..."):
                df_bt = backtest_walk_forward(
                    df_historicos, df_avanzado, df_tradicional,
                    dispersion_fn=avedev, multiplicador=multiplicador
                )
                st.session_state['df_bt'] = df_bt

                # Exploración de multiplicadores
                with st.spinner("Explorando multiplicadores..."):
                    from nba_ou_avanzado import explorar_multiplicadores
                    df_mult = explorar_multiplicadores(
                        df_historicos, df_avanzado, df_tradicional
                    )
                    st.session_state['df_mult'] = df_mult

        df_bt = st.session_state.get('df_bt')
        df_mult = st.session_state.get('df_mult')

        if df_bt is not None and not df_bt.empty:
            metricas = calcular_metricas(df_bt)

            # KPIs
            st.markdown("#### Métricas principales")
            col1, col2, col3, col4 = st.columns(4)

            delta_hit = metricas['AV_HIT_RATE'] - metricas['OR_HIT_RATE']
            delta_mae = metricas['OR_MAE'] - metricas['AV_MAE']

            with col1:
                st.metric("Hit Rate (AV)", f"{metricas['AV_HIT_RATE']}%",
                          f"{delta_hit:+.2f}%")
            with col2:
                st.metric("MAE (AV)", f"{metricas['AV_MAE']} pts",
                          f"{delta_mae:+.2f}")
            with col3:
                st.metric("Bias (AV)", f"{metricas['AV_BIAS']:+.2f} pts")
            with col4:
                st.metric("Ancho medio", f"{metricas['AV_ANCHO']} pts")

            # Comparación AV vs OR
            st.markdown("#### Comparación de modelos")
            df_comp = pd.DataFrame({
                'Métrica': ['Hit Rate (%)', 'MAE (pts)', 'Bias (pts)', 'Ancho medio (pts)'],
                'Avanzado': [metricas['AV_HIT_RATE'], metricas['AV_MAE'],
                             metricas['AV_BIAS'], metricas['AV_ANCHO']],
                'Original': [metricas['OR_HIT_RATE'], metricas['OR_MAE'],
                             metricas['OR_BIAS'], metricas['OR_ANCHO']]
            })
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Distribución del error")
                fig = px.histogram(
                    df_bt, x='AV_ERROR', nbins=40,
                    color_discrete_sequence=['#667eea']
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red",
                              annotation_text="Cero")
                fig.add_vline(x=df_bt['AV_ERROR'].mean(), line_dash="dash",
                              line_color="green",
                              annotation_text=f"Media: {df_bt['AV_ERROR'].mean():.2f}")
                fig.update_layout(
                    xaxis_title="Error (real − centro)",
                    yaxis_title="Frecuencia",
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Calibración: predicho vs real")
                fig = px.scatter(
                    df_bt, x='AV_CENTRO', y='TOTAL_REAL',
                    opacity=0.4,
                    color_discrete_sequence=['#667eea']
                )
                # Línea ideal
                lims = [df_bt['AV_CENTRO'].min(), df_bt['AV_CENTRO'].max()]
                fig.add_trace(go.Scatter(
                    x=lims, y=lims, mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Ideal'
                ))
                fig.update_layout(
                    xaxis_title="Total predicho (centro)",
                    yaxis_title="Total real",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

            # Exploración de multiplicadores
            if df_mult is not None and not df_mult.empty:
                st.markdown("#### Trade-off por multiplicador")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_mult['AV_ANCHO'], y=df_mult['AV_HIT_RATE'],
                    mode='markers+lines+text',
                    text=df_mult['ETIQUETA'],
                    textposition='top center',
                    marker=dict(size=12, color='#667eea'),
                    name='Avanzado'
                ))
                fig.add_trace(go.Scatter(
                    x=df_mult['OR_ANCHO'], y=df_mult['OR_HIT_RATE'],
                    mode='markers+lines',
                    marker=dict(size=12, color='#f5576c'),
                    name='Original'
                ))
                fig.add_hline(y=70, line_dash="dot", line_color="green",
                              annotation_text="Objetivo 70%")
                fig.update_layout(
                    xaxis_title="Ancho medio (pts)",
                    yaxis_title="Hit Rate (%)",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabla de multiplicadores
                with st.expander("Ver tabla de multiplicadores"):
                    st.dataframe(df_mult, use_container_width=True, hide_index=True)

            # Descargar
            csv = df_bt.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar backtest completo (CSV)",
                csv,
                file_name=f"backtest_{season}.csv",
                mime="text/csv"
            )
        else:
            st.warning("El backtest no devolvió resultados.")
    else:
        st.info("👆 Pulsa **Ejecutar backtest** para empezar. "
                "Puede tardar 10-30 segundos según la temporada.")


# ============================================================
# TAB 3 — COMPARADOR
# ============================================================
with tab3:
    st.markdown("### Comparador de equipos")
    st.caption("Selecciona dos equipos para comparar el modelo avanzado vs el original.")

    equipos = sorted(df_avanzado['TEAM_NAME'].tolist())

    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("🏠 Equipo local", equipos, index=0)
    with col2:
        visitante = st.selectbox("✈️ Equipo visitante", equipos, index=1)

    if local == visitante:
        st.warning("⚠️ Selecciona dos equipos diferentes.")
    else:
        # Info del equipo local
        info_l = df_avanzado[df_avanzado['TEAM_NAME'] == local]
        info_v = df_avanzado[df_avanzado['TEAM_NAME'] == visitante]

        if not info_l.empty and not info_v.empty:
            st.markdown("#### Estadísticas de temporada")

            df_stats = pd.DataFrame({
                'Métrica': ['ORtg (Casa)', 'DRtg (Casa)', 'Pace (Casa)',
                            'ORtg (Fuera)', 'DRtg (Fuera)', 'Pace (Fuera)'],
                local: [
                    info_l.iloc[0]['ORtg_Casa'], info_l.iloc[0]['DRtg_Casa'],
                    info_l.iloc[0]['Pace_Casa'], info_l.iloc[0]['ORtg_Fuera'],
                    info_l.iloc[0]['DRtg_Fuera'], info_l.iloc[0]['Pace_Fuera']
                ],
                visitante: [
                    info_v.iloc[0]['ORtg_Casa'], info_v.iloc[0]['DRtg_Casa'],
                    info_v.iloc[0]['Pace_Casa'], info_v.iloc[0]['ORtg_Fuera'],
                    info_v.iloc[0]['DRtg_Fuera'], info_v.iloc[0]['Pace_Fuera']
                ]
            })
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

        # Predicciones
        st.markdown("#### Predicciones")

        r_av = calcular_rango_avanzado(local, visitante, df_avanzado, ultimos,
                                        multiplicador=multiplicador)
        r_or = calcular_rango_original(local, visitante, df_tradicional, ultimos)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🚀 Modelo Avanzado")
            if r_av:
                st.metric("MIN OVER", f"{r_av['MIN_OVER']} pts")
                st.metric("MAX UNDER", f"{r_av['MAX_UNDER']} pts")
                st.metric("CENTRO", f"{r_av['CENTRO']} pts")
                st.metric("Pace esperado", r_av['PACE_ESPERADO'])
                st.metric("Desviación", f"{r_av['DESVIACION']} pts")
            else:
                st.warning("No se pudo calcular.")

        with col2:
            st.markdown("##### 📊 Modelo Original")
            if r_or:
                st.metric("MIN OVER", f"{r_or['MIN_OVER']} pts")
                st.metric("MAX UNDER", f"{r_or['MAX_UNDER']} pts")
                centro_or = (r_or['MIN_OVER'] + r_or['MAX_UNDER']) / 2
                st.metric("CENTRO", f"{centro_or:.2f} pts")
                st.metric("Ancho", f"{r_or['MAX_UNDER'] - r_or['MIN_OVER']:.2f} pts")
            else:
                st.warning("No se pudo calcular.")

        # Visualización comparativa
        if r_av and r_or:
            st.markdown("#### Visualización comparativa")

            fig = go.Figure()

            # Rango avanzado
            fig.add_trace(go.Scatter(
                x=[r_av['MIN_OVER'], r_av['MAX_UNDER']],
                y=['Avanzado', 'Avanzado'],
                mode='lines',
                line=dict(color='#667eea', width=25),
                name='Avanzado'
            ))
            fig.add_trace(go.Scatter(
                x=[r_av['CENTRO']], y=['Avanzado'],
                mode='markers',
                marker=dict(color='white', size=14,
                            line=dict(color='#1a1a2e', width=2)),
                showlegend=False, hoverinfo='skip'
            ))

            # Rango original
            fig.add_trace(go.Scatter(
                x=[r_or['MIN_OVER'], r_or['MAX_UNDER']],
                y=['Original', 'Original'],
                mode='lines',
                line=dict(color='#f5576c', width=25),
                name='Original'
            ))
            fig.add_trace(go.Scatter(
                x=[(r_or['MIN_OVER'] + r_or['MAX_UNDER']) / 2], y=['Original'],
                mode='markers',
                marker=dict(color='white', size=14,
                            line=dict(color='#1a1a2e', width=2)),
                showlegend=False, hoverinfo='skip'
            ))

            fig.update_layout(
                height=250,
                xaxis_title="Puntos totales",
                yaxis_title="",
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 — ANÁLISIS HISTÓRICO
# ============================================================
with tab4:
    st.markdown("### Análisis histórico")
    st.caption("Distribución de errores por equipo, franja horaria y otros factores.")

    df_bt = st.session_state.get('df_bt')

    if df_bt is None or df_bt.empty:
        st.info("⚠️ Ejecuta primero el **Backtest** en la pestaña 2.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### MAE por equipo local")
            pivot = df_bt.groupby('LOCAL').agg(
                MAE=('AV_ABS_ERROR', 'mean'),
                N=('AV_ABS_ERROR', 'count')
            ).sort_values('MAE', ascending=False).head(15)

            fig = px.bar(
                pivot.reset_index(),
                x='MAE', y='LOCAL', orientation='h',
                color='MAE', color_continuous_scale='RdYlGn_r',
                text='MAE'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Bias por equipo local")
            pivot = df_bt.groupby('LOCAL').agg(
                BIAS=('AV_ERROR', 'mean'),
                N=('AV_ERROR', 'count')
            ).sort_values('BIAS')

            fig = px.bar(
                pivot.reset_index(),
                x='BIAS', y='LOCAL', orientation='h',
                color='BIAS', color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            fig.update_layout(
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Análisis por rango de total
        st.markdown("#### Error por rango de total real")
        df_bt['RANGO_TOTAL'] = pd.cut(
            df_bt['TOTAL_REAL'],
            bins=[150, 190, 200, 210, 220, 230, 240, 300],
            labels=['<190', '190-200', '200-210', '210-220',
                    '220-230', '230-240', '>240']
        )

        pivot = df_bt.groupby('RANGO_TOTAL', observed=True).agg(
            MAE=('AV_ABS_ERROR', 'mean'),
            BIAS=('AV_ERROR', 'mean'),
            N=('AV_ERROR', 'count'),
            HIT=('AV_DENTRO', 'mean')
        ).reset_index()
        pivot['HIT'] = (pivot['HIT'] * 100).round(1)

        st.dataframe(pivot, use_container_width=True, hide_index=True)

        # Hit rate vs ancho
        st.markdown("#### Hit rate por cuartil de ancho")
        df_bt['CUARTIL_ANCHO'] = pd.qcut(
            df_bt['AV_ANCHO'], q=4,
            labels=['Q1 (estrecho)', 'Q2', 'Q3', 'Q4 (amplio)']
        )

        pivot = df_bt.groupby('CUARTIL_ANCHO', observed=True).agg(
            ANCHO=('AV_ANCHO', 'mean'),
            HIT=('AV_DENTRO', 'mean'),
            N=('AV_DENTRO', 'count')
        ).reset_index()
        pivot['HIT'] = (pivot['HIT'] * 100).round(1)
        pivot['ANCHO'] = pivot['ANCHO'].round(1)

        st.dataframe(pivot, use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"🏀 NBA Over/Under Calculator · Temporada {season} · "
    f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)