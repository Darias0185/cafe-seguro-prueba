# ============================================================
# CAFÉ SEGURO — App Streamlit v4
# Calculadora del caficultor + Historial de Caldas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Café Seguro · Caldas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CONSTANTES ────────────────────────────────────────────────
SUMA_HA   = 1_800_000
HECTAREAS = 20_000
U9_OPT    = -0.70
U6_OPT    = -0.094
EXIT_SPI6 = -2.0

MESES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
         7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
         11:"Noviembre",12:"Diciembre"}

VERDE_M = "#2C5F2D"
ROJO    = "#C0392B"
AMBAR   = "#B8620A"
GRIS    = "#6B7F70"

# Percentiles históricos por mes (precalculados del dataset)
PRECIP_PCT = {
    1:(110.3,163.0,185.9), 2:(137.4,144.9,218.8), 3:(206.9,250.6,301.0),
    4:(268.5,293.6,336.1), 5:(273.8,305.1,351.8), 6:(150.5,214.2,263.3),
    7:(120.5,147.0,212.6), 8:(157.5,194.2,231.9), 9:(222.5,240.7,271.3),
    10:(324.6,337.8,354.2),11:(278.8,308.2,343.0),12:(178.3,194.8,244.0),
}
NDVI_PCT = {
    1:(0.743,0.777,0.793), 2:(0.704,0.742,0.764), 3:(0.701,0.726,0.754),
    4:(0.707,0.738,0.761), 5:(0.752,0.780,0.794), 6:(0.773,0.794,0.799),
    7:(0.765,0.781,0.795), 8:(0.745,0.760,0.777), 9:(0.731,0.754,0.774),
    10:(0.617,0.660,0.708),11:(0.672,0.727,0.757),12:(0.727,0.768,0.789),
}
PROD_HIST = {
    1:88.8,2:80.4,3:69.7,4:71.8,5:86.0,6:87.2,
    7:82.9,8:75.4,9:69.0,10:90.6,11:105.4,12:105.8,
}
PROD_POR_LLUVIA = {
    1:(73.0,92.0,82.8),  2:(86.0,77.5,75.1),  3:(68.6,67.8,68.9),
    4:(61.9,71.4,61.6),  5:(82.0,77.8,84.7),  6:(89.6,82.3,82.6),
    7:(81.8,88.3,73.8),  8:(86.0,69.2,74.5),  9:(57.0,77.8,67.5),
    10:(87.6,87.0,90.7), 11:(99.3,109.2,92.1), 12:(86.1,111.1,109.6),
}
SPI9_BASE = {
    1:-0.079,2:-0.095,3:-0.028,4:-0.012,5:0.017,6:0.111,
    7:0.084, 8:0.054, 9:-0.067,10:-0.038,11:0.010,12:-0.038,
}
SPI6_BASE = {
    1:-0.077,2:-0.080,3:0.106,4:0.093,5:0.067,6:0.051,
    7:0.036, 8:0.061, 9:-0.129,10:-0.100,11:-0.054,12:-0.058,
}
SPI_DELTA = {"Poca lluvia": -0.80, "Lluvia normal": 0.0, "Mucha lluvia": +0.70}

def fmt_cop(v):
    if v >= 1e9:  return f"${v/1e9:.1f} MM"
    if v >= 1e6:  return f"${v/1e6:.0f} M"
    return f"${v:,.0f}"

def pago_hibrido(spi9, spi6):
    if spi9 > U9_OPT or spi6 > U6_OPT: return 0.0
    return float(np.clip((spi6 - U6_OPT) / (EXIT_SPI6 - U6_OPT), 0, 1))

def mes_ant(m, n=1): return ((m - 1 - n) % 12) + 1

# ── CARGA ─────────────────────────────────────────────────────
@st.cache_data
def cargar():
    dm = pd.read_csv("cafe_seguro_master.csv")
    dm['fecha'] = pd.to_datetime(dm['fecha'])
    dm = dm.sort_values('fecha').reset_index(drop=True)

    COLS = ['departamento','fecha','spi_6','spi_9','cat_oficial_spi_9',
            'pago_spi_mejor','evento_sequia_conocido']
    ds = (pd.read_csv("resultado_spi_multiescala_pago.csv", usecols=COLS)
          .assign(fecha=lambda x: pd.to_datetime(x['fecha']))
          .query("departamento=='Caldas'")
          .drop(columns='departamento'))

    df = pd.merge(dm, ds, on='fecha', how='inner').sort_values('fecha').reset_index(drop=True)

    for i in [1,2,3,6,12]: df[f'prod_lag{i}']  = df['prod_caldas_miles_sacos'].shift(i)
    for i in [1,3,6,10]:   df[f'precip_lag{i}'] = df['precip_mm'].shift(i)
    for i in [1,3,5]:      df[f'ndvi_lag{i}']   = df['ndvi_mean'].shift(i)
    df['mes_sin']   = np.sin(2*np.pi*df['month']/12)
    df['mes_cos']   = np.cos(2*np.pi*df['month']/12)
    df['tendencia'] = range(len(df))

    FC = ['prod_lag1','prod_lag2','prod_lag3','prod_lag6','prod_lag12',
          'precip_lag1','precip_lag3','precip_lag6','precip_lag10',
          'ndvi_lag1','ndvi_lag3','ndvi_lag5','mes_sin','mes_cos','tendencia']

    mask  = df[FC].notna().all(axis=1)
    model = joblib.load("modelo_rf_m4.pkl")
    dfm   = df[mask].copy()
    dfm['prod_estimada_rf'] = model.predict(dfm[FC])
    df['prod_estimada_rf']  = np.nan
    df.loc[mask, 'prod_estimada_rf'] = dfm['prod_estimada_rf'].values

    ph = (df[df['fecha'].dt.year<=2018]
          .groupby('month')['prod_caldas_miles_sacos'].mean()
          .rename('prod_hist_mensual'))
    df = df.merge(ph, on='month', how='left')
    df['var_obs']  = ((df['prod_caldas_miles_sacos']-df['prod_hist_mensual'])
                      / df['prod_hist_mensual']*100).round(1)
    df['pago_cop'] = (df['pago_spi_mejor']*SUMA_HA*HECTAREAS).round(0)
    return df, model, FC

with st.spinner("Cargando datos y modelo..."):
    df, modelo, FC = cargar()

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("### 🌿 Café Seguro · Caldas")
st.caption("Herramienta de consulta para el caficultor · Caldas, Colombia")
st.divider()

tab1, tab2 = st.tabs(["☕  Calculadora", "📊  Historial de Caldas"])

# ════════════════════════════════════════════════════════════
# PESTAÑA 1 — CALCULADORA
# ════════════════════════════════════════════════════════════
with tab1:

    col_form, col_result = st.columns([1, 2], gap="large")

    with col_form:
        st.markdown("#### ¿Qué quiere consultar?")

        mes_sel = st.selectbox(
            "Mes a consultar",
            options=list(range(1, 13)),
            format_func=lambda m: MESES[m],
            index=10,
        )
        lluvia_sel = st.radio(
            "¿Cómo estuvo la lluvia?",
            ["Poca lluvia", "Lluvia normal", "Mucha lluvia"],
            index=1,
        )
        vigor_sel = st.radio(
            "¿Cómo está la vegetación de su finca?",
            ["Vegetación baja", "Vegetación normal", "Vegetación alta"],
            index=1,
        )

        st.button("🔍 Consultar", type="primary", use_container_width=True)

        st.info(
            "Estimación de referencia departamental para Caldas. "
            "No reemplaza el dato real de su finca ni constituye "
            "una liquidación oficial del seguro.",
            icon="ℹ️",
        )

    with col_result:

        # ── CÁLCULO ───────────────────────────────────────────
        ll_idx = {"Poca lluvia":0,"Lluvia normal":1,"Mucha lluvia":2}[lluvia_sel]
        vg_idx = {"Vegetación baja":0,"Vegetación normal":1,"Vegetación alta":2}[vigor_sel]

        precip_val = PRECIP_PCT[mes_sel][ll_idx]
        ndvi_val   = NDVI_PCT[mes_sel][vg_idx]
        prod_hist  = PROD_HIST[mes_sel]

        features = {
            'prod_lag1'   : PROD_HIST[mes_ant(mes_sel, 1)],
            'prod_lag2'   : PROD_HIST[mes_ant(mes_sel, 2)],
            'prod_lag3'   : PROD_HIST[mes_ant(mes_sel, 3)],
            'prod_lag6'   : PROD_HIST[mes_ant(mes_sel, 6)],
            'prod_lag12'  : prod_hist,
            'precip_lag1' : precip_val,
            'precip_lag3' : PRECIP_PCT[mes_ant(mes_sel, 3)][1],
            'precip_lag6' : PRECIP_PCT[mes_ant(mes_sel, 6)][1],
            'precip_lag10': PRECIP_PCT[mes_ant(mes_sel, 10)][1],
            'ndvi_lag1'   : ndvi_val,
            'ndvi_lag3'   : NDVI_PCT[mes_ant(mes_sel, 3)][1],
            'ndvi_lag5'   : NDVI_PCT[mes_ant(mes_sel, 5)][1],
            'mes_sin'     : np.sin(2*np.pi*mes_sel/12),
            'mes_cos'     : np.cos(2*np.pi*mes_sel/12),
            'tendencia'   : 264,
        }

        X_pred       = pd.DataFrame([features])[FC]
        prod_est     = round(float(modelo.predict(X_pred)[0]), 1)
        diff_pct     = round((prod_est - prod_hist) / prod_hist * 100, 1)
        delta_spi    = SPI_DELTA[lluvia_sel]
        spi9_est     = round(SPI9_BASE[mes_sel] + delta_spi, 3)
        spi6_est     = round(SPI6_BASE[mes_sel] + delta_spi * 0.8, 3)
        pago_pct     = pago_hibrido(spi9_est, spi6_est)
        pago_cop     = round(pago_pct * SUMA_HA * HECTAREAS, 0)
        activa9      = spi9_est <= U9_OPT
        activa6      = spi6_est <= U6_OPT
        seguro_activa= activa9 and activa6

        # ── CARDS DE RESULTADO ────────────────────────────────
        st.markdown(
            f"#### Resultado · {MESES[mes_sel]} · "
            f"{lluvia_sel.lower()} · {vigor_sel.lower()}"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Producción estimada",
            f"{prod_est:.0f} mil sacos",
            delta=f"{diff_pct:+.1f}% vs. promedio de {MESES[mes_sel]}",
            delta_color="normal" if diff_pct >= 0 else "inverse",
        )
        r1.caption(f"Promedio histórico {MESES[mes_sel]}: {prod_hist:.0f} mil sacos")

        if seguro_activa:
            r2.metric("¿El seguro paga?", "Sí activa",
                      delta=f"SPI-9 = {spi9_est:+.2f}")
            r2.success("El clima supera el umbral del seguro.")
        else:
            r2.metric("¿El seguro paga?", "No activa",
                      delta=f"SPI-9 = {spi9_est:+.2f}", delta_color="off")
            r2.info("El clima no supera el umbral del seguro.")

        r3.metric(
            "Pago estimado",
            fmt_cop(pago_cop) if pago_cop > 0 else "$0",
            delta=f"{pago_pct:.1%} de la suma asegurada" if pago_cop > 0 else "Sin pago",
            delta_color="normal" if pago_cop > 0 else "off",
        )
        r3.caption(f"Suma asegurada máxima: {fmt_cop(SUMA_HA * HECTAREAS)}")

        st.divider()

        # ── MENSAJE EN LENGUAJE LLANO ─────────────────────────
        mes_nom = MESES[mes_sel]
        if seguro_activa and diff_pct < 0:
            st.success(
                f"Con **{lluvia_sel.lower()}** y **{vigor_sel.lower()}** en {mes_nom}, "
                f"la producción estimada de Caldas estaría "
                f"**{abs(diff_pct):.1f}% por debajo** del promedio histórico. "
                f"La sequía acumulada **activa el seguro** y recibiría un pago de "
                f"**{fmt_cop(pago_cop)}**."
            )
        elif seguro_activa and diff_pct >= 0:
            st.warning(
                f"Con **{lluvia_sel.lower()}** en {mes_nom}, el seguro activaría "
                f"(**{fmt_cop(pago_cop)}**), aunque la producción estimada no muestra "
                f"caída significativa. Esto ocurre cuando la sequía es acumulada "
                f"pero aún no impacta la cosecha."
            )
        elif not seguro_activa and diff_pct < -10:
            st.warning(
                f"La producción estimada estaría **{abs(diff_pct):.1f}% por debajo** "
                f"del promedio de {mes_nom}, pero la lluvia acumulada "
                f"**no supera el umbral del seguro**. "
                f"En este escenario no habría pago aunque haya caída productiva."
            )
        else:
            st.info(
                f"Con **{lluvia_sel.lower()}** y **{vigor_sel.lower()}** en {mes_nom}, "
                f"las condiciones están dentro de lo normal. "
                f"Producción estimada: **{prod_est:.0f} mil sacos**. "
                f"El seguro **no se activaría** este mes."
            )

        st.divider()

        # ── CHIPS SPI ─────────────────────────────────────────
        st.markdown("**Condición de lluvia acumulada (referencia)**")
        cs1, cs2 = st.columns(2)
        cs1.metric(
            f"9 meses acumulados {'✅' if activa9 else '⬜'}",
            f"{spi9_est:+.2f}",
            delta="Activa el seguro" if activa9 else f"Umbral: {U9_OPT}",
            delta_color="inverse" if activa9 else "off",
        )
        cs2.metric(
            f"6 meses acumulados {'✅' if activa6 and activa9 else '⬜'}",
            f"{spi6_est:+.2f}",
            delta="Determina el pago" if (activa9 and activa6) else f"Umbral: {U6_OPT}",
            delta_color="inverse" if (activa9 and activa6) else "off",
        )
        st.caption(
            "Estimación basada en el promedio histórico del mes ajustado "
            "por el nivel de lluvia seleccionado. No son mediciones en tiempo real."
        )

        st.divider()

        # ── CONTEXTO HISTÓRICO ────────────────────────────────
        st.markdown(f"**Producción histórica de {MESES[mes_sel]} en Caldas según lluvia**")
        st.caption("Promedio 2002–2023 · miles de sacos de 60 kg")

        prod_b, prod_n, prod_a = PROD_POR_LLUVIA[mes_sel]
        colores = [
            ROJO    if ll_idx == 0 else "#CBD5C0",
            VERDE_M if ll_idx == 1 else "#CBD5C0",
            AMBAR   if ll_idx == 2 else "#CBD5C0",
        ]
        fig_b = go.Figure(go.Bar(
            x=["Poca lluvia", "Lluvia normal", "Mucha lluvia"],
            y=[prod_b, prod_n, prod_a],
            marker_color=colores,
            text=[f"{v:.0f}" for v in [prod_b, prod_n, prod_a]],
            textposition='outside',
            hovertemplate='%{x}<br>Promedio: <b>%{y:.0f}</b> mil sacos<extra></extra>',
        ))
        fig_b.add_hline(
            y=prod_hist,
            line=dict(color=GRIS, width=1.5, dash='dot'),
            annotation_text=f"Promedio general: {prod_hist:.0f}",
            annotation_position="right",
            annotation_font_size=10,
        )
        fig_b.update_layout(
            height=200, margin=dict(l=0, r=80, t=20, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(title='Miles de sacos', gridcolor='#E8EDE9',
                       range=[0, max(prod_b, prod_n, prod_a, prod_hist) * 1.25]),
            showlegend=False, font=dict(size=11),
        )
        st.plotly_chart(fig_b, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PESTAÑA 2 — HISTORIAL
# ════════════════════════════════════════════════════════════
with tab2:

    anios = sorted(df['fecha'].dt.year.unique())
    col_sb, col_main = st.columns([1, 3], gap="large")

    with col_sb:
        st.markdown("#### Filtros")
        anio_sel = st.selectbox("Año", anios, index=anios.index(2020))
        meses_d  = sorted(df[df['fecha'].dt.year==anio_sel]['fecha'].dt.month.unique())
        mes_h    = st.selectbox("Mes", meses_d,
                                format_func=lambda m: MESES[m],
                                index=min(3, len(meses_d)-1))
        st.divider()
        ult = df.dropna(subset=['spi_9']).iloc[-1]
        st.markdown(f"**Condición actual · {ult['fecha'].strftime('%b %Y')}**")
        st.metric("SPI-9", f"{ult['spi_9']:+.2f}", ult['cat_oficial_spi_9'])
        st.divider()
        st.markdown("**Contrato de referencia**")
        st.markdown("Suma asegurada / ha: **$1.8 M COP**")
        st.markdown("Prima estimada: **$42 k / mes**")
        st.caption("Fuente: FNC · CHIRPS · NASA MODIS\nModelo: Random Forest M4")

    with col_main:
        fila = df[(df['fecha'].dt.year==anio_sel)&(df['fecha'].dt.month==mes_h)]
        if fila.empty:
            st.warning("Sin datos para el mes seleccionado.")
            st.stop()

        r          = fila.iloc[0]
        pago_pct_h = float(r['pago_spi_mejor'])
        prod_real  = float(r['prod_caldas_miles_sacos'])
        var_obs    = float(r['var_obs'])
        spi9_h     = float(r['spi_9'])
        pago_cop_h = float(r['pago_cop'])
        nivel_h    = str(r['cat_oficial_spi_9'])
        sequia_h   = int(r['evento_sequia_conocido'])

        if pago_pct_h > 0 and sequia_h == 1:
            est_txt, est_fn = "✅ El seguro pagó", st.success
        elif pago_pct_h > 0 and sequia_h == 0:
            est_txt, est_fn = "⚠️ Pagó sin sequía confirmada", st.warning
        elif pago_pct_h == 0 and sequia_h == 1:
            est_txt, est_fn = "🚨 Hubo sequía pero no pagó", st.error
        else:
            est_txt, est_fn = "⬜ Mes sin novedad", st.info

        st.markdown(f"**{MESES[mes_h]} {anio_sel} — {est_txt}**")

        h1, h2, h3 = st.columns(3)
        h1.metric("Producción real", f"{prod_real:.0f} mil sacos",
                  delta=f"{var_obs:+.1f}% vs. promedio",
                  delta_color="normal" if var_obs >= 0 else "inverse")
        h2.metric("Pago del seguro",
                  fmt_cop(pago_cop_h) if pago_cop_h > 0 else "$0",
                  delta=f"{pago_pct_h:.1%} de la suma asegurada" if pago_pct_h > 0 else "No activó",
                  delta_color="normal" if pago_pct_h > 0 else "off")
        h3.metric("Lluvia acumulada (SPI-9)", f"{spi9_h:+.3f}", nivel_h)

        est_fn(
            f"Producción real: **{prod_real:.0f} mil sacos** "
            f"({'por debajo' if var_obs < 0 else 'por encima'} del promedio "
            f"de {MESES[mes_h]} en {abs(var_obs):.1f}%). "
            f"{'El seguro activó con un pago de **' + fmt_cop(pago_cop_h) + '**.' if pago_cop_h > 0 else 'El seguro no activó este mes.'}"
        )

        st.divider()

        # Gráfica
        st.markdown("**Producción observada vs. esperada · Caldas**")
        yr_min = int(df['fecha'].dt.year.min())
        yr_max = int(df['fecha'].dt.year.max())
        rango = st.select_slider(
            "Periodo", label_visibility="collapsed",
            options=list(range(yr_min, yr_max+1)),
            value=(max(yr_min, anio_sel-4), min(yr_max, anio_sel+2)),
        )
        dg = df[(df['fecha'].dt.year >= rango[0]) &
                (df['fecha'].dt.year <= rango[1])].dropna(subset=['prod_estimada_rf']).copy()

        fig = go.Figure()
        dg['caida'] = dg['prod_caldas_miles_sacos'] < dg['prod_estimada_rf']
        grp = (dg['caida'] != dg['caida'].shift()).cumsum()
        for _, b in dg[dg['caida']].groupby(grp[dg['caida']]):
            fig.add_trace(go.Scatter(
                x=list(b['fecha'])+list(b['fecha'][::-1]),
                y=list(b['prod_estimada_rf'])+list(b['prod_caldas_miles_sacos'][::-1]),
                fill='toself', fillcolor='rgba(192,57,43,0.10)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=dg['fecha'], y=dg['prod_estimada_rf'],
            name='Esperada (RF)', line=dict(color=VERDE_M, width=2, dash='dash'),
            hovertemplate='%{x|%b %Y}<br>Esperada: <b>%{y:.1f}</b> mil sacos<extra></extra>'))
        fig.add_trace(go.Scatter(x=dg['fecha'], y=dg['prod_caldas_miles_sacos'],
            name='Real', line=dict(color=ROJO, width=2),
            hovertemplate='%{x|%b %Y}<br>Real: <b>%{y:.1f}</b> mil sacos<extra></extra>'))
        sv = dg[(dg['fecha'].dt.year==anio_sel)&(dg['fecha'].dt.month==mes_h)]
        if not sv.empty:
            s = sv.iloc[0]
            fig.add_trace(go.Scatter(x=[s['fecha']], y=[s['prod_caldas_miles_sacos']],
                mode='markers', marker=dict(color=ROJO, size=12, line=dict(color='white',width=2)),
                name='Mes seleccionado'))
            fig.add_vline(x=s['fecha'].timestamp()*1000,
                          line=dict(color=GRIS, width=1, dash='dot'))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=4,b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h',yanchor='bottom',y=1.02,x=0,font=dict(size=10)),
            xaxis=dict(showgrid=False,tickformat='%Y'),
            yaxis=dict(title='Miles de sacos',gridcolor='#E8EDE9'),
            hovermode='x unified', font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "⚠️ Las caídas de producción por exceso de lluvia (SPI positivo) no activan "
            "el seguro — limitación documentada del índice SPI para el café en Caldas."
        )

        st.divider()

        # Tabla detalle
        st.markdown("**Últimos 6 meses**")
        sel_f = pd.Timestamp(year=anio_sel, month=mes_h, day=1)
        tabla = df[df['fecha'] <= sel_f].tail(6).copy()

        def clf(row):
            p = row['pago_spi_mejor'] > 0
            s = row['evento_sequia_conocido'] == 1
            if p and s:       return 'Pagó'
            if p and not s:   return 'Falso positivo'
            if not p and s:   return 'Pérdida sin pago'
            return 'Normal'

        tabla['Estado']     = tabla.apply(clf, axis=1)
        tabla['% Pago']     = tabla['pago_spi_mejor'].map(lambda x: f"{x:.1%}")
        tabla['Var. prod.'] = tabla['var_obs'].map(lambda x: f"{x:+.1f}%")
        tabla['Pago COP']   = tabla['pago_cop'].map(lambda x: fmt_cop(x) if x > 0 else "—")
        tabla['SPI-9']      = tabla['spi_9'].map(lambda x: f"{x:.3f}")
        tabla['Mes']        = tabla['fecha'].dt.strftime('%b %Y')

        st.dataframe(
            tabla[['Mes','SPI-9','Estado','% Pago','Var. prod.','Pago COP']],
            use_container_width=True, hide_index=True,
            column_config={
                "Mes"       : st.column_config.TextColumn("Mes",        width="small"),
                "SPI-9"     : st.column_config.TextColumn("SPI-9",      width="small"),
                "Estado"    : st.column_config.TextColumn("Estado",     width="medium"),
                "% Pago"    : st.column_config.TextColumn("% Pago",     width="small"),
                "Var. prod.": st.column_config.TextColumn("Var. prod.", width="small"),
                "Pago COP"  : st.column_config.TextColumn("Pago COP",   width="small"),
            })

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Café Seguro · Prototipo académico · Grupo 16 · MIAD · Universidad de los Andes · Mayo 2026  |  "
    "Modelo: Random Forest M4 · Esquema híbrido SPI-9/SPI-6  |  "
    f"Fuentes: FNC · CHIRPS · NASA MODIS  |  "
    f"Datos hasta: {df['fecha'].max().strftime('%b %Y')}"
)
