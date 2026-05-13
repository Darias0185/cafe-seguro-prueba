# ============================================================
# CAFÉ SEGURO — APP STREAMLIT v3
# Fixes: sin CSS/HTML crudo, cobertura corregida, botón → checkbox
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
    initial_sidebar_state="expanded",
)

# ── CONSTANTES ────────────────────────────────────────────────
SUMA_HA      = 1_800_000
HECTAREAS    = 20_000
U9_OPT       = -0.70
U6_OPT       = -0.094
EXIT_SPI6    = -2.0
VERDE_MED    = "#2C5F2D"
ROJO         = "#C0392B"
AMBAR        = "#B8620A"
VERDE_P      = "#1A6E3A"
GRIS         = "#6B7F70"

MESES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
         7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
         11:"Noviembre",12:"Diciembre"}

def fmt_cop(v):
    if v >= 1e9:  return f"${v/1e9:.1f} MM"
    if v >= 1e6:  return f"${v/1e6:.0f} M"
    return f"${v:,.0f}"

def pago_hibrido(spi9, spi6, u9, u6, exit6=EXIT_SPI6):
    if spi9 > u9 or spi6 > u6: return 0.0
    return float(np.clip((spi6 - u6) / (exit6 - u6), 0, 1))

def estado_label(pago, sequia):
    if pago > 0 and sequia == 1:  return "✅ Pagó",              VERDE_P
    if pago > 0 and sequia == 0:  return "⚠️ Falso positivo",    AMBAR
    if pago == 0 and sequia == 1: return "🚨 Pérdida sin pago",  ROJO
    return "⬜ Normal",            GRIS

# ── CARGA ─────────────────────────────────────────────────────
@st.cache_data
def cargar():
    dm = pd.read_csv("cafe_seguro_master.csv")
    dm['fecha'] = pd.to_datetime(dm['fecha'])
    dm = dm.sort_values('fecha').reset_index(drop=True)

    COLS = ['departamento','fecha','spi_6','spi_9','cat_oficial_spi_9',
            'pago_spi_mejor','pago_spi_6','pago_spi_9','evento_sequia_conocido']
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

    ph = (df[df['fecha'].dt.year <= 2018]
          .groupby('month')['prod_caldas_miles_sacos'].mean()
          .rename('prod_hist_mensual'))
    df = df.merge(ph, on='month', how='left')
    df['var_obs']  = ((df['prod_caldas_miles_sacos'] - df['prod_hist_mensual'])
                      / df['prod_hist_mensual'] * 100).round(1)
    df['var_est']  = ((df['prod_estimada_rf'] - df['prod_hist_mensual'])
                      / df['prod_hist_mensual'] * 100).round(1)
    df['pago_cop'] = (df['pago_spi_mejor'] * SUMA_HA * HECTAREAS).round(0)
    return df

with st.spinner("Cargando datos y modelo..."):
    df = cargar()

anios = sorted(df['fecha'].dt.year.unique())

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Café Seguro")
    st.caption("Seguro indexado · Caldas")
    st.divider()
    st.markdown("**CONSULTA**")
    st.selectbox("Departamento", ["Caldas"])
    anio_sel = st.selectbox("Año",  anios, index=anios.index(2020))
    meses_d  = sorted(df[df['fecha'].dt.year == anio_sel]['fecha'].dt.month.unique())
    mes_sel  = st.selectbox("Mes", meses_d,
                             format_func=lambda m: MESES[m],
                             index=min(3, len(meses_d) - 1))
    st.divider()
    ult = df.dropna(subset=['spi_9']).iloc[-1]
    st.markdown(f"**Riesgo actual · {ult['fecha'].strftime('%b %Y')}**")
    st.metric("SPI-9", f"{ult['spi_9']:+.2f}", ult['cat_oficial_spi_9'])
    st.divider()
    st.markdown("**Contrato de referencia**")
    st.markdown("Suma asegurada / ha: **$1.8 M COP**")
    st.markdown("Prima estimada: **$42 k / mes**")
    st.caption("Fuente: FNC · CHIRPS · NASA MODIS\nModelo: Random Forest M4")

# ── DATOS DEL MES ─────────────────────────────────────────────
fila = df[(df['fecha'].dt.year == anio_sel) & (df['fecha'].dt.month == mes_sel)]
if fila.empty:
    st.warning("Sin datos para el mes seleccionado.")
    st.stop()

r         = fila.iloc[0]
pago_pct  = float(r['pago_spi_mejor'])
prod_real = float(r['prod_caldas_miles_sacos'])
prod_hist = float(r['prod_hist_mensual'])
var_obs   = float(r['var_obs'])
spi9      = float(r['spi_9'])
spi6      = float(r['spi_6'])
pago_cop  = float(r['pago_cop'])
nivel     = str(r['cat_oficial_spi_9'])
sequia    = int(r['evento_sequia_conocido'])
est_txt, est_color = estado_label(pago_pct, sequia)

# Pago máximo posible = 100% de la suma asegurada total
pago_maximo = SUMA_HA * HECTAREAS
# Cobertura = qué % del pago máximo se activó
pct_suma_asegurada = pago_pct * 100   # e.g. 67.7%

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("### 🌿 Seguro indexado café · Caldas")
st.caption(f"Departamento: Caldas  ·  Periodo de análisis: 2002–2025")
st.divider()

# ════════════════════════════════════════════════════════════
# SECCIÓN 1 — RESPUESTA PRINCIPAL
# ════════════════════════════════════════════════════════════
st.markdown(f"**1. Respuesta principal — {MESES[mes_sel]} {anio_sel}**")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("##### 📉 Afectación del cultivo")
    ca, cb = st.columns(2)
    ca.metric("Prod. esperada (hist.)",
              f"{prod_hist:.0f} k sacos")
    cb.metric("Prod. observada",
              f"{prod_real:.0f} k sacos",
              delta=f"{var_obs:+.1f}%",
              delta_color="normal" if var_obs >= 0 else "inverse")

with c2:
    st.markdown("##### 🛡️ ¿El seguro respondió?")
    # Estado con color usando st.markdown mínimo
    estado_icon = "🟢" if "Pagó" in est_txt else ("🔴" if "sin pago" in est_txt else "🟡")
    st.markdown(f"**Estado:** {est_txt}")
    ce, cf = st.columns(2)
    ce.metric("Pago estimado", fmt_cop(pago_cop))
    cf.metric("% Suma asegurada", f"{pct_suma_asegurada:.1f}%",
              delta=f"de ${pago_maximo/1e9:.1f} MM máximo")

with c3:
    st.markdown("##### ⭐ Evaluación del sistema")
    if pago_pct > 0 and var_obs < 0:
        st.info(f"El seguro activó con **{pct_suma_asegurada:.1f}%** de la suma asegurada.\n\n"
                f"La producción cayó **{abs(var_obs):.1f}%** respecto al histórico.")
    elif pago_pct > 0 and var_obs >= 0:
        st.warning("El seguro pagó sin caída productiva significativa.")
    elif sequia == 1 and pago_pct == 0:
        st.error("Sequía real confirmada pero el seguro **no activó** (falso negativo).")
    else:
        st.success("Mes sin afectación relevante.")
    cg, ch = st.columns(2)
    cg.metric("Pago total", fmt_cop(pago_cop) if pago_pct > 0 else "—")
    ch.metric("SPI-9", f"{spi9:+.3f}", nivel)

st.divider()

# ════════════════════════════════════════════════════════════
# SECCIÓN 2 + 3
# ════════════════════════════════════════════════════════════
col_prod, col_spi = st.columns([6, 4])

with col_prod:
    st.markdown("**2. Producción observada vs. esperada**")

    yr_min = int(df['fecha'].dt.year.min())
    yr_max = int(df['fecha'].dt.year.max())
    rango = st.select_slider(
        "Periodo del gráfico", label_visibility="collapsed",
        options=list(range(yr_min, yr_max + 1)),
        value=(max(yr_min, anio_sel - 4), min(yr_max, anio_sel + 2)))

    dg = df[(df['fecha'].dt.year >= rango[0]) &
            (df['fecha'].dt.year <= rango[1])].dropna(subset=['prod_estimada_rf']).copy()

    fig = go.Figure()
    dg['caida'] = dg['prod_caldas_miles_sacos'] < dg['prod_estimada_rf']
    grp = (dg['caida'] != dg['caida'].shift()).cumsum()
    for _, b in dg[dg['caida']].groupby(grp[dg['caida']]):
        fig.add_trace(go.Scatter(
            x=list(b['fecha']) + list(b['fecha'][::-1]),
            y=list(b['prod_estimada_rf']) + list(b['prod_caldas_miles_sacos'][::-1]),
            fill='toself', fillcolor='rgba(192,57,43,0.12)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(
        x=dg['fecha'], y=dg['prod_estimada_rf'],
        name='Prod. esperada (RF)',
        line=dict(color=VERDE_MED, width=2, dash='dash'),
        hovertemplate='%{x|%b %Y}<br>Esperada: <b>%{y:.1f}</b> k sacos<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=dg['fecha'], y=dg['prod_caldas_miles_sacos'],
        name='Prod. real',
        line=dict(color=ROJO, width=2),
        hovertemplate='%{x|%b %Y}<br>Real: <b>%{y:.1f}</b> k sacos<extra></extra>'))

    sv = dg[(dg['fecha'].dt.year == anio_sel) & (dg['fecha'].dt.month == mes_sel)]
    if not sv.empty:
        s = sv.iloc[0]
        fig.add_trace(go.Scatter(
            x=[s['fecha']], y=[s['prod_caldas_miles_sacos']],
            mode='markers',
            marker=dict(color=ROJO, size=12, line=dict(color='white', width=2)),
            name='Mes seleccionado'))
        fig.add_vline(x=s['fecha'].timestamp() * 1000,
                      line=dict(color=GRIS, width=1, dash='dot'))

    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=4, b=0),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=10)),
        xaxis=dict(showgrid=False, tickformat='%Y'),
        yaxis=dict(title='Miles de sacos', gridcolor='#E8EDE9'),
        hovermode='x unified', font=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)
    st.warning("⚠️ **Nota:** Las caídas por exceso hídrico (SPI positivo) no activan el seguro "
               "— limitación estructural del índice SPI documentada en el análisis.")

with col_spi:
    st.markdown("**3. Condiciones climáticas (SPI)**")
    activa9 = spi9 <= U9_OPT
    activa6 = spi6 <= U6_OPT

    st.metric(
        label=f"Persistencia — SPI-9  {'✅ ACTIVA' if activa9 else '⬜ no activa'}",
        value=f"{spi9:+.3f}",
        delta=f"umbral ≤ {U9_OPT}",
        delta_color="inverse" if activa9 else "off")

    st.metric(
        label=f"Intensidad — SPI-6  {'✅ Determina pago' if (activa9 and activa6) else '⬜ sin intensidad'}",
        value=f"{spi6:+.3f}",
        delta=f"umbral ≤ {U6_OPT}  |  exit {EXIT_SPI6}",
        delta_color="inverse" if (activa9 and activa6) else "off")

    st.info(f"**Fórmula:** Pago = A(SPI-9) × P(SPI-6)  →  **{pago_pct:.1%}**\n\n"
            f"SPI-9 activa el evento · SPI-6 modula la severidad\n\n"
            f"Condición: **{nivel}**")

st.divider()

# ════════════════════════════════════════════════════════════
# HISTORIAL + TABLA
# ════════════════════════════════════════════════════════════
st.markdown("**Historial de activaciones**")

rango_h = st.select_slider(
    "Rango historial", label_visibility="collapsed",
    options=list(range(yr_min, yr_max + 1)),
    value=(max(yr_min, anio_sel - 5), min(yr_max, anio_sel + 1)))

dh = df[(df['fecha'].dt.year >= rango_h[0]) &
        (df['fecha'].dt.year <= rango_h[1])].copy()

def clf(row):
    p = row['pago_spi_mejor'] > 0
    s = row['evento_sequia_conocido'] == 1
    if p and s:       return 'Pagó',            VERDE_P
    if p and not s:   return 'Falso positivo',  AMBAR
    if not p and s:   return 'Pérdida sin pago','#E67E22'
    return 'Normal',  '#CBD5C0'

dh['est_h'], dh['col_h'] = zip(*dh.apply(clf, axis=1))
dh['altura'] = dh['pago_spi_mejor'].clip(lower=0.08)
sel_f = pd.Timestamp(year=anio_sel, month=mes_sel, day=1)

figh = go.Figure()
for est, col in [('Pagó', VERDE_P), ('Falso positivo', AMBAR),
                 ('Pérdida sin pago', '#E67E22'), ('Normal', '#CBD5C0')]:
    sub = dh[dh['est_h'] == est]
    if sub.empty: continue
    figh.add_trace(go.Bar(
        x=sub['fecha'], y=sub['altura'],
        name=est, marker_color=col, marker_line_width=0,
        customdata=sub['pago_spi_mejor'],
        hovertemplate='%{x|%b %Y}<br>Pago: <b>%{customdata:.1%}</b>'
                      '<extra>' + est + '</extra>'))

sel_h = dh[dh['fecha'] == sel_f]
if not sel_h.empty:
    figh.add_trace(go.Bar(
        x=sel_h['fecha'], y=sel_h['altura'],
        marker=dict(color='rgba(0,0,0,0)', line=dict(color='#1A3A2A', width=2.5)),
        showlegend=False, hoverinfo='skip'))

figh.update_layout(
    barmode='overlay', height=150,
    margin=dict(l=0, r=0, t=4, b=0),
    plot_bgcolor='white', paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=10)),
    xaxis=dict(showgrid=False, tickformat='%Y', dtick='M12'),
    yaxis=dict(showticklabels=False, range=[0, 1.15]),
    bargap=0.15, font=dict(size=10))
st.plotly_chart(figh, use_container_width=True)

# ── TABLA DETALLE ─────────────────────────────────────────────
st.markdown("**Detalle por periodo · últimos 6 meses**")
tabla = df[df['fecha'] <= sel_f].tail(6).copy()
tabla['Estado']    = tabla.apply(lambda r: clf(r)[0], axis=1)
tabla['% Pago']    = tabla['pago_spi_mejor'].map(lambda x: f"{x:.1%}")
tabla['Var. obs.'] = tabla['var_obs'].map(lambda x: f"{x:+.1f}%")
tabla['Pago COP']  = tabla['pago_cop'].map(lambda x: fmt_cop(x) if x > 0 else "—")
tabla['SPI-9']     = tabla['spi_9'].map(lambda x: f"{x:.3f}")
tabla['SPI-6']     = tabla['spi_6'].map(lambda x: f"{x:.3f}")
tabla['Mes']       = tabla['fecha'].dt.strftime('%b %Y')

st.dataframe(
    tabla[['Mes', 'SPI-9', 'SPI-6', 'Estado', '% Pago', 'Var. obs.', 'Pago COP']],
    use_container_width=True, hide_index=True,
    column_config={
        "Mes"      : st.column_config.TextColumn("Mes",       width="small"),
        "SPI-9"    : st.column_config.TextColumn("SPI-9",     width="small"),
        "SPI-6"    : st.column_config.TextColumn("SPI-6",     width="small"),
        "Estado"   : st.column_config.TextColumn("Estado",    width="medium"),
        "% Pago"   : st.column_config.TextColumn("% Pago",    width="small"),
        "Var. obs.": st.column_config.TextColumn("Var. obs.", width="small"),
        "Pago COP" : st.column_config.TextColumn("Pago COP",  width="small"),
    })
st.caption(f"▶ mes seleccionado: {sel_f.strftime('%b %Y')}  ·  {len(df)} periodos disponibles")
st.divider()

# ════════════════════════════════════════════════════════════
# SECCIÓN 4 + 5
# ════════════════════════════════════════════════════════════
col_ev, col_sim = st.columns([5, 5])

with col_ev:
    st.markdown("**4. Evaluación histórica (2015–2023)**")
    dev = df[(df['fecha'].dt.year >= 2015) & (df['fecha'].dt.year <= 2023)].copy()
    pa  = dev['pago_spi_mejor'] > 0
    sq  = dev['evento_sequia_conocido'] == 1
    tot = len(dev)
    cor = int(((pa & sq) | (~pa & ~sq)).sum())
    fp  = int((pa  & ~sq).sum())
    fn  = int((~pa & sq).sum())
    pr  = cor / (cor + fp) if (cor + fp) > 0 else 0
    rc  = cor / (cor + fn) if (cor + fn) > 0 else 0
    f2  = (5 * pr * rc) / (4 * pr + rc) if (pr + rc) > 0 else 0

    fig_d = go.Figure(go.Pie(
        labels=['Correctos', 'Falsos positivos', 'Falsos negativos'],
        values=[cor, fp, fn], hole=0.55,
        marker=dict(colors=[VERDE_P, AMBAR, ROJO],
                    line=dict(color='white', width=2)),
        textinfo='percent', textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>%{value} meses (%{percent})<extra></extra>'))
    fig_d.add_annotation(
        text=f'<b>{tot}</b><br>meses',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color='#1A3A2A'), align='center')
    fig_d.update_layout(
        height=220, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor='white', showlegend=True,
        legend=dict(orientation='v', x=1.0, y=0.5, font=dict(size=10)))
    st.plotly_chart(fig_d, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Correctos",   f"{cor/tot*100:.1f}%", f"{cor} meses")
    m2.metric("Falsos neg.", f"{fn/tot*100:.1f}%",  f"{fn} meses", delta_color="inverse")
    m3.metric("F2-score",    f"{f2:.3f}",            "híbrido")

    pt    = (dev['spi_9'] <= -1.0) & (dev['spi_6'] <= -1.0)
    cor_t = int(((pt & sq) | (~pt & ~sq)).sum())
    fp_t  = int((pt  & ~sq).sum())
    fn_t  = int((~pt & sq).sum())
    pr_t  = cor_t / (cor_t + fp_t) if (cor_t + fp_t) > 0 else 0
    rc_t  = cor_t / (cor_t + fn_t) if (cor_t + fn_t) > 0 else 0
    f2_t  = (5 * pr_t * rc_t) / (4 * pr_t + rc_t) if (pr_t + rc_t) > 0 else 0
    d_cor = (cor - cor_t) / tot * 100
    d_fn  = (fn  - fn_t)  / tot * 100

    st.success(f"**Mejora vs. SPI-9 ≤ −1.0 (tradicional):**\n\n"
               f"+{d_cor:.1f} pp en correctos  ·  "
               f"{d_fn:+.1f} pp en FN  ·  "
               f"F2: {f2_t:.3f} → **{f2:.3f}**")
    st.caption("FN = sequía real sin pago — el caso más crítico. Por eso se usa F2.")

with col_sim:
    st.markdown("**5. Simulador de diseño del seguro**")

    # Checkbox para restaurar → evita el botón que falla en localtunnel
    usar_optimos = st.checkbox("Usar umbrales calibrados óptimos", value=True)

    if usar_optimos:
        u9_val, u6_val = U9_OPT, U6_OPT
        st.caption(f"Umbrales óptimos activos: SPI-9 = {U9_OPT}  ·  SPI-6 = {U6_OPT}")
        u9 = st.slider("Umbral activación (SPI-9)", -1.5, -0.2,
                       U9_OPT, 0.05, disabled=True,
                       help=f"Calibrado óptimo: {U9_OPT}")
        u6 = st.slider("Umbral intensidad (SPI-6)", -1.5, 0.5,
                       U6_OPT, 0.05, disabled=True,
                       help=f"Calibrado óptimo: {U6_OPT}")
    else:
        u9 = st.slider("Umbral activación (SPI-9)", -1.5, -0.2,
                       U9_OPT, 0.05,
                       help=f"Calibrado óptimo: {U9_OPT}")
        u6 = st.slider("Umbral intensidad (SPI-6)", -1.5, 0.5,
                       U6_OPT, 0.05,
                       help=f"Calibrado óptimo: {U6_OPT}")
        u9_val, u6_val = u9, u6

    dsim = dev.copy()
    dsim['ps'] = dsim.apply(
        lambda r: pago_hibrido(r['spi_9'], r['spi_6'], u9_val, u6_val), axis=1)
    pas  = dsim['ps'] > 0
    acts = int(pas.sum())
    cors = int(((pas & sq) | (~pas & ~sq)).sum())
    fps  = int((pas  & ~sq).sum())
    fns  = int((~pas & sq).sum())
    tseq = int(sq.sum())
    cobs = (pas & sq).sum() / tseq * 100 if tseq > 0 else 0
    prs  = cors / (cors + fps) if (cors + fps) > 0 else 0
    rcs  = cors / (cors + fns) if (cors + fns) > 0 else 0
    f2s  = (5 * prs * rcs) / (4 * prs + rcs) if (prs + rcs) > 0 else 0
    cob_ref = (pa & sq).sum() / tseq * 100 if tseq > 0 else 0

    ks1, ks2 = st.columns(2)
    ks3, ks4 = st.columns(2)
    ks1.metric("Activaciones",     f"{acts}",
               f"{acts - int(pa.sum()):+d} vs óptimo")
    ks2.metric("Cobertura",        f"{cobs:.1f}%",
               f"{cobs - cob_ref:+.1f} pp")
    ks3.metric("Falsos positivos", f"{fps/tot*100:.1f}%",
               f"{fps/tot*100 - fp/tot*100:+.1f} pp", delta_color="inverse")
    ks4.metric("Falsos negativos", f"{fns/tot*100:.1f}%",
               f"{fns/tot*100 - fn/tot*100:+.1f} pp", delta_color="inverse")

    if f2s >= f2:
        st.success(f"**F2 = {f2s:.3f}** ≥ referencia óptima ({f2:.3f}) ✅")
    else:
        st.warning(f"**F2 = {f2s:.3f}** < referencia óptima ({f2:.3f}) "
                   f"— los umbrales reducen el desempeño.")
    st.caption(f"Deltas vs umbrales calibrados: SPI-9 = {U9_OPT}  /  SPI-6 = {U6_OPT}")

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Prototipo: Random Forest M4 + esquema híbrido SPI-9/SPI-6  |  "
    "Grupo 16 · MIAD · Universidad de los Andes · Mayo 2026  |  "
    f"Fuentes: FNC · CHIRPS · NASA MODIS  |  "
    f"Última actualización: {df['fecha'].max().strftime('%b %Y')}"
)
