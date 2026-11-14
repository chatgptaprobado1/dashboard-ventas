import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dashboard Comercial de Ventas", page_icon="📊", layout="wide")


@st.cache_data
def cargar_datos():
    df = pd.read_csv("ventas.csv", parse_dates=["Fecha"])
    return df.drop_duplicates().dropna()


df_total = cargar_datos()

st.title("📊 Dashboard Comercial de Ventas")

# ---------- Filtros ----------
st.sidebar.header("Filtros")
vendedores = st.sidebar.multiselect(
    "Vendedor", sorted(df_total["Vendedor"].unique()), default=sorted(df_total["Vendedor"].unique())
)
productos = st.sidebar.multiselect(
    "Producto", sorted(df_total["Producto"].unique()), default=sorted(df_total["Producto"].unique())
)

df = df_total[df_total["Vendedor"].isin(vendedores) & df_total["Producto"].isin(productos)]

if df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ---------- KPIs ----------
ventas_producto = df.groupby("Producto")["Ventas"].sum().sort_values(ascending=False)
ventas_vendedor = df.groupby("Vendedor")["Ventas"].sum().sort_values(ascending=False)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ventas totales", f"S/ {df['Ventas'].sum():,.0f}")
k2.metric("Unidades vendidas", f"{df['Cantidad'].sum()}")
k3.metric("Ticket promedio", f"S/ {df['Ventas'].mean():,.1f}")
k4.metric("Producto top", ventas_producto.idxmax())
k5.metric("Mejor vendedor", ventas_vendedor.idxmax())

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 Resumen", "🤖 Predicción IA", "📋 Datos"])

# ---------- TAB 1: Resumen ----------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Ventas por Producto")
        fig = px.bar(
            ventas_producto.reset_index(), x="Producto", y="Ventas",
            color="Producto", text_auto=True,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Participación por Vendedor")
        fig = px.pie(ventas_vendedor.reset_index(), names="Vendedor", values="Ventas")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Tendencia de Ventas")
        ventas_fecha = df.groupby("Fecha", as_index=False)["Ventas"].sum()
        fig = px.line(ventas_fecha, x="Fecha", y="Ventas", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Unidades Vendidas por Producto")
        cantidad = df.groupby("Producto", as_index=False)["Cantidad"].sum()
        fig = px.bar(cantidad, x="Producto", y="Cantidad", color="Producto", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2: Predicción ----------
with tab2:
    st.subheader("Predicción IA: crecimiento esperado")
    diario = df.groupby("Fecha", as_index=False)["Ventas"].sum().sort_values("Fecha")

    if len(diario) < 3:
        st.info("Se necesitan al menos 3 fechas con ventas para calcular la predicción.")
    else:
        n_dias = st.slider("Días a proyectar", 3, 14, 7)

        x = (diario["Fecha"] - diario["Fecha"].min()).dt.days.to_numpy().reshape(-1, 1)
        modelo = LinearRegression().fit(x, diario["Ventas"])

        ultimo = int(x.max())
        x_futuro = np.arange(ultimo + 1, ultimo + 1 + n_dias).reshape(-1, 1)
        pred = np.clip(modelo.predict(x_futuro), 0, None)
        fechas_futuras = diario["Fecha"].max() + pd.to_timedelta(np.arange(1, n_dias + 1), unit="D")

        pendiente = modelo.coef_[0]
        m1, m2 = st.columns(2)
        m1.metric("Tendencia diaria", f"S/ {pendiente:+,.1f} por día")
        m2.metric(f"Ventas proyectadas ({n_dias} días)", f"S/ {pred.sum():,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diario["Fecha"], y=diario["Ventas"],
                                 mode="lines+markers", name="Ventas reales"))
        fig.add_trace(go.Scatter(x=fechas_futuras, y=pred, mode="lines+markers",
                                 name="Predicción", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Ventas (S/)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Modelo: regresión lineal sobre las ventas diarias. "
                   "Con pocos datos es una estimación referencial.")

# ---------- TAB 3: Datos ----------
with tab3:
    st.subheader("Base de datos comercial")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="ventas_filtradas.csv",
        mime="text/csv",
    )