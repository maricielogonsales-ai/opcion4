import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Inventory Intelligence Framework",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Framework de Optimización de Inventarios")
st.caption(
    "Pronóstico mensual + selección automática del mejor método por producto + simulación + optimización de inventarios"
)

METODOS_PRONOSTICO = [
    "Naive",
    "Promedio móvil",
    "SES",
    "Regresión lineal",
    "ARIMA",
    "SARIMA",
    "Holt-Winters",
    "Croston",
]


# =========================================================
# FUNCIONES DE DATOS
# =========================================================
def convertir_a_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte cualquier base diaria/semanal/mensual a demanda mensual por producto."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    df["demand_real"] = pd.to_numeric(df["demand_real"], errors="coerce").fillna(0)
    df["demand_real"] = df["demand_real"].clip(lower=0)
    df = df.dropna(subset=["date"])

    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df_mensual = (
        df.groupby(["product_id", "date"], as_index=False)["demand_real"]
        .sum()
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )

    if df_mensual.empty:
        raise ValueError("No hay datos válidos después de convertir la información a meses.")

    return df_mensual


def generar_demanda_sintetica(n_productos: int = 5, meses: int = 36, seed: int = 42) -> pd.DataFrame:
    """Genera demanda mensual sintética para pruebas."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(start="2023-01-01", periods=meses, freq="MS")
    dataframes = []

    for i in range(1, n_productos + 1):
        producto = f"PROD_{i:03d}"
        base = rng.integers(500, 2500)
        tendencia = rng.uniform(-10, 30)
        estacionalidad = rng.uniform(100, 400)
        ruido = rng.normal(0, base * 0.15, meses)
        tiempo = np.arange(meses)

        demanda = base + tendencia * tiempo + estacionalidad * np.sin(2 * np.pi * tiempo / 12) + ruido
        demanda = np.maximum(0, np.round(demanda)).astype(int)

        if i % 4 == 0:
            mascara_intermitente = rng.random(meses) < 0.45
            demanda = np.where(mascara_intermitente, 0, demanda)

        dataframes.append(
            pd.DataFrame(
                {
                    "date": fechas,
                    "product_id": producto,
                    "demand_real": demanda,
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)


def leer_archivo_subido(uploaded_file) -> pd.DataFrame:
    """Lee CSV o Excel, normaliza columnas y agrupa la demanda por mes."""
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Sube un archivo CSV o Excel.")

    df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "fecha": "date",
        "mes": "date",
        "periodo": "date",
        "período": "date",
        "día": "date",
        "dia": "date",
        "producto": "product_id",
        "sku": "product_id",
        "id_producto": "product_id",
        "codigo": "product_id",
        "código": "product_id",
        "demanda": "demand_real",
        "venta": "demand_real",
        "ventas": "demand_real",
        "cantidad": "demand_real",
        "unidades": "demand_real",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    columnas_requeridas = ["date", "product_id", "demand_real"]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(faltantes)
            + ". Usa columnas: date, product_id, demand_real. "
            + "También puede reconocer nombres como fecha, mes, producto, sku, ventas o demanda."
        )

    df = df[columnas_requeridas].copy()
    return convertir_a_mensual(df)


# =========================================================
# PRONÓSTICOS MENSUALES
# =========================================================
def asegurar_prediccion_valida(pred, serie) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pred.size != len(serie):
        pred = np.resize(pred, len(serie))
    valor_relleno = float(np.nanmean(serie)) if len(serie) else 0.0
    if np.isnan(valor_relleno):
        valor_relleno = 0.0
    pred = np.where(np.isfinite(pred), pred, valor_relleno)
    return np.maximum(0, pred)


def forecast_naive(serie: np.ndarray) -> np.ndarray:
    if len(serie) == 0:
        return np.array([])
    pred = np.empty(len(serie), dtype=float)
    pred[0] = serie[0]
    if len(serie) > 1:
        pred[1:] = serie[:-1]
    return asegurar_prediccion_valida(pred, serie)


def forecast_promedio_movil(serie: np.ndarray, ventana: int = 3) -> np.ndarray:
    if len(serie) == 0:
        return np.array([])
    pred = np.empty(len(serie), dtype=float)
    pred[0] = serie[0]
    for i in range(1, len(serie)):
        inicio = max(0, i - ventana)
        pred[i] = np.mean(serie[inicio:i])
    return asegurar_prediccion_valida(pred, serie)


def forecast_regresion(serie: np.ndarray) -> np.ndarray:
    if len(serie) == 0:
        return np.array([])
    x = np.arange(len(serie)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, serie)
    pred = modelo.predict(x)
    return asegurar_prediccion_valida(pred, serie)


def forecast_ses(serie: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    if len(serie) < 3:
        return np.repeat(np.mean(serie), len(serie))

    try:
        modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
        ajuste = modelo.fit(smoothing_level=alpha, optimized=False)
        pred = np.asarray(ajuste.fittedvalues)
        return asegurar_prediccion_valida(pred, serie)
    except Exception:
        return forecast_promedio_movil(serie)


def forecast_arima(serie: np.ndarray) -> np.ndarray:
    if len(serie) < 12:
        return forecast_ses(serie)

    try:
        modelo = ARIMA(serie, order=(1, 1, 1))
        ajuste = modelo.fit()
        pred = np.asarray(ajuste.fittedvalues)
        return asegurar_prediccion_valida(pred, serie)
    except Exception:
        return forecast_ses(serie)


def forecast_sarima(serie: np.ndarray) -> np.ndarray:
    if len(serie) < 24:
        return forecast_arima(serie)

    try:
        modelo = SARIMAX(
            serie,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        ajuste = modelo.fit(disp=False)
        pred = np.asarray(ajuste.fittedvalues)
        return asegurar_prediccion_valida(pred, serie)
    except Exception:
        return forecast_arima(serie)


def forecast_holt_winters(serie: np.ndarray) -> np.ndarray:
    if len(serie) < 24:
        try:
            modelo = ExponentialSmoothing(
                serie,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            ajuste = modelo.fit(optimized=True)
            pred = np.asarray(ajuste.fittedvalues)
            return asegurar_prediccion_valida(pred, serie)
        except Exception:
            return forecast_ses(serie)

    try:
        modelo = ExponentialSmoothing(
            serie,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        ajuste = modelo.fit(optimized=True)
        pred = np.asarray(ajuste.fittedvalues)
        return asegurar_prediccion_valida(pred, serie)
    except Exception:
        return forecast_ses(serie)


def forecast_croston(serie: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    serie = np.asarray(serie, dtype=float)
    n = len(serie)
    if n == 0:
        return np.array([])
    if np.all(serie == 0):
        return np.zeros(n)

    first_nonzero_idx = np.argmax(serie > 0)
    z = serie[first_nonzero_idx]
    p = first_nonzero_idx + 1 if first_nonzero_idx + 1 > 0 else 1
    q = z / p

    pred = np.zeros(n, dtype=float)
    interval = 1
    for t in range(n):
        pred[t] = q
        if serie[t] > 0:
            z = alpha * serie[t] + (1 - alpha) * z
            p = alpha * interval + (1 - alpha) * p
            q = z / max(p, 1e-9)
            interval = 1
        else:
            interval += 1
    return asegurar_prediccion_valida(pred, serie)


def aplicar_metodo_pronostico(serie: np.ndarray, metodo: str) -> np.ndarray:
    if metodo == "Naive":
        return forecast_naive(serie)
    if metodo == "Promedio móvil":
        return forecast_promedio_movil(serie)
    if metodo == "Regresión lineal":
        return forecast_regresion(serie)
    if metodo == "ARIMA":
        return forecast_arima(serie)
    if metodo == "SARIMA":
        return forecast_sarima(serie)
    if metodo == "Holt-Winters":
        return forecast_holt_winters(serie)
    if metodo == "Croston":
        return forecast_croston(serie)
    return forecast_ses(serie)


def calcular_errores(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = y_real.sum()

    mae = np.mean(np.abs(y_real - y_pred)) if len(y_real) else 0.0

    if suma_real == 0:
        return {"wMAPE": 0.0, "Bias": 0.0, "MAE": mae}

    wmape = np.sum(np.abs(y_real - y_pred)) / suma_real
    bias = np.sum(y_pred - y_real) / suma_real
    return {"wMAPE": wmape, "Bias": bias, "MAE": mae}


def generar_forecast(df: pd.DataFrame, metodo: str) -> pd.DataFrame:
    resultados = []

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        pred = aplicar_metodo_pronostico(serie, metodo)
        err = calcular_errores(serie, pred)

        sub["demand_forecast"] = np.round(pred, 2)
        sub["method_used"] = metodo
        sub["method_wmape"] = err["wMAPE"]
        sub["method_bias"] = err["Bias"]
        resultados.append(sub)

    return pd.concat(resultados, ignore_index=True)


def generar_forecast_mejor_por_producto(df: pd.DataFrame):
    forecasts_finales = []
    comparacion = []

    for producto, sub in df.groupby("product_id"):
        sub = sub.sort_values("date").copy()
        serie = sub["demand_real"].to_numpy(dtype=float)
        predicciones = {}
        filas_producto = []

        for metodo in METODOS_PRONOSTICO:
            pred = aplicar_metodo_pronostico(serie, metodo)
            predicciones[metodo] = pred
            err = calcular_errores(serie, pred)

            fila = {
                "Producto": producto,
                "Método": metodo,
                "wMAPE": err["wMAPE"],
                "Bias": err["Bias"],
                "Abs_Bias": abs(err["Bias"]),
                "MAE": err["MAE"],
            }
            comparacion.append(fila)
            filas_producto.append(fila)

        comp_producto = pd.DataFrame(filas_producto)
        mejor_fila = comp_producto.sort_values(["wMAPE", "Abs_Bias", "MAE"]).iloc[0]
        mejor_metodo = mejor_fila["Método"]

        sub["demand_forecast"] = np.round(predicciones[mejor_metodo], 2)
        sub["method_used"] = mejor_metodo
        sub["method_wmape"] = float(mejor_fila["wMAPE"])
        sub["method_bias"] = float(mejor_fila["Bias"])
        forecasts_finales.append(sub)

    df_comparacion = pd.DataFrame(comparacion)
    mejores = (
        df_comparacion.sort_values(["Producto", "wMAPE", "Abs_Bias", "MAE"])
        .groupby("Producto", as_index=False)
        .first()[["Producto", "Método"]]
        .rename(columns={"Método": "Mejor método"})
    )

    df_comparacion = df_comparacion.merge(mejores, on="Producto", how="left")
    df_comparacion["Es mejor"] = df_comparacion["Método"] == df_comparacion["Mejor método"]
    df_comparacion = df_comparacion.drop(columns=["Abs_Bias"])

    return pd.concat(forecasts_finales, ignore_index=True), df_comparacion


# =========================================================
# SIMULACIÓN DE INVENTARIO MENSUAL
# =========================================================
@dataclass
class ParametrosInventario:
    initial_stock: int
    lead_time_months: int
    review_period_months: int
    ss_months: int
    q_fixed: int
    lot_size: int
    cost_order: float
    cost_holding_month: float
    cost_stockout: float


def redondear_lote(cantidad: float, lote: int) -> int:
    if cantidad <= 0:
        return 0
    lote = max(1, int(lote))
    return int(math.ceil(cantidad / lote) * lote)


def simular_producto(df_producto: pd.DataFrame, politica: str, p: ParametrosInventario) -> pd.DataFrame:
    df_producto = df_producto.sort_values("date").reset_index(drop=True).copy()
    stock_fisico = float(p.initial_stock)
    pipeline = {}
    resultados = []

    demanda_promedio_mensual = max(0.01, df_producto["demand_forecast"].mean())

    for t, fila in df_producto.iterrows():
        llegada = pipeline.pop(t, 0)
        stock_fisico += llegada

        demanda_durante_lead_time = demanda_promedio_mensual * p.lead_time_months
        stock_seguridad = demanda_promedio_mensual * p.ss_months
        punto_reorden = demanda_durante_lead_time + stock_seguridad
        nivel_objetivo = demanda_promedio_mensual * (
            p.lead_time_months + p.review_period_months + p.ss_months
        )

        posicion_inventario = stock_fisico + sum(pipeline.values())
        orden = 0

        if politica == "RS - revisión periódica":
            if t % p.review_period_months == 0:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sS - punto de reorden y nivel máximo":
            if posicion_inventario <= punto_reorden:
                orden = max(0, nivel_objetivo - posicion_inventario)
        elif politica == "sQ - punto de reorden y cantidad fija":
            if posicion_inventario <= punto_reorden:
                orden = p.q_fixed

        orden = redondear_lote(orden, p.lot_size)

        if orden > 0:
            mes_llegada = t + p.lead_time_months
            pipeline[mes_llegada] = pipeline.get(mes_llegada, 0) + orden

        demanda_real = float(fila["demand_real"])
        venta_real = min(stock_fisico, demanda_real)
        venta_perdida = max(0, demanda_real - stock_fisico)
        stock_fisico -= venta_real

        resultados.append(
            {
                "date": fila["date"],
                "product_id": fila["product_id"],
                "method_used": fila.get("method_used", ""),
                "demand_real": demanda_real,
                "demand_forecast": fila["demand_forecast"],
                "inventory_level": stock_fisico,
                "inventory_position": posicion_inventario,
                "order_placed": orden,
                "arrivals": llegada,
                "sales_real": venta_real,
                "sales_lost": venta_perdida,
                "reorder_point_s": punto_reorden,
                "target_level_S": nivel_objetivo,
                "is_stockout": int(venta_perdida > 0),
            }
        )

    return pd.DataFrame(resultados)


def calcular_kpis(df_sim: pd.DataFrame, p: ParametrosInventario) -> dict:
    demanda_total = df_sim["demand_real"].sum()
    ventas_perdidas = df_sim["sales_lost"].sum()
    ordenes = (df_sim["order_placed"] > 0).sum()
    inventario_promedio = df_sim["inventory_level"].mean()

    fill_rate = 1 - ventas_perdidas / demanda_total if demanda_total > 0 else 1
    costo_ordenar = ordenes * p.cost_order
    costo_mantener = df_sim["inventory_level"].sum() * p.cost_holding_month
    costo_quiebre = ventas_perdidas * p.cost_stockout
    costo_total = costo_ordenar + costo_mantener + costo_quiebre

    return {
        "fill_rate": fill_rate,
        "avg_inventory": inventario_promedio,
        "lost_sales_units": ventas_perdidas,
        "stockout_months": int(df_sim["is_stockout"].sum()),
        "orders": int(ordenes),
        "ordering_cost": costo_ordenar,
        "holding_cost": costo_mantener,
        "stockout_cost": costo_quiebre,
        "total_cost": costo_total,
    }


def optimizar_stock_seguridad(
    df_producto: pd.DataFrame,
    politica: str,
    p_base: ParametrosInventario,
    ss_max: int,
) -> pd.DataFrame:
    filas = []

    for ss in range(0, ss_max + 1):
        p = ParametrosInventario(
            initial_stock=p_base.initial_stock,
            lead_time_months=p_base.lead_time_months,
            review_period_months=p_base.review_period_months,
            ss_months=ss,
            q_fixed=p_base.q_fixed,
            lot_size=p_base.lot_size,
            cost_order=p_base.cost_order,
            cost_holding_month=p_base.cost_holding_month,
            cost_stockout=p_base.cost_stockout,
        )
        sim = simular_producto(df_producto, politica, p)
        kpis = calcular_kpis(sim, p)
        filas.append({"ss_months": ss, **kpis})

    return pd.DataFrame(filas)


# =========================================================
# VISUALIZACIONES
# =========================================================
def grafico_forecast(df_producto: pd.DataFrame) -> go.Figure:
    metodo = df_producto["method_used"].iloc[0] if "method_used" in df_producto.columns else ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_producto["date"],
            y=df_producto["demand_real"],
            mode="lines+markers",
            name="Demanda real mensual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_producto["date"],
            y=df_producto["demand_forecast"],
            mode="lines+markers",
            name=f"Pronóstico mensual ({metodo})",
        )
    )
    fig.update_layout(
        title=f"Demanda real mensual vs pronóstico mensual - Método usado: {metodo}",
        xaxis_title="Mes",
        yaxis_title="Unidades",
        hovermode="x unified",
    )
    return fig


def grafico_inventario(df_sim: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_sim["date"], y=df_sim["inventory_level"], name="Inventario", mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sim["date"],
            y=df_sim["reorder_point_s"],
            name="Punto s",
            mode="lines",
            line={"dash": "dot"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df_sim["date"], y=df_sim["demand_real"], name="Demanda mensual", opacity=0.35),
        secondary_y=True,
    )

    pedidos = df_sim[df_sim["order_placed"] > 0]
    fig.add_trace(
        go.Scatter(
            x=pedidos["date"],
            y=pedidos["order_placed"],
            name="Pedido generado",
            mode="markers",
            marker={"size": 10, "symbol": "triangle-up"},
        ),
        secondary_y=True,
    )

    fig.update_layout(title="Simulación mensual de inventario", hovermode="x unified")
    fig.update_yaxes(title_text="Inventario", secondary_y=False)
    fig.update_yaxes(title_text="Demanda / Pedidos", secondary_y=True)
    return fig


def grafico_tradeoff(df_opt: pd.DataFrame) -> go.Figure:
    mejor = df_opt.loc[df_opt["total_cost"].idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["total_cost"], mode="lines+markers", name="Costo total"))
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["holding_cost"], mode="lines", name="Costo mantener"))
    fig.add_trace(go.Scatter(x=df_opt["ss_months"], y=df_opt["stockout_cost"], mode="lines", name="Costo quiebre"))
    fig.add_vline(
        x=int(mejor["ss_months"]),
        line_dash="dash",
        annotation_text=f"Óptimo: {int(mejor['ss_months'])} meses",
    )
    fig.update_layout(
        title="Trade-off de costos",
        xaxis_title="Meses de stock de seguridad",
        yaxis_title="Costo",
        hovermode="x unified",
    )
    return fig


def formatear_comparacion(df_comparacion: pd.DataFrame) -> pd.DataFrame:
    df = df_comparacion.copy()
    df["wMAPE"] = df["wMAPE"].map(lambda x: f"{x:.2%}")
    df["Bias"] = df["Bias"].map(lambda x: f"{x:.2%}")
    df["MAE"] = df["MAE"].map(lambda x: f"{x:,.2f}")
    df["Resultado"] = np.where(df["Es mejor"], "✅ Mejor", "")
    return df[["Producto", "Método", "wMAPE", "Bias", "MAE", "Resultado"]]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("1. Carga de datos")
modo_datos = st.sidebar.radio("Modo de datos", ["Generar datos sintéticos", "Subir CSV/Excel"])

if modo_datos == "Generar datos sintéticos":
    n_productos = st.sidebar.slider("Número de productos", 1, 50, 5)
    meses = st.sidebar.slider("Meses de historial", 12, 84, 36)
    seed = st.sidebar.number_input("Semilla", min_value=1, max_value=9999, value=42)
    df_real = generar_demanda_sintetica(n_productos=n_productos, meses=meses, seed=seed)
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls"])
    if archivo is None:
        st.info(
            "Sube un CSV o Excel con columnas: date, product_id, demand_real. "
            "Si tus datos son diarios, la app los agrupará por mes."
        )
        st.stop()

    try:
        df_real = leer_archivo_subido(archivo)
    except Exception as e:
        st.error(str(e))
        st.stop()

st.sidebar.header("2. Pronóstico mensual")
modo_pronostico = st.sidebar.selectbox(
    "Selección del método",
    ["Automático: mejor método por producto", "Manual: elegir un método"],
)

df_forecast_auto, df_comparacion = generar_forecast_mejor_por_producto(df_real)

if modo_pronostico == "Manual: elegir un método":
    metodo_manual = st.sidebar.selectbox("Método manual", METODOS_PRONOSTICO)
    df_forecast = generar_forecast(df_real, metodo_manual)
else:
    metodo_manual = None
    df_forecast = df_forecast_auto

productos = sorted(df_forecast["product_id"].unique())
producto_sel = st.sidebar.selectbox("Producto a visualizar", productos)

sub_comparacion_producto = df_comparacion[df_comparacion["Producto"] == producto_sel].copy()
mejor_metodo_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "Método"].iloc[0]
mejor_wmape_producto = sub_comparacion_producto.loc[sub_comparacion_producto["Es mejor"], "wMAPE"].iloc[0]

if modo_pronostico == "Automático: mejor método por producto":
    st.sidebar.success(f"Método elegido para {producto_sel}: {mejor_metodo_producto}")
else:
    st.sidebar.info(f"Mejor método para {producto_sel}: {mejor_metodo_producto}")

st.sidebar.header("3. Política de inventario mensual")
politica = st.sidebar.selectbox(
    "Política",
    [
        "RS - revisión periódica",
        "sS - punto de reorden y nivel máximo",
        "sQ - punto de reorden y cantidad fija",
    ],
)

initial_stock = st.sidebar.number_input("Stock inicial", min_value=0, value=1000, step=100)
lead_time_months = st.sidebar.number_input("Lead time / tiempo de entrega (meses)", min_value=1, value=1, step=1)
review_period_months = st.sidebar.number_input("Periodo de revisión R (meses)", min_value=1, value=1, step=1)
ss_months = st.sidebar.number_input("Stock de seguridad inicial (meses)", min_value=0, value=1, step=1)
q_fixed = st.sidebar.number_input("Cantidad fija Q", min_value=1, value=1000, step=100)
lot_size = st.sidebar.number_input("Tamaño de lote / empaque", min_value=1, value=1, step=1)

st.sidebar.header("4. Costos")
cost_order = st.sidebar.number_input("Costo por orden", min_value=0.0, value=200.0, step=10.0)
cost_holding_month = st.sidebar.number_input("Costo mensual de mantener 1 unidad", min_value=0.0, value=1.5, step=0.5)
cost_stockout = st.sidebar.number_input("Costo por unidad perdida", min_value=0.0, value=500.0, step=10.0)
ss_max = st.sidebar.slider("Máximo SS para optimizar (meses)", 1, 24, 6)

parametros = ParametrosInventario(
    initial_stock=int(initial_stock),
    lead_time_months=int(lead_time_months),
    review_period_months=int(review_period_months),
    ss_months=int(ss_months),
    q_fixed=int(q_fixed),
    lot_size=int(lot_size),
    cost_order=float(cost_order),
    cost_holding_month=float(cost_holding_month),
    cost_stockout=float(cost_stockout),
)


# =========================================================
# CONTENIDO PRINCIPAL
# =========================================================
sub_forecast = df_forecast[df_forecast["product_id"] == producto_sel].copy()
metodo_usado = sub_forecast["method_used"].iloc[0]
sub_sim = simular_producto(sub_forecast, politica, parametros)
kpis = calcular_kpis(sub_sim, parametros)
sub_opt = optimizar_stock_seguridad(sub_forecast, politica, parametros, ss_max=ss_max)
mejor = sub_opt.loc[sub_opt["total_cost"].idxmin()]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método usado", metodo_usado)
col2.metric("Fill rate", f"{kpis['fill_rate']:.2%}")
col3.metric("Inventario promedio", f"{kpis['avg_inventory']:.1f}")
col4.metric("Ventas perdidas", f"{kpis['lost_sales_units']:.0f}")
col5.metric("Costo total", f"S/ {kpis['total_cost']:,.2f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Mejor método",
    "📊 Datos y pronóstico",
    "📦 Simulación",
    "🎯 Optimización",
    "📋 Tablas",
])

with tab1:
    st.subheader("Mejor método de pronóstico por producto")
    st.write(
        "La app compara Naive, Promedio móvil, SES, Regresión lineal, ARIMA, SARIMA, Holt-Winters y Croston para cada producto. "
        "El mejor método se elige por menor wMAPE. Si hay empate, se toma el Bias más cercano a cero y luego el MAE más bajo."
    )

    resumen_mejores = (
        df_comparacion[df_comparacion["Es mejor"]]
        .copy()
        .sort_values("Producto")
    )

    # IMPORTANTE:
    # df_comparacion ya trae una columna llamada "Mejor método".
    # Por eso NO debemos renombrar "Método" directamente a "Mejor método",
    # porque se crean columnas duplicadas y Streamlit/PyArrow muestra error.
    resumen_mejores = resumen_mejores[["Producto", "Método", "wMAPE", "Bias", "MAE"]].rename(
        columns={"Método": "Mejor método"}
    )

    resumen_mostrar = resumen_mejores.copy()
    resumen_mostrar["wMAPE"] = resumen_mostrar["wMAPE"].map(lambda x: f"{x:.2%}")
    resumen_mostrar["Bias"] = resumen_mostrar["Bias"].map(lambda x: f"{x:.2%}")
    resumen_mostrar["MAE"] = resumen_mostrar["MAE"].map(lambda x: f"{x:,.2f}")

    st.dataframe(resumen_mostrar, use_container_width=True, hide_index=True)

    fig_best = px.bar(
        resumen_mejores,
        x="Producto",
        y="wMAPE",
        color="Mejor método",
        text="Mejor método",
        title="Método ganador por producto según menor wMAPE",
        labels={"wMAPE": "wMAPE", "Producto": "Producto"},
    )
    fig_best.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_best, use_container_width=True)

    csv_mejores = resumen_mejores.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar mejores métodos en CSV",
        data=csv_mejores,
        file_name="mejor_metodo_por_producto.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Pronóstico mensual de demanda")
    st.write(
        "La demanda se trabaja por mes. Si cargaste datos diarios, el sistema los sumó automáticamente por producto y mes."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(grafico_forecast(sub_forecast), use_container_width=True)
    with col_b:
        st.write(f"Comparación de métodos para {producto_sel}")
        st.dataframe(formatear_comparacion(sub_comparacion_producto), use_container_width=True, hide_index=True)
        st.success(
            f"Mejor método para {producto_sel}: {mejor_metodo_producto} "
            f"con wMAPE {mejor_wmape_producto:.2%}."
        )

with tab3:
    st.subheader("Simulación mensual de inventario")
    st.plotly_chart(grafico_inventario(sub_sim), use_container_width=True)

    st.write("KPIs de la simulación")
    kpi_df = pd.DataFrame([kpis]).T.reset_index()
    kpi_df.columns = ["Indicador", "Valor"]
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Optimización de stock de seguridad mensual")
    st.info(
        f"Para el producto {producto_sel}, usando el método de pronóstico {metodo_usado}, "
        f"el stock de seguridad óptimo encontrado es {int(mejor['ss_months'])} meses, "
        f"con costo total aproximado de S/ {mejor['total_cost']:,.2f}."
    )
    st.plotly_chart(grafico_tradeoff(sub_opt), use_container_width=True)

    fig_servicio = px.line(
        sub_opt,
        x="ss_months",
        y="fill_rate",
        markers=True,
        title="Nivel de servicio según meses de stock de seguridad",
        labels={"ss_months": "Meses de stock de seguridad", "fill_rate": "Fill rate"},
    )
    fig_servicio.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_servicio, use_container_width=True)

with tab5:
    st.subheader("Tablas de resultados")
    st.write("Comparación completa de métodos")
    st.dataframe(formatear_comparacion(df_comparacion), use_container_width=True, hide_index=True)

    st.write("Datos mensuales base con pronóstico elegido")
    st.dataframe(sub_forecast, use_container_width=True, hide_index=True)

    st.write("Simulación mensual")
    st.dataframe(sub_sim, use_container_width=True, hide_index=True)

    st.write("Resultados de optimización")
    st.dataframe(sub_opt, use_container_width=True, hide_index=True)

    csv = sub_sim.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar simulación mensual en CSV",
        data=csv,
        file_name=f"simulacion_mensual_{producto_sel}.csv",
        mime="text/csv",
    )

    csv_comparacion = df_comparacion.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar comparación de métodos en CSV",
        data=csv_comparacion,
        file_name="comparacion_metodos_pronostico.csv",
        mime="text/csv",
    )
