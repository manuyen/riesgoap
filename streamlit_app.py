# streamlit_app.py
import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =========================
# Configuración básica
# =========================
ART_DIR = os.environ.get("ART_DIR", os.path.join("artefactos", "v1"))

# Debe ser la PRIMERA llamada a Streamlit
st.set_page_config(
    page_title="Riesgo Aportantes",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =========================
# Carga de artefactos
# =========================
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ART_DIR, "input_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(os.path.join(ART_DIR, "decision_policy.json"), "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner = policy["winner"]
    pipe   = joblib.load(os.path.join(ART_DIR, f"pipeline_{winner}.joblib"))

    # compat: esquema nuevo (columns+dtypes) o antiguo (col->dtype)
    if "columns" in schema and "dtypes" in schema:
        features, dtypes = schema["columns"], schema["dtypes"]
    else:
        features, dtypes = list(schema.keys()), schema

    rev_label = {v: k for k, v in label_map.items()}
    thr = float(policy.get("threshold", 0.5))
    return pipe, features, dtypes, rev_label, thr, winner

PIPE, FEATURES, DTYPES, REV_LABEL, DEFAULT_THR, WINNER = load_artifacts()

# =========================
# Preprocesamiento (idéntico al entrenamiento)
# =========================
def agrupar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa SOLO las columnas que fueron agrupadas en entrenamiento.
       tip_particular NO se agrupa."""
    df = df.copy()
    if "estado_civil" in df:
        df["estado_civil"] = df["estado_civil"].apply(
            lambda x: x if x in ["Soltero", "Casado"] else "OTROS"
        )
    if "pais_nacimiento" in df:
        df["pais_nacimiento"] = df["pais_nacimiento"].apply(
            lambda x: x if x in ["PERU", "VENEZUELA"] else "OTROS"
        )
    if "universidad_pais" in df:
        df["universidad_pais"] = df["universidad_pais"].apply(
            lambda x: x if x in ["PERU", "VENEZUELA"] else "OTROS"
        )
    # tip_particular: sin agrupación

    if "cod_grupo_entidad_pagadora" in df:
        df["cod_grupo_entidad_pagadora"] = df["cod_grupo_entidad_pagadora"].apply(
            lambda x: x if x in ["CONSEJOS CEP", "MINSA"] else "OTROS"
        )
    if "nombre_local_pago" in df:
        pequeños = ["TUMBES", "TACNA", "MOQUEGUA", "PASCO", "MADRE DE DIOS"]
        df["nombre_local_pago"] = df["nombre_local_pago"].apply(
            lambda x: "OTROS" if x in pequeños else x
        )
    if "banco" in df:
        df["banco"] = df["banco"].apply(
            lambda x: x if x in ["BBVA", "CAJA", "SCOTIA BANK"] else "OTROS"
        )
    if "cod_forma_de_pago" in df:
        df["cod_forma_de_pago"] = df["cod_forma_de_pago"].apply(
            lambda x: x if x in ["VOUCHER", "EFECTIVO"] else "OTROS"
        )
    return df

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False, "si": True, "sí": True, "no": False}

def build_csv_template() -> pd.DataFrame:
    """Plantilla con todas las columnas esperadas y una fila de ejemplo."""
    row = {c: "" for c in FEATURES}
    defaults = {
        "edad_colegiacion": 24,
        "n_especialista": 2,
        "descripcion_sexo": "Femenino",
        "estado_civil": "Soltero",
        "pais_nacimiento": "PERU",
        "universidad_pais": "PERU",
        "tip_particular": "PRIVADA",  # sin agrupar
        "cod_grupo_entidad_pagadora": "OTROS",
        "nombre_local_pago": "LIMA",
        "banco": "BBVA",
        "cod_forma_de_pago": "VOUCHER",
    }
    for k, v in defaults.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row], columns=FEATURES)

def coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # columnas faltantes -> NaN
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan

    # Agrupar categorías ANTES de tipar
    df = agrupar_categorias(df)

    # Tipos
    for c in FEATURES:
        t = str(DTYPES.get(c, "object")).lower()
        if t.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t in ("bool", "boolean"):
            s = df[c].astype("string").str.strip().str.lower()
            df[c] = s.map(_BOOL_MAP)
            num = pd.to_numeric(s, errors="coerce")
            df.loc[num == 1, c] = True
            df.loc[num == 0, c] = False
            df[c] = df[c].astype("boolean").astype(bool)
        else:
            df[c] = df[c].astype("string")

    return df[FEATURES]

def predict_batch(records, thr: float):
    if isinstance(records, dict):
        records = [records]
    df = coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]
    yhat  = (proba >= thr).astype(int)
    return [
        {"proba_ALTO": float(p), "pred_num": int(y), "pred_label": REV_LABEL[int(y)], "threshold": thr}
        for p, y in zip(proba, yhat)
    ], df

# =========================
# UI
# =========================
st.title("📈 Riesgo Aportantes — Streamlit")
st.caption(f"Modelo: **{WINNER}** | artefactos: `{ART_DIR}`")

# Sidebar
thr = st.sidebar.slider(
    "Umbral de decisión (ALTO si proba ≥ umbral)",
    0.0, 1.0, float(DEFAULT_THR), 0.01
)

with st.sidebar.expander("Plantilla CSV", expanded=True):
    st.caption("Descarga la plantilla, duplica/edita filas y súbela en la sección de abajo.")
    plantilla_df = build_csv_template()
    st.download_button(
        label="⬇️ Descargar plantilla",
        data=plantilla_df.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_prediccion.csv",
        mime="text/csv",
        key="dl_template"
    )

with st.sidebar.expander("Artefactos"):
    if st.button("🔁 Recargar artefactos"):
        load_artifacts.clear()
        st.rerun()

# Predicción individual
st.subheader("Predicción individual")

rec = {}
for col in ["edad_colegiacion", "n_especialista"]:
    if col in FEATURES:
        rec[col] = st.number_input(col, value=24 if col == "edad_colegiacion" else 2, step=1)

for col, opts, default in [
    ("descripcion_sexo", ["Femenino", "Masculino", "OTROS"], "Femenino"),
    ("estado_civil", ["Soltero", "Casado", "OTROS"], "Soltero"),
    ("pais_nacimiento", ["PERU", "VENEZUELA", "OTROS"], "PERU"),
    ("universidad_pais", ["PERU", "VENEZUELA", "OTROS"], "PERU"),
    ("tip_particular", ["PRIVADA", "PUBLICA", "SIN DATO"], "PRIVADA"),  # <- ahora sí en el formulario
    ("cod_grupo_entidad_pagadora", ["CONSEJOS CEP", "MINSA", "OTROS"], "OTROS"),
    ("nombre_local_pago", ["LIMA","TUMBES","TACNA","MOQUEGUA","PASCO","MADRE DE DIOS","OTROS"], "LIMA"),
    ("banco", ["BBVA", "CAJA", "SCOTIA BANK", "OTROS"], "BBVA"),
    ("cod_forma_de_pago", ["VOUCHER", "EFECTIVO", "OTROS"], "VOUCHER"),
]:
    if col in FEATURES:
        rec[col] = st.selectbox(col, opts, index=opts.index(default))

if st.button("Predecir caso", use_container_width=True):
    out, _ = predict_batch(rec, thr)
    st.success(f"Predicción: **{out[0]['pred_label']}** | proba_ALTO={out[0]['proba_ALTO']:.3f} | umbral={out[0]['threshold']:.2f}")
    st.json(out[0])

# Batch por CSV
st.markdown("---")
st.subheader("Predicción por CSV (batch)")

up = st.file_uploader("Sube un CSV con columnas similares a las de entrenamiento", type=["csv"])

# preservar resultados entre reruns
if "batch_res" not in st.session_state:
    st.session_state["batch_res"] = None

if up is not None:
    try:
        raw_df = pd.read_csv(up)
        st.info(f"Archivo: **{up.name}** | Filas: {len(raw_df)} | Columnas: {len(raw_df.columns)}")

        missing = [c for c in FEATURES if c not in raw_df.columns]
        extra   = [c for c in raw_df.columns if c not in FEATURES]
        if missing:
            st.warning(f"Faltan columnas requeridas: {missing}")
        if extra:
            st.caption(f"Columnas no utilizadas (se ignorarán/alinearán): {extra}")

        if st.button("🔮 Predecir CSV", use_container_width=True, key="btn_predict_csv"):
            out, clean_df = predict_batch(raw_df.to_dict(orient="records"), thr)
            res = clean_df.copy()
            res["proba_ALTO"] = [o["proba_ALTO"] for o in out]
            res["pred_num"]   = [o["pred_num"] for o in out]
            res["pred_label"] = [o["pred_label"] for o in out]
            st.session_state["batch_res"] = res
            st.success("Predicciones generadas.")

    except Exception as e:
        st.error(f"Error al procesar el CSV: {e}")

if st.session_state["batch_res"] is not None:
    res = st.session_state["batch_res"]
    st.dataframe(res.head(50), use_container_width=True)
    st.download_button(
        label="⬇️ Descargar predicciones",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_streamlit.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_predictions"
    )
