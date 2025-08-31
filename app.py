\
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
from pathlib import Path
from utils import compute_viscosity, cavitation_risk_proxy, risk_score, classify_status

APP_TITLE = "Circulix – Cold‑Aware Pump Health (One‑App)"
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
for d in [DATA_DIR, MODEL_DIR, Path("uploads")]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Auth (simple local) ---
def load_users():
    f = Path("users.json")
    if not f.exists():
        f.write_text(json.dumps({"admin":"admin"}))
    return json.loads(f.read_text())

if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    st.sidebar.subheader("Sign in")
    u = st.sidebar.text_input("Username", value="admin")
    p = st.sidebar.text_input("Password", type="password", value="admin")
    if st.sidebar.button("Login"):
        users = load_users()
        if users.get(u) == p:
            st.session_state["user"] = u
            st.success("Logged in.")
        else:
            st.error("Invalid credentials.")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data", "Train", "Predict (Batch)", "Live Monitor (Demo)", "Account"])

def load_model():
    choices = [str(p) for p in MODEL_DIR.glob("*.joblib")]
    sel = st.sidebar.selectbox("Model file", options=["(none)"]+choices, index=0)
    if sel != "(none)":
        try:
            model = joblib.load(sel)
            return model, sel
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            return None, None
    return None, None

model, model_path = load_model()

if page == "Dashboard":
    st.title(APP_TITLE)
    cols = st.columns(3)
    with cols[0]: st.metric("User", st.session_state["user"])
    with cols[1]: st.metric("Data files", len(list(DATA_DIR.glob("*.csv"))))
    with cols[2]: st.metric("Model loaded", "Yes" if model else "No")
    st.markdown("---")
    try:
        latest = max(DATA_DIR.glob("*.csv"), key=os.path.getmtime)
        st.write(f"Latest data: `{latest.name}`")
        df = pd.read_csv(latest)
        st.dataframe(df.head(10))
    except ValueError:
        st.info("No data yet. Use Data page to upload CSV.")

elif page == "Data":
    st.header("Data")
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            (DATA_DIR / uf.name).write_bytes(uf.read())
        st.success(f"Saved {len(uploaded)} file(s).")
    files = list(DATA_DIR.glob("*.csv"))
    if files:
        sel = st.selectbox("Preview file", [f.name for f in files])
        df = pd.read_csv(DATA_DIR/sel)
        st.write(df.head(20))
        st.write("Columns:", list(df.columns))
        st.write("Rows:", len(df))
        st.write(df.describe(include="all").T)

elif page == "Train":
    st.header("Train a Model (Real + Synthetic Cold)")
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("Upload data first.")
    else:
        use = st.multiselect("Training files", [f.name for f in files], default=[f.name for f in files[:1]])
        algo = st.selectbox("Algorithm", ["RandomForest", "SVM (linear)"])
        ntrees = st.slider("RF: n_estimators", 50, 400, 200, 50)
        test_size = st.slider("Test size (%)", 10, 40, 20, 5)
        if st.button("Train"):
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import LinearSVC
            from sklearn.metrics import classification_report, confusion_matrix

            df_list = [pd.read_csv(DATA_DIR/n) for n in use]
            df = pd.concat(df_list, ignore_index=True)

            if "Label" not in df.columns:
                st.error("Need 'Label' column (Normal, Bearing, Misalignment, Unbalance, Cavitation, Seal_Leak, Unknown_Fault).")
                st.stop()

            if "Ambient_T" in df.columns and "Viscosity_mPa_s" not in df.columns:
                df["Viscosity_mPa_s"] = df["Ambient_T"].apply(lambda t: compute_viscosity(t) if pd.notna(t) else np.nan)
            if "Pressure_Fluct_bar" in df.columns and "Cavitation_Proxy" not in df.columns:
                df["Cavitation_Proxy"] = df.apply(cavitation_risk_proxy, axis=1)

            numeric_cols = [c for c in df.columns if c not in ("Label","timestamp") and pd.api.types.is_numeric_dtype(df[c])]
            X = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
            y = df["Label"].astype(str)

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)

            if algo == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("rf", RandomForestClassifier(n_estimators=ntrees, random_state=42))])
            else:
                from sklearn.svm import LinearSVC
                pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("svm", LinearSVC(random_state=42))])

            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)
            rep = classification_report(yte, ypred, digits=3)
            cm = confusion_matrix(yte, ypred, labels=sorted(y.unique()))

            st.code(rep, language="text")
            st.write(pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())))

            MODEL_DIR.mkdir(exist_ok=True, parents=True)
            outpath = MODEL_DIR / f"circulix_model_{algo.lower()}_{ntrees if algo=='RandomForest' else 'svm'}.joblib"
            joblib.dump(pipe, outpath)
            st.success(f"Model saved to {outpath}")

elif page == "Predict (Batch)":
    st.header("Batch Prediction")
    f = st.file_uploader("Upload CSV for prediction", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        if "Ambient_T" in df.columns and "Viscosity_mPa_s" not in df.columns:
            df["Viscosity_mPa_s"] = df["Ambient_T"].apply(lambda t: compute_viscosity(t) if pd.notna(t) else np.nan)
        if "Pressure_Fluct_bar" in df.columns and "Cavitation_Proxy" not in df.columns:
            df["Cavitation_Proxy"] = df.apply(cavitation_risk_proxy, axis=1)

        drop_cols = [c for c in ("Label","timestamp") if c in df.columns]
        X = df.drop(columns=drop_cols).copy()
        X = X.fillna(X.median(numeric_only=True))

        model, _ = load_model()
        proba_normal = None
        pred_label = [""]*len(df)
        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    cls = model.classes_.tolist()
                    pred_label = model.predict(X)
                    proba_normal = probs[:, cls.index("Normal")].tolist() if "Normal" in cls else [0.0]*len(df)
                else:
                    pred_label = model.predict(X)
                    proba_normal = [0.0]*len(df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred_label = [""]*len(df)
                proba_normal = [0.0]*len(df)
        else:
            proba_normal = [0.0]*len(df)

        scores, status = [], []
        for i in range(len(df)):
            s = risk_score(df.iloc[i], proba_normal[i] if proba_normal else None)
            scores.append(s)
            status.append(classify_status(s, str(pred_label[i])))

        df_out = df.copy()
        df_out["Predicted_Label"] = pred_label
        df_out["Normal_Prob"] = proba_normal
        df_out["Risk_Score_0_1"] = scores
        df_out["Health_Status"] = status

        st.dataframe(df_out.head(50))
        st.download_button("Download predictions CSV", df_out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

elif page == "Live Monitor (Demo)":
    st.header("Live Monitor (Demo)")
    files = list(DATA_DIR.glob("*.csv"))
    if files:
        sel = st.selectbox("Select data file", [f.name for f in files])
        df = pd.read_csv(DATA_DIR/sel)
        st.write("Last 200 rows")
        st.dataframe(df.tail(200))
    else:
        st.info("No data. Upload in Data page.")

elif page == "Account":
    st.header("Account")
    users = json.loads(Path("users.json").read_text()) if Path("users.json").exists() else {"admin":"admin"}
    cur = st.text_input("Current password", type="password")
    new = st.text_input("New password", type="password")
    if st.button("Change password"):
        u = st.session_state["user"]
        if users.get(u) == cur:
            users[u] = new
            Path("users.json").write_text(json.dumps(users))
            st.success("Updated.")
        else:
            st.error("Wrong current password.")
