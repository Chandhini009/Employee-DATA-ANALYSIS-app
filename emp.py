# emp_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page config & Dark Gradient
# ---------------------------
st.set_page_config(page_title="Dyashin HR Insights (Final)", layout="wide")

dark_css = """
<style>
/* Gradient background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #071428 0%, #06384a 50%, #064f4f 100%);
  color: #eaf6ff;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: rgba(4,8,20,0.85);
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Main card */
div.block-container {
  background: rgba(6,12,26,0.6);
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 8px 30px rgba(3, 169, 244, 0.06);
}

/* Headings */
h1, h2, h3, h4 {
  color: #7ce7ff !important;
  font-weight: 700 !important;
  text-shadow: 0 0 10px rgba(124,231,255,0.08);
}
h1 { font-size: 2.6rem !important; }
h2 { font-size: 1.8rem !important; }

/* Text */
p, label, span, div, li {
  color: #d6eefc !important;
}

/* Buttons */
div.stButton > button {
  background: linear-gradient(90deg,#00d0ff,#3aa0ff);
  color: #021220;
  border-radius: 10px;
  font-weight: 700;
  border: none;
}
div.stButton > button:hover {
  transform: scale(1.02);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 14px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(10,25,40,0.55);
  color: #cfeeff;
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 600;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div style="text-align:center; padding:18px; border-radius:10px; margin-bottom:10px;">
        <h1>Dyashin HR Insights</h1>
        <p style="font-size:1rem; color:#cfeeff">Upload your Excel HR dataset and explore workforce insights. Attrition model includes preprocessing, SMOTE balancing, scaling, and a RandomForest classifier.</p>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------------
# Sidebar - File upload & options
# ---------------------------
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload Employee Excel (.xlsx)", type=["xlsx"])
model_train_button = st.sidebar.checkbox("Retrain model on upload (recommended)", value=True)
show_confusion = st.sidebar.checkbox("Show confusion matrix", value=True)
show_feature_imp = st.sidebar.checkbox("Show feature importance", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Needed columns (recommended): `EmployeeID`, `Name`, `JoiningDate`, `Experience(Years)`, `Salary`, `Department`, `Designation`, `Location`, `Gender`, `Attrition`")

# Helper: safe numeric conversion
def safe_to_numeric(series, fill=None):
    try:
        out = pd.to_numeric(series)
    except:
        out = pd.to_numeric(series.astype(str).str.extract(r'(-?\d+\.?\d*)')[0], errors='coerce')
    if fill is not None:
        out = out.fillna(fill)
    return out

# ---------------------------
# Main flow
# ---------------------------
if uploaded_file is None:
    st.info("Upload an Excel file (.xlsx) using the sidebar to begin analysis.")
    st.stop()

# Load selected sheet
try:
    xls = pd.ExcelFile(uploaded_file)
    sheet = xls.sheet_names[0] if len(xls.sheet_names) == 1 else st.sidebar.selectbox("Select sheet", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Error reading uploaded file: {e}")
    st.stop()

# Basic cleaning & normalizing
df.columns = df.columns.str.strip()
df_original = df.copy()

# Ensure JoiningDate parsed
for col in df.select_dtypes(include=['object']).columns:
    if col.lower().startswith("joining") or "date" in col.lower() and col.lower().startswith("join"):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass

# If any datetime columns exist, compute tenure (Years) and drop raw datetime columns
datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
if "JoiningDate" in df.columns and "JoiningDate" in datetime_cols:
    df["Tenure_Years"] = (pd.Timestamp.now() - df["JoiningDate"]).dt.days / 365
# General: if other datetime columns present, drop them after extracting tenure-like info if possible
for dcol in datetime_cols:
    if dcol not in ["JoiningDate"]:
        # try to derive a delta from joining if exists, else drop
        if "JoiningDate" in df.columns:
            # nothing additional for now
            pass
# drop datetime raw columns to avoid dtype mixes (we already saved Tenure_Years)
for dcol in datetime_cols:
    df = df.drop(columns=[dcol])

# Fill common missing values
if "Salary" in df.columns:
    df["Salary"] = safe_to_numeric(df["Salary"], fill=df["Salary"].median() if df["Salary"].dtype != object else 0)
else:
    df["Salary"] = 0.0

if "Experience(Years)" in df.columns:
    df["Experience(Years)"] = safe_to_numeric(df["Experience(Years)"], fill=df["Experience(Years)"].median())
elif "Tenure_Years" in df.columns:
    df["Experience(Years)"] = safe_to_numeric(df["Tenure_Years"], fill=0.0)
else:
    df["Experience(Years)"] = 0.0

for col in ["Department", "Designation", "Location", "Gender"]:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("Unknown")
    else:
        df[col] = "Unknown"

# Normalize Attrition if exists
has_attr = "Attrition" in df.columns
if has_attr:
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.lower()
    df["Attrition"] = df["Attrition"].replace({'yes': 1, 'y': 1, 'true': 1, '1': 1, 'no': 0, 'n': 0, 'false': 0, '0': 0})
    # if still not numeric, try to coerce
    df["Attrition"] = pd.to_numeric(df["Attrition"], errors='coerce').fillna(0).astype(int)

# Show quick preview & summary
st.subheader("Dataset preview")
st.dataframe(df.head())

st.subheader("Dataset summary")
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
with col2:
    if has_attr:
        st.write("Attrition distribution:")
        st.bar_chart(df["Attrition"].value_counts())
with col3:
    st.write("Numeric summary:")
    st.write(df.select_dtypes(include=[np.number]).describe().T[['mean','50%','std']])

# Tabs for Dashboard / Salary / Attrition
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💰 Salary Analysis", "🔮 Attrition Prediction"])

# ---------------------------
# Dashboard tab
# ---------------------------
with tab1:
    st.header("Workforce Overview")
    c1, c2 = st.columns(2)
    if "Gender" in df.columns:
        gc = df["Gender"].value_counts()
        fig = px.pie(values=gc.values, names=gc.index, title="Gender Distribution")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        c1.plotly_chart(fig, use_container_width=True)
    if "Department" in df.columns:
        dc = df["Department"].value_counts()
        fig2 = px.bar(x=dc.index, y=dc.values, title="Department Size", labels={'x':'Department','y':'Employees'})
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        c2.plotly_chart(fig2, use_container_width=True)
    if "JoiningDate" not in df_original.columns and "Tenure_Years" in df.columns:
        st.write("Tenure distribution (years)")
        st.bar_chart(pd.cut(df["Tenure_Years"].fillna(0), bins=10).value_counts().sort_index())

# ---------------------------
# Salary tab
# ---------------------------
with tab2:
    st.header("Salary Analysis")
    if "Salary" in df.columns:
        c1, c2 = st.columns(2)
        dept_sal = df.groupby("Department")["Salary"].mean().sort_values()
        fig = px.bar(x=dept_sal.index, y=dept_sal.values, title="Avg Salary by Department")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        c1.plotly_chart(fig, use_container_width=True)

        loc_sal = df.groupby("Location")["Salary"].mean().sort_values()
        fig2 = px.bar(x=loc_sal.index, y=loc_sal.values, title="Avg Salary by Location")
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        c2.plotly_chart(fig2, use_container_width=True)

    if "Designation" in df.columns:
        st.subheader("Salary distribution by Designation")
        fig3 = px.box(df, x="Designation", y="Salary", color="Department", title="Salary by Designation & Department")
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Attrition tab
# ---------------------------
with tab3:
    st.header("Attrition Prediction (Enhanced Model)")

    if not has_attr:
        st.warning("No 'Attrition' column found in dataset; training/skipping prediction.")
        st.stop()

    # Build modeling DF: choose candidate features (numeric + categorical)
    model_df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ["Salary", "Experience(Years)"]
    for nc in numeric_cols:
        if nc not in model_df.columns:
            model_df[nc] = 0.0
        else:
            model_df[nc] = safe_to_numeric(model_df[nc], fill=model_df[nc].median())

    # Include Tenure_Years if exists
    if "Tenure_Years" in model_df.columns:
        model_df["Tenure_Years"] = safe_to_numeric(model_df["Tenure_Years"], fill=0.0)
        numeric_cols.append("Tenure_Years")

    # Choose categorical columns to include
    categorical_cols = []
    for col in ["Department", "Designation", "Location", "Gender", "PerformanceRating"]:
        if col in model_df.columns:
            # convert to string for get_dummies
            model_df[col] = model_df[col].astype(str).fillna("Unknown")
            categorical_cols.append(col)

    # Drop irrelevant columns
    drop_candidates = [c for c in ["EmployeeID","Name","JoiningDate","Tenure_Years"] if c in model_df.columns and c not in numeric_cols]
    # We'll keep Tenure_Years if in numeric_cols; else drop
    model_df = model_df.drop(columns=[c for c in ["EmployeeID","Name"] if c in model_df.columns], errors='ignore')

    # Create dummies, but keep columns consistent after split
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

    # Final features
    feature_cols = [c for c in model_df.columns if c not in ["Attrition"]]
    X = model_df[feature_cols].fillna(0)
    y = model_df["Attrition"].astype(int)

    # Ensure X is purely numeric (it should be after get_dummies)
    # Scale numeric features
    scaler = StandardScaler()
    try:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        st.stop()

    # Balance with SMOTE if imbalance
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X_scaled, y)
    except Exception as e:
        st.warning("SMOTE failed (possibly very small dataset). Proceeding without SMOTE.")
        X_res, y_res = X_scaled, y

    # Train/test split and RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained. Test accuracy: {acc*100:.2f}%")

    # Optional confusion matrix
    if show_confusion:
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix (Test set)")
        cm_df = pd.DataFrame(cm, index=["Actual_No","Actual_Yes"], columns=["Pred_No","Pred_Yes"])
        st.table(cm_df)

    # Feature importance
    if show_feature_imp:
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
        st.subheader("Top feature importances")
        st.bar_chart(fi)

    # ---------------------------
    # Prediction UI: Build input aligned to model's columns
    # ---------------------------
    st.markdown("---")
    st.subheader("Predict new employee attrition")
    with st.form("predict_form"):
        # Basic numeric inputs
        in_salary = st.number_input("Salary", value=float(df["Salary"].median()), step=1000.0)
        in_exp = st.number_input("Experience (Years)", value=float(df["Experience(Years)"].median()), step=0.1)
        in_tenure = None
        if "Tenure_Years" in X.columns or "Tenure_Years" in df.columns:
            in_tenure = st.number_input("Tenure (Years)", value=float(df.get("Tenure_Years", df["Experience(Years)"]).median()), step=0.1)

        # categorical selections based on original df values
        sel_vals = {}
        for col in ["Department", "Designation", "Location", "Gender"]:
            if col in df_original.columns:
                sel = st.selectbox(f"{col}", options=sorted(df_original[col].astype(str).unique()))
                sel_vals[col] = str(sel)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build blank input vector aligned with X.columns
        input_vec = pd.Series(0, index=X.columns, dtype=float)

        # Set numeric features (after scaling)
        raw_input = {}
        raw_input["Salary"] = in_salary
        raw_input["Experience(Years)"] = in_exp
        if in_tenure is not None:
            raw_input["Tenure_Years"] = in_tenure

        # Put raw numeric into a temp DataFrame to scale like training
        temp_df = pd.DataFrame([raw_input])
        # Ensure all numeric columns exist in temp_df
        for n in [c for c in X.columns if c in numeric_cols]:
            if n not in temp_df.columns:
                temp_df[n] = 0.0
        # For any other numeric columns in model (rare), fill zeros
        numeric_model_cols = [c for c in X.columns if c in numeric_cols]
        try:
            scaled_temp = scaler.transform(temp_df.reindex(columns=X.columns, fill_value=0.0).fillna(0))
            # scaled_temp is 2D aligned to X.columns
            scaled_temp_ser = pd.Series(scaled_temp[0], index=X.columns)
        except Exception:
            # fallback: scale only known numeric columns; set categorical zeros
            scaled_temp_ser = pd.Series(0.0, index=X.columns)
            for n in numeric_model_cols:
                # scale manually using mean/std from X (scaler)
                try:
                    idx = list(X.columns).index(n)
                    mean = scaler.mean_[idx]
                    scale = scaler.scale_[idx]
                    scaled_temp_ser[n] = (raw_input.get(n, 0.0) - mean) / (scale if scale != 0 else 1.0)
                except Exception:
                    scaled_temp_ser[n] = 0.0

        # Now fill in categorical dummies for selected values
        for col, val in sel_vals.items():
            # Dummy column name patterns used by pd.get_dummies with drop_first=True
            # There are two possibilities: "col_val" or "col_val" with spaces/special chars replaced.
            candidate = f"{col}_{val}"
            if candidate in X.columns:
                scaled_temp_ser[candidate] = 1.0
            else:
                # try sanitized candidate (replace spaces, '/', '-', etc.)
                sanitized = candidate.replace(" ", "_").replace("/", "_").replace("-", "_")
                if sanitized in X.columns:
                    scaled_temp_ser[sanitized] = 1.0
                else:
                    # try any column that starts with col_ and contains part of val
                    for c in X.columns:
                        if c.startswith(f"{col}_") and val.lower() in c.lower():
                            scaled_temp_ser[c] = 1.0

        # For any remaining columns not set, keep the scaled_temp_ser value (mostly zeros)
        input_array = scaled_temp_ser.values.reshape(1, -1)

        # Predict using trained RF
        pred = rf.predict(input_array)[0]
        prob = rf.predict_proba(input_array)[0][1] if hasattr(rf, "predict_proba") else None

        if pred == 1:
            st.error(f"🚨 Predicted: Likely to Leave  — Confidence {prob:.2%}" if prob is not None else "🚨 Predicted: Likely to Leave")
        else:
            st.success(f"✅ Predicted: Likely to Stay  — Confidence {(1-prob):.2%}" if prob is not None else "✅ Predicted: Likely to Stay")

    st.markdown("---")
    st.info("Notes: The model uses scaled numeric features, one-hot encoded categorical features, and SMOTE to reduce class imbalance. Prediction inputs are aligned to the trained model's feature set to prevent dtype/shape mismatch errors.")
