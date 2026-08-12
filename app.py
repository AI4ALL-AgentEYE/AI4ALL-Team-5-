import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.set_page_config(
    page_title="AgentEYE",
    page_icon="🔐",
    layout="wide"
)

RANDOM_STATE = 42
CSV_PATH = "data/agent_security_risk_scores.csv"
TARGET = "action_risk_score"
LEAKY = ["human_approval_required", "access_decision"]

cat_cols = [
    "agent_role",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type"
]

df = pd.read_csv(CSV_PATH)

st.title("AgentEYE")
st.subheader("AI Agent Security Risk Analysis")

st.write(
    "Can we predict the security risk score of an AI agent's action "
    "based on its autonomy level, permissions, requested tools, and "
    "resource access patterns?"
)

st.write(
    "Can we decide whether an agent should be granted access based on that score?"
)

st.divider()

with st.expander("Dataset", expanded=True):

    st.write(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.write("Descriptive statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

with st.expander("Missing Values"):

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) == 0:
        st.success("No missing values.")
    else:
        st.dataframe(
            missing_values.to_frame("Missing Values"),
            use_container_width=True
        )

with st.expander("Column Distributions"):

    selected_column = st.selectbox(
        "Select a column",
        df.columns
    )

    value_counts = (
        df[selected_column]
        .value_counts(dropna=False)
        .head(30)
        .reset_index()
    )

    value_counts.columns = [
        selected_column,
        "Count"
    ]

    fig = px.bar(
        value_counts,
        x=selected_column,
        y="Count",
        title=f"Distribution of {selected_column}"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Access Decision by Agent Role", expanded=True):

    fig = px.histogram(
        df,
        x="agent_role",
        color="access_decision",
        barmode="group",
        title="Access Decision Count by Agent Role"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Agent Role Distribution"):

    counts = (
        df["agent_role"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Agent Role",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Agent Role",
        y="Count",
        title="Distribution of Agent Roles"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Access Decision Distribution"):

    counts = (
        df["access_decision"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Access Decision",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Access Decision",
        y="Count",
        title="Distribution of Access Decisions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("User Role Distribution"):

    counts = (
        df["user_role"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "User Role",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="User Role",
        y="Count",
        title="Distribution of User Roles"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Access Decision Proportions by Agent Role"):

    table = pd.crosstab(
        df["agent_role"],
        df["access_decision"],
        normalize="index"
    )

    fig = px.bar(
        table.reset_index(),
        x="agent_role",
        y=table.columns.tolist(),
        title="Access Decision Proportions by Agent Role",
        barmode="stack"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Access Decision Proportions by User Role"):

    table = pd.crosstab(
        df["user_role"],
        df["access_decision"],
        normalize="index"
    )

    fig = px.bar(
        table.reset_index(),
        x="user_role",
        y=table.columns.tolist(),
        title="Access Decision Proportions by User Role",
        barmode="stack"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Average Risk Score by Access Decision"):

    risk_by_decision = (
        df.groupby("access_decision")[TARGET]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        risk_by_decision,
        x="access_decision",
        y=TARGET,
        title="Average Risk Score by Access Decision",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("User Role × Requested Action"):

    table = pd.crosstab(
        df["user_role"],
        df["requested_action"]
    )

    fig = px.imshow(
        table,
        text_auto=True,
        aspect="auto",
        title="Which User Roles Perform Which Actions?"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Previous Failed Attempts"):

    failed = (
        df.groupby("access_decision")[
            "previous_failed_attempts"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        failed,
        x="access_decision",
        y="previous_failed_attempts",
        title="Average Previous Failed Attempts by Access Decision",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Agent Autonomy Level"):

    autonomy = (
        df.groupby("access_decision")[
            "agent_autonomy_level"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        autonomy,
        x="access_decision",
        y="agent_autonomy_level",
        title="Average Agent Autonomy Level by Access Decision",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Action Risk Score Distribution", expanded=True):

    fig = px.histogram(
        df,
        x=TARGET,
        nbins=30,
        marginal="box",
        title="Action Risk Score Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Data Exfiltration Risk"):

    fig = px.histogram(
        df,
        x="data_exfiltration_risk",
        nbins=20,
        marginal="box",
        title="Distribution of Data Exfiltration Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Prompt Injection"):

    prompt = (
        df.groupby("prompt_injection_detected")[
            TARGET
        ]
        .mean()
        .reset_index()
    )

    prompt["Prompt Injection"] = prompt[
        "prompt_injection_detected"
    ].map({
        0: "Not Detected",
        1: "Detected"
    })

    fig = px.bar(
        prompt,
        x="Prompt Injection",
        y=TARGET,
        title="Average Risk Score by Prompt Injection",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ct = pd.crosstab(
        df["prompt_injection_detected"],
        df["access_decision"],
        normalize="index"
    )

    ct.index = [
        "Not Detected" if x == 0 else "Detected"
        for x in ct.index
    ]

    fig = px.bar(
        ct.reset_index(),
        x="prompt_injection_detected",
        y=ct.columns.tolist(),
        title="Access Decision Rate by Prompt Injection Detection",
        barmode="stack"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    injection_not_blocked = df[
        (df["prompt_injection_detected"] == 1)
        &
        (df["access_decision"] != "Blocked")
    ]

    st.write("Prompt injection detected but not blocked")

    st.dataframe(
        injection_not_blocked,
        use_container_width=True
    )

with st.expander("Audit Log Availability"):

    audit = (
        df.groupby("audit_log_available")[
            TARGET
        ]
        .mean()
        .reset_index()
    )

    audit["Audit Log"] = audit[
        "audit_log_available"
    ].map({
        0: "Unavailable",
        1: "Available"
    })

    fig = px.bar(
        audit,
        x="Audit Log",
        y=TARGET,
        title="Average Risk Score by Audit Log Availability",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    ct = pd.crosstab(
        df["audit_log_available"],
        df["access_decision"],
        normalize="index"
    )

    ct.index = [
        "Unavailable" if x == 0 else "Available"
        for x in ct.index
    ]

    fig = px.bar(
        ct.reset_index(),
        x="audit_log_available",
        y=ct.columns.tolist(),
        title="Access Decision Rate by Audit Log Availability",
        barmode="stack"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Numeric Correlations"):

    corr = (
        df.select_dtypes("number")
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values()
    )

    corr_df = corr.reset_index()
    corr_df.columns = [
        "Feature",
        "Correlation"
    ]

    fig = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Correlation with Action Risk Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

TARGET = "action_risk_score"

X = pd.get_dummies(
    df.drop(columns=[TARGET] + LEAKY),
    columns=cat_cols,
    drop_first=False
)

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

with st.expander("Model 1 — XGBoost", expanded=True):

    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

    with st.spinner("Training XGBoost..."):
        xgb_model.fit(
            X_train,
            y_train
        )

        pred = xgb_model.predict(
            X_test
        )

    r2 = r2_score(
        y_test,
        pred
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "R²",
        f"{r2:.3f}"
    )

    c2.metric(
        "MAE",
        f"{mae:.2f} points"
    )

    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": pred
    })

    fig = px.scatter(
        comparison,
        x="Actual",
        y="Predicted",
        title="Predicted vs Actual Risk Score"
    )

    minimum = min(
        comparison["Actual"].min(),
        comparison["Predicted"].min()
    )

    maximum = max(
        comparison["Actual"].max(),
        comparison["Predicted"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect Prediction"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Model 1 — SHAP"):

    import shap

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer.shap_values(
        X_test
    )

    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    )

    top_drivers = (
        mean_abs
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
        .reset_index()
    )

    top_drivers.columns = [
        "Feature",
        "Mean Absolute SHAP"
    ]

    fig = px.bar(
        top_drivers,
        x="Mean Absolute SHAP",
        y="Feature",
        orientation="h",
        title="Top 15 Drivers of Action Risk Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        "Top 10 drivers"
    )

    st.dataframe(
        mean_abs
        .sort_values(ascending=False)
        .head(10)
        .round(2)
        .to_frame("Mean |SHAP|"),
        use_container_width=True
    )

with st.expander("Model 2 — Human-Readable Decision Tree"):

    tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=50,
        random_state=RANDOM_STATE
    )

    tree.fit(
        X_train,
        y_train
    )

    tree_pred = tree.predict(
        X_test
    )

    tree_r2 = r2_score(
        y_test,
        tree_pred
    )

    st.metric(
        "Tree R²",
        f"{tree_r2:.3f}"
    )

    st.code(
        export_text(
            tree,
            feature_names=list(X.columns),
            decimals=1
        ),
        language="text"
    )

with st.expander("Model 3 — Isolation Forest / Blind Spots"):

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=RANDOM_STATE
    )

    anomaly = (
        iso.fit_predict(X)
        == -1
    )

    df_a = df.assign(
        anomaly=anomaly
    )

    st.write(
        f"Flagged **{anomaly.sum():,}** anomalous requests "
        f"({anomaly.mean():.1%})"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean Score — Anomalies",
        f"{df_a.loc[anomaly, TARGET].mean():.1f}"
    )

    c2.metric(
        "Mean Score — Normal",
        f"{df_a.loc[~anomaly, TARGET].mean():.1f}"
    )

    score_comparison = pd.DataFrame({
        "Group": [
            "Anomalies",
            "Normal"
        ],
        "Mean Risk Score": [
            df_a.loc[anomaly, TARGET].mean(),
            df_a.loc[~anomaly, TARGET].mean()
        ]
    })

    fig = px.bar(
        score_comparison,
        x="Group",
        y="Mean Risk Score",
        title="Mean Risk Score: Anomalous vs Normal Requests",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    blind_threshold = st.slider(
        "Blind-spot risk threshold",
        min_value=0,
        max_value=100,
        value=40
    )

    blind_spots = df_a[
        anomaly
        &
        (df_a[TARGET] < blind_threshold)
    ]

    st.metric(
        "Potential Blind Spots",
        len(blind_spots)
    )

    st.dataframe(
        blind_spots.sort_values(TARGET),
        use_container_width=True
    )

    if len(blind_spots):

        for col in [
            "requested_action",
            "resource_type",
            "user_role",
            "access_decision"
        ]:

            if col in blind_spots.columns:

                counts = (
                    blind_spots[col]
                    .value_counts()
                    .head(10)
                    .reset_index()
                )

                counts.columns = [
                    col,
                    "Count"
                ]

                fig = px.bar(
                    counts,
                    x=col,
                    y="Count",
                    title=f"Blind Spots — {col}"
                )

                fig.update_layout(
                    xaxis_tickangle=-45
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

with st.expander("Model 4 — Do Access Decisions Follow the Score?"):

    X_dec = pd.get_dummies(
        df.drop(
            columns=[
                "access_decision",
                "human_approval_required"
            ]
        ),
        columns=cat_cols
    )

    y_dec = df["access_decision"]

    scaler = StandardScaler()

    X_dec_s = scaler.fit_transform(
        X_dec
    )

    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        X_dec_s,
        y_dec,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dec
    )

    logit = LogisticRegression(
        max_iter=2000,
        C=1.0
    )

    logit.fit(
        Xd_tr,
        yd_tr
    )

    logit_pred = logit.predict(
        Xd_te
    )

    report = classification_report(
        yd_te,
        logit_pred,
        zero_division=0,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).T,
        use_container_width=True
    )

    coefs = pd.DataFrame(
        logit.coef_.T,
        index=X_dec.columns,
        columns=logit.classes_
    )

    selected_class = st.selectbox(
        "View strongest coefficients for",
        list(logit.classes_)
    )

    strongest = (
        coefs[selected_class]
        .sort_values(
            key=abs,
            ascending=False
        )
        .head(15)
        .sort_values()
        .reset_index()
    )

    strongest.columns = [
        "Feature",
        "Coefficient"
    ]

    fig = px.bar(
        strongest,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title=f"Strongest Logistic Regression Signals — {selected_class}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Contradictions in Enforcement"):

    allowed_high = df[
        (df.access_decision == "Allowed")
        &
        (df[TARGET] >= 70)
    ]

    blocked_low = df[
        (df.access_decision == "Blocked")
        &
        (df[TARGET] <= 30)
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Allowed Despite Risk ≥ 70",
        len(allowed_high)
    )

    c2.metric(
        "Blocked Despite Risk ≤ 30",
        len(blocked_low)
    )

    st.subheader("Allowed despite high risk")

    st.dataframe(
        allowed_high,
        use_container_width=True
    )

    st.subheader("Blocked despite low risk")

    st.dataframe(
        blocked_low,
        use_container_width=True
    )

with st.expander("Research Question 1 — Four Named Factor Groups"):

    feature_groups = {
        "Autonomy Level": [
            c for c in X.columns
            if "autonomy" in c.lower()
        ],
        "Permissions": [
            c for c in X.columns
            if (
                "permission" in c.lower()
                or "permit" in c.lower()
            )
        ],
        "Requested Tools": [
            c for c in X.columns
            if (
                "tool" in c.lower()
                or "api" in c.lower()
            )
        ],
        "Resource Access": [
            c for c in X.columns
            if "resource" in c.lower()
        ]
    }

    named_cols = sorted({
        c
        for cols in feature_groups.values()
        for c in cols
    })

    factor_counts = pd.DataFrame({
        "Factor Group": list(feature_groups.keys()),
        "Encoded Features": [
            len(v)
            for v in feature_groups.values()
        ]
    })

    st.dataframe(
        factor_counts,
        use_container_width=True
    )

    if named_cols:

        xgb_named = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE
        )

        xgb_named.fit(
            X_train[named_cols],
            y_train
        )

        pred_named = xgb_named.predict(
            X_test[named_cols]
        )

        r2_named = r2_score(
            y_test,
            pred_named
        )

        mae_named = mean_absolute_error(
            y_test,
            pred_named
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "R² — Named Factors",
            f"{r2_named:.3f}"
        )

        c2.metric(
            "MAE — Named Factors",
            f"{mae_named:.2f}"
        )

        comparison = pd.DataFrame({
            "Actual": y_test,
            "Predicted": pred_named
        })

        fig = px.scatter(
            comparison,
            x="Actual",
            y="Predicted",
            title="Predicted vs Actual — Named Factors Only"
        )

        minimum = min(
            comparison["Actual"].min(),
            comparison["Predicted"].min()
        )

        maximum = max(
            comparison["Actual"].max(),
            comparison["Predicted"].max()
        )

        fig.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                name="Perfect Prediction"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        r2_full = r2_score(
            y_test,
            xgb_model.predict(X_test)
        )

        model_comparison = pd.DataFrame({
            "Model": [
                "Named Factors",
                "All Features"
            ],
            "R²": [
                r2_named,
                r2_full
            ]
        })

        fig = px.bar(
            model_comparison,
            x="Model",
            y="R²",
            title="Named Factors vs All Features",
            text_auto=".3f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with st.expander("Research Question 2 — Can Risk Score Alone Decide Access?"):

    score_stats = (
        df.groupby("access_decision")[TARGET]
        .agg([
            "mean",
            "median",
            "count"
        ])
        .round(2)
    )

    st.dataframe(
        score_stats,
        use_container_width=True
    )

    Xs = df[[TARGET]].values
    ys = df["access_decision"]

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs,
        ys,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=ys
    )

    score_only = LogisticRegression(
        max_iter=1000
    )

    score_only.fit(
        Xs_tr,
        ys_tr
    )

    pred_so = score_only.predict(
        Xs_te
    )

    acc_so = accuracy_score(
        ys_te,
        pred_so
    )

    f1_so = f1_score(
        ys_te,
        pred_so,
        average="macro",
        zero_division=0
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Score-Only Accuracy",
        f"{acc_so:.3f}"
    )

    c2.metric(
        "Score-Only Macro F1",
        f"{f1_so:.3f}"
    )

    st.dataframe(
        pd.DataFrame(
            classification_report(
                ys_te,
                pred_so,
                zero_division=0,
                output_dict=True
            )
        ).T,
        use_container_width=True
    )

    labels = sorted(
        ys.unique()
    )

    cm = confusion_matrix(
        ys_te,
        pred_so,
        labels=labels
    )

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        title="Score-Only Confusion Matrix"
    )

    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    acc_full = accuracy_score(
        yd_te,
        logit.predict(Xd_te)
    )

    share = (
        acc_so / acc_full
        if acc_full
        else np.nan
    )

    comparison = pd.DataFrame({
        "Model": [
            "Risk Score Alone",
            "All Features"
        ],
        "Accuracy": [
            acc_so,
            acc_full
        ]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        title="Score Alone vs All Features",
        text_auto=".3f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        f"Score alone recovers **{share:.0%}** of the accuracy "
        "of the full-feature model."
    )

with st.expander("Score + Key Security Signals"):

    extra = [
        c for c in [
            "permission_match",
            "prompt_injection_detected",
            "data_exfiltration_risk"
        ]
        if c in df.columns
    ]

    Xb = df[
        [TARGET] + extra
    ].values

    yb = df[
        "access_decision"
    ]

    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        Xb,
        yb,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=yb
    )

    clf_plus = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    clf_plus.fit(
        Xb_tr,
        yb_tr
    )

    pred_plus = clf_plus.predict(
        Xb_te
    )

    acc_plus = accuracy_score(
        yb_te,
        pred_plus
    )

    f1_plus = f1_score(
        yb_te,
        pred_plus,
        average="macro",
        zero_division=0
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Accuracy",
        f"{acc_plus:.3f}"
    )

    c2.metric(
        "Macro F1",
        f"{f1_plus:.3f}"
    )

    st.dataframe(
        pd.DataFrame(
            classification_report(
                yb_te,
                pred_plus,
                zero_division=0,
                output_dict=True
            )
        ).T,
        use_container_width=True
    )

with st.expander("Robustness — Linear Baseline and Cross Validation"):

    lin = LinearRegression().fit(
        X_train,
        y_train
    )

    linear_r2 = r2_score(
        y_test,
        lin.predict(X_test)
    )

    cv_xgb = cross_val_score(
        xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE
        ),
        X,
        y,
        cv=5,
        scoring="r2"
    )

    cv_lin = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=5,
        scoring="r2"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Linear Test R²",
        f"{linear_r2:.3f}"
    )

    c2.metric(
        "XGBoost Test R²",
        f"{r2:.3f}"
    )

    c3.metric(
        "XGBoost CV R²",
        f"{cv_xgb.mean():.3f}"
    )

    c4.metric(
        "Linear CV R²",
        f"{cv_lin.mean():.3f}"
    )

    cv_df = pd.DataFrame({
        "Fold": range(1, 6),
        "XGBoost": cv_xgb,
        "Linear": cv_lin
    })

    fig = px.bar(
        cv_df,
        x="Fold",
        y=["XGBoost", "Linear"],
        barmode="group",
        title="5-Fold Cross-Validation R²"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Secondary Leakage Check"):

    suspects = [
        "data_exfiltration_risk",
        "prompt_injection_detected",
        "previous_failed_attempts"
    ]

    present = [
        c for c in suspects
        if c in df.columns
    ]

    correlations = (
        df[present + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .round(3)
        .reset_index()
    )

    correlations.columns = [
        "Feature",
        "Correlation"
    ]

    st.dataframe(
        correlations,
        use_container_width=True
    )

    drop_cols = [
        c for c in present
        if c in X.columns
    ]

    m_no_threat = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

    m_no_threat.fit(
        X_train.drop(columns=drop_cols),
        y_train
    )

    no_threat_r2 = r2_score(
        y_test,
        m_no_threat.predict(
            X_test.drop(columns=drop_cols)
        )
    )

    comparison = pd.DataFrame({
        "Model": [
            "With Threat Signals",
            "Without Threat Signals"
        ],
        "R²": [
            r2,
            no_threat_r2
        ]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="R²",
        title="Threat Signal Sensitivity",
        text_auto=".3f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Class Imbalance — Balanced Logistic Regression"):

    logit_bal = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced"
    )

    logit_bal.fit(
        Xd_tr,
        yd_tr
    )

    balanced_pred = logit_bal.predict(
        Xd_te
    )

    st.dataframe(
        pd.DataFrame(
            classification_report(
                yd_te,
                balanced_pred,
                zero_division=0,
                output_dict=True
            )
        ).T,
        use_container_width=True
    )

with st.expander("Risk Score Distribution by Access Decision"):

    fig = px.box(
        df,
        x=TARGET,
        y="access_decision",
        color="access_decision",
        points="all",
        title="Risk Score Distribution by Access Decision"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with st.expander("Over-Blocked Cases"):

    over_blocked = df[
        (df.access_decision == "Blocked")
        &
        (df[TARGET] <= 30)
    ]

    threshold = st.slider(
        "Low-risk threshold",
        0,
        100,
        30
    )

    over_blocked = df[
        (df.access_decision == "Blocked")
        &
        (df[TARGET] <= threshold)
    ]

    st.metric(
        "Over-Blocked Cases",
        len(over_blocked)
    )

    st.dataframe(
        over_blocked,
        use_container_width=True
    )

    for col in [
        "requested_action",
        "resource_type",
        "user_role",
        "agent_role"
    ]:

        if col in over_blocked.columns:

            counts = (
                over_blocked[col]
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                col,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=col,
                y="Count",
                title=f"Over-Blocked Cases — {col}"
            )

            fig.update_layout(
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

with st.expander("SHAP Dependence — Permission Match"):

    if "permission_match" in X_test.columns:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        shap.dependence_plot(
            "permission_match",
            shap_values,
            X_test,
            ax=ax,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

with st.expander("SHAP Dependence — Resource Sensitivity"):

    if "resource_sensitivity" in X_test.columns:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        shap.dependence_plot(
            "resource_sensitivity",
            shap_values,
            X_test,
            ax=ax,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

with st.expander("YData Profiling"):

    generate_report = st.button(
        "Generate Full YData Profiling Report"
    )

    if generate_report:

        try:

            from data_profiling import ProfileReport

            with st.spinner(
                "Generating profiling report..."
            ):

                report = ProfileReport(
                    df,
                    title="AgentEYE Dataset Audit Report",
                    explorative=True
                )

                report_html = report.to_html()

            st.components.v1.html(
                report_html,
                height=1200,
                scrolling=True
            )

        except Exception as e:

            st.error(
                "YData Profiling could not be loaded."
            )

            st.code(
                str(e)
            )

st.divider()

st.caption(
    "AgentEYE — AI Agent Security Risk Analysis"
)