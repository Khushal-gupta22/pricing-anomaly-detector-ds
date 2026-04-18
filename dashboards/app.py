"""
Dynamic Pricing Anomaly Detector — Streamlit Dashboard
======================================================
Production-grade monitoring dashboard that surfaces pricing anomalies
in real time. Designed to look like what Uber/Lyft internal pricing
monitoring tools actually look like.

Pages:
1. Command Center  — Live KPIs, severity heatmap, recent critical alerts
2. Time Series Explorer — Interactive price charts with anomaly overlays
3. Alert Investigation — Drill-down into individual anomalies
4. Model Performance — Precision/recall tracking, method comparison
5. Data Explorer — Raw data access with filters

Run: streamlit run dashboards/app.py
"""

import sys
import os
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    print(f"[dashboard] Detected venv at .venv/ — re-launching with venv Python...", flush=True)
    result = subprocess.run(
        [VENV_PYTHON, "-u", __file__] + sys.argv[1:],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)
    
# Ensure print output is visible immediately (no buffering)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

print(f"[starting app] Starting (Python: {sys.executable})", flush=True)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pricing Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────
DB_PATH = os.path.join(PROJECT_ROOT, "data", "pricing_anomalies.db")


@st.cache_data(ttl=60)
def load_pricing_events():
    """Load pricing events from SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM pricing_events ORDER BY timestamp", conn
        )
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=60)
def load_detected_anomalies():
    """Load detected anomalies from SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM detected_anomalies ORDER BY anomaly_score DESC", conn
        )
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(ttl=60)
def load_ground_truth():
    """Load ground truth anomalies."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM anomaly_ground_truth", conn
            )
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_model_performance():
    """Load model performance history."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                """SELECT mp.*, dr.started_at as run_date
                   FROM model_performance mp
                   JOIN detection_runs dr ON mp.run_id = dr.run_id
                   ORDER BY dr.started_at DESC""", conn
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_hourly_stats():
    """Load hourly aggregated stats."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                category,
                COUNT(*) as event_count,
                AVG(final_price) as avg_price,
                MIN(final_price) as min_price,
                MAX(final_price) as max_price,
                AVG(surge_multiplier) as avg_surge,
                AVG(demand_level) as avg_demand,
                AVG(supply_level) as avg_supply,
                SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
            FROM pricing_events
            GROUP BY hour, category
            ORDER BY hour""", conn
        )
    if not df.empty:
        df['hour'] = pd.to_datetime(df['hour'])
    return df


# ─────────────────────────────────────────────────────────
# Severity color mapping
# ─────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    'critical': '#DC2626',
    'high': '#EA580C',
    'medium': '#CA8A04',
    'low': '#2563EB',
    'none': '#6B7280'
}

SEVERITY_ORDER = ['critical', 'high', 'medium', 'low']


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Pricing Anomaly Detector")
    st.caption("Real-time monitoring for dynamic pricing algorithms")
    
    page = st.radio(
        "Navigate",
        [
            "Command Center",
            "Time Series Explorer",
            "Alert Investigation",
            "Model Performance",
            "Data Explorer"
        ],
        index=0
    )
    
    st.divider()
    
    # Data status
    pricing_df = load_pricing_events()
    anomalies_df = load_detected_anomalies()
    
    if pricing_df.empty:
        st.error("No data loaded. Run the pipeline first:")
        st.code("python run_pipeline.py", language="bash")
        st.stop()
    
    st.metric("Total Events", f"{len(pricing_df):,}")
    st.metric("Detected Anomalies", f"{len(anomalies_df):,}")
    
    date_range = st.date_input(
        "Date Range",
        value=(
            pricing_df['timestamp'].min().date(),
            pricing_df['timestamp'].max().date()
        ),
        min_value=pricing_df['timestamp'].min().date(),
        max_value=pricing_df['timestamp'].max().date(),
    )
    
    categories = st.multiselect(
        "Categories",
        options=sorted(pricing_df['category'].unique()),
        default=sorted(pricing_df['category'].unique())
    )
    
    st.divider()
    st.caption(f"Data: {pricing_df['timestamp'].min().strftime('%b %d')} — "
               f"{pricing_df['timestamp'].max().strftime('%b %d, %Y')}")


# ─────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────
def apply_filters(df, date_range, categories):
    """Apply sidebar filters to a dataframe."""
    if df.empty or 'timestamp' not in df.columns:
        return df
    mask = df['category'].isin(categories)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        mask = mask & (df['timestamp'] >= start) & (df['timestamp'] < end)
    return df[mask]


pricing_filtered = apply_filters(pricing_df, date_range, categories)
anomalies_filtered = apply_filters(anomalies_df, date_range, categories)


# =============================================================
# PAGE 1: Command Center
# =============================================================
if page == "Command Center":
    st.header("Command Center")
    st.caption("Real-time overview of pricing health across all categories")
    
    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_events = len(pricing_filtered)
    total_anomalies = len(anomalies_filtered)
    anomaly_rate = total_anomalies / max(total_events, 1) * 100
    
    critical_count = len(anomalies_filtered[anomalies_filtered['severity'] == 'critical']) if not anomalies_filtered.empty else 0
    high_count = len(anomalies_filtered[anomalies_filtered['severity'] == 'high']) if not anomalies_filtered.empty else 0
    
    # Estimated revenue impact
    if not anomalies_filtered.empty and 'final_price' in anomalies_filtered.columns:
        impact = anomalies_filtered['final_price'].abs().sum()
    else:
        impact = 0
    
    col1.metric("Events Monitored", f"{total_events:,}")
    col2.metric("Anomalies Detected", f"{total_anomalies:,}", 
                delta=f"{anomaly_rate:.1f}% rate", delta_color="inverse")
    col3.metric("Critical Alerts", f"{critical_count}",
                delta="Needs Immediate Action" if critical_count > 0 else "All Clear",
                delta_color="inverse" if critical_count > 0 else "normal")
    col4.metric("High Alerts", f"{high_count}")
    col5.metric("Est. Impact", f"${impact:,.0f}")
    
    st.divider()
    
    # ── Severity distribution + Timeline ──
    left, right = st.columns([1, 2])
    
    with left:
        st.subheader("Severity Distribution")
        if not anomalies_filtered.empty and 'severity' in anomalies_filtered.columns:
            sev_counts = anomalies_filtered['severity'].value_counts()
            # Ensure order
            sev_data = pd.DataFrame({
                'severity': SEVERITY_ORDER,
                'count': [sev_counts.get(s, 0) for s in SEVERITY_ORDER]
            })
            sev_data = sev_data[sev_data['count'] > 0]
            
            fig = px.bar(
                sev_data, x='severity', y='count',
                color='severity',
                color_discrete_map=SEVERITY_COLORS,
                text='count'
            )
            fig.update_layout(
                showlegend=False, height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="", yaxis_title="Count"
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected in selected range.")
    
    with right:
        st.subheader("Anomaly Timeline")
        if not anomalies_filtered.empty:
            # Bin anomalies by hour
            anom_hourly = anomalies_filtered.copy()
            anom_hourly['hour'] = anom_hourly['timestamp'].dt.floor('h')
            timeline = anom_hourly.groupby(['hour', 'severity']).size().reset_index(name='count')
            
            fig = px.bar(
                timeline, x='hour', y='count', color='severity',
                color_discrete_map=SEVERITY_COLORS,
                category_orders={'severity': SEVERITY_ORDER}
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="", yaxis_title="Anomalies / Hour",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies to plot.")
    
    # ── Anomalies by category heatmap ──
    st.subheader("Category x Severity Heatmap")
    if not anomalies_filtered.empty:
        heatmap_data = anomalies_filtered.groupby(
            ['category', 'severity']
        ).size().reset_index(name='count')
        
        heatmap_pivot = heatmap_data.pivot_table(
            values='count', index='category', columns='severity', fill_value=0
        )
        # Reorder columns
        for s in SEVERITY_ORDER:
            if s not in heatmap_pivot.columns:
                heatmap_pivot[s] = 0
        heatmap_pivot = heatmap_pivot[SEVERITY_ORDER]
        
        fig = px.imshow(
            heatmap_pivot.values,
            labels=dict(x="Severity", y="Category", color="Count"),
            x=SEVERITY_ORDER,
            y=list(heatmap_pivot.index),
            color_continuous_scale="OrRd",
            text_auto=True,
            aspect="auto"
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Recent critical alerts table ──
    st.subheader("Recent Critical & High Alerts")
    if not anomalies_filtered.empty:
        critical_alerts = anomalies_filtered[
            anomalies_filtered['severity'].isin(['critical', 'high'])
        ].sort_values('anomaly_score', ascending=False).head(20)
        
        if not critical_alerts.empty:
            display_cols = ['timestamp', 'category', 'severity', 'anomaly_score',
                           'final_price', 'description']
            available_cols = [c for c in display_cols if c in critical_alerts.columns]
            st.dataframe(
                critical_alerts[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'anomaly_score': st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=1, format="%.2f"
                    ),
                    'final_price': st.column_config.NumberColumn(
                        "Price", format="$%.2f"
                    ),
                }
            )
        else:
            st.success("No critical or high severity alerts. Pricing looks healthy!")
    

# =============================================================
# PAGE 2: Time Series Explorer
# =============================================================
elif page == "Time Series Explorer":
    st.header("Time Series Explorer")
    st.caption("Interactive price time series with anomaly overlays and Prophet forecast bands")
    
    hourly_stats = load_hourly_stats()
    
    if hourly_stats.empty:
        st.warning("No hourly stats available. Run the pipeline first.")
        st.stop()
    
    # Category selector (single for this page)
    selected_cat = st.selectbox(
        "Select Category", 
        sorted(pricing_filtered['category'].unique()),
        index=0
    )
    
    cat_hourly = hourly_stats[hourly_stats['category'] == selected_cat].copy()
    
    if cat_hourly.empty:
        st.warning(f"No data for {selected_cat}")
        st.stop()
    
    # ── Main time series chart ──
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"{selected_cat} — Average Price",
            "Surge Multiplier",
            "Demand vs Supply"
        ),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price with min/max band
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['max_price'],
            fill=None, mode='lines', line=dict(width=0),
            showlegend=False, name='Max'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['min_price'],
            fill='tonexty', mode='lines', line=dict(width=0),
            fillcolor='rgba(99, 102, 241, 0.15)',
            showlegend=True, name='Price Range'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['avg_price'],
            mode='lines', line=dict(color='#4F46E5', width=2),
            name='Avg Price'
        ), row=1, col=1
    )
    
    # Overlay anomaly hours
    anom_hours = cat_hourly[cat_hourly['anomaly_count'] > 0]
    if not anom_hours.empty:
        fig.add_trace(
            go.Scatter(
                x=anom_hours['hour'], y=anom_hours['avg_price'],
                mode='markers',
                marker=dict(
                    color='red', size=8, symbol='x',
                    line=dict(width=1, color='darkred')
                ),
                name=f'Anomalies ({len(anom_hours)} hours)'
            ), row=1, col=1
        )
    
    # Surge multiplier
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['avg_surge'],
            mode='lines', line=dict(color='#EA580C', width=1.5),
            name='Avg Surge'
        ), row=2, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="No surge", row=2, col=1)
    
    # Demand vs Supply
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['avg_demand'],
            mode='lines', line=dict(color='#DC2626', width=1.5),
            name='Demand'
        ), row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cat_hourly['hour'], y=cat_hourly['avg_supply'],
            mode='lines', line=dict(color='#16A34A', width=1.5),
            name='Supply'
        ), row=3, col=1
    )
    
    fig.update_layout(
        height=700,
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Surge", row=2, col=1)
    fig.update_yaxes(title_text="Level", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Price distribution ──
    st.subheader("Price Distribution")
    col1, col2 = st.columns(2)
    
    cat_events = pricing_filtered[pricing_filtered['category'] == selected_cat]
    
    with col1:
        fig_hist = px.histogram(
            cat_events, x='final_price', nbins=80,
            color='is_anomaly',
            color_discrete_map={0: '#4F46E5', 1: '#DC2626'},
            labels={'is_anomaly': 'Is Anomaly'},
            title="Price Distribution (Normal vs Anomalous)"
        )
        fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_surge = px.histogram(
            cat_events, x='surge_multiplier', nbins=50,
            color='is_anomaly',
            color_discrete_map={0: '#EA580C', 1: '#DC2626'},
            labels={'is_anomaly': 'Is Anomaly'},
            title="Surge Distribution"
        )
        fig_surge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_surge, use_container_width=True)
    
    # ── Hourly pattern ──
    st.subheader("Typical Hourly Pattern")
    hourly_pattern = cat_events.copy()
    hourly_pattern['hour_of_day'] = hourly_pattern['timestamp'].dt.hour
    pattern = hourly_pattern.groupby('hour_of_day').agg(
        avg_price=('final_price', 'mean'),
        avg_surge=('surge_multiplier', 'mean'),
        avg_demand=('demand_level', 'mean'),
    ).reset_index()
    
    fig_pattern = px.line(
        pattern, x='hour_of_day', y='avg_price',
        title="Average Price by Hour of Day",
        markers=True
    )
    fig_pattern.update_layout(
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(dtick=1, title="Hour of Day"),
        yaxis=dict(title="Avg Price ($)")
    )
    st.plotly_chart(fig_pattern, use_container_width=True)


# =============================================================
# PAGE 3: Alert Investigation
# =============================================================
elif page == "Alert Investigation":
    st.header("Alert Investigation")
    st.caption("Drill down into individual anomalies — understand what triggered each alert")
    
    if anomalies_filtered.empty:
        st.info("No anomalies detected in the selected date range.")
        st.stop()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sev_filter = st.multiselect(
            "Severity", SEVERITY_ORDER, default=SEVERITY_ORDER
        )
    with col2:
        if 'anomaly_type' in anomalies_filtered.columns:
            types = sorted(anomalies_filtered['anomaly_type'].dropna().unique())
            type_filter = st.multiselect("Anomaly Type", types, default=types)
        else:
            type_filter = []
    with col3:
        score_min = st.slider("Min Score", 0.0, 1.0, 0.0, 0.05)
    
    # Apply alert filters
    mask = anomalies_filtered['severity'].isin(sev_filter)
    if type_filter and 'anomaly_type' in anomalies_filtered.columns:
        mask = mask & anomalies_filtered['anomaly_type'].isin(type_filter)
    mask = mask & (anomalies_filtered['anomaly_score'] >= score_min)
    filtered_alerts = anomalies_filtered[mask].sort_values(
        'anomaly_score', ascending=False
    )
    
    st.metric("Matching Alerts", len(filtered_alerts))
    
    # Alert table
    display_cols = ['timestamp', 'category', 'severity', 'anomaly_score',
                    'anomaly_type', 'final_price', 'expected_price_low',
                    'expected_price_high', 'description', 'detection_method']
    available_cols = [c for c in display_cols if c in filtered_alerts.columns]
    
    st.dataframe(
        filtered_alerts[available_cols].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            'anomaly_score': st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=1, format="%.3f"
            ),
            'final_price': st.column_config.NumberColumn("Price", format="$%.2f"),
            'expected_price_low': st.column_config.NumberColumn("Exp. Low", format="$%.2f"),
            'expected_price_high': st.column_config.NumberColumn("Exp. High", format="$%.2f"),
        }
    )
    
    # Anomaly type breakdown
    st.subheader("Anomaly Type Breakdown")
    if 'anomaly_type' in filtered_alerts.columns:
        type_counts = filtered_alerts['anomaly_type'].value_counts().reset_index()
        type_counts.columns = ['anomaly_type', 'count']
        
        fig = px.bar(
            type_counts.head(15), x='count', y='anomaly_type',
            orientation='h', color='count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(categoryorder='total ascending'),
            showlegend=False,
            xaxis_title="Count", yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detection method overlap
    st.subheader("Detection Method Effectiveness")
    if 'detection_method' in filtered_alerts.columns:
        method_data = filtered_alerts['detection_method'].str.split(' \\+ ', expand=True)
        method_counts = {}
        for col in method_data.columns:
            for method in method_data[col].dropna().unique():
                method = method.strip()
                if method:
                    method_counts[method] = method_counts.get(method, 0) + len(
                        method_data[method_data[col].str.strip() == method]
                    )
        
        if method_counts:
            mc_df = pd.DataFrame(
                list(method_counts.items()), columns=['method', 'detections']
            ).sort_values('detections', ascending=False)
            
            fig = px.bar(mc_df, x='method', y='detections', color='detections',
                        color_continuous_scale='Blues')
            fig.update_layout(
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================
# PAGE 4: Model Performance
# =============================================================
elif page == "Model Performance":
    st.header("Model Performance")
    st.caption("How well does the detection pipeline perform against known anomalies?")
    
    perf_df = load_model_performance()
    ground_truth = load_ground_truth()
    
    if perf_df.empty:
        st.warning("No performance data available. Run the pipeline first.")
        st.stop()
    
    # ── Latest performance metrics ──
    st.subheader("Latest Run — Detection Performance")
    
    latest_methods = perf_df.drop_duplicates(subset=['method'], keep='first')
    
    cols = st.columns(len(latest_methods))
    for i, (_, row) in enumerate(latest_methods.iterrows()):
        with cols[i]:
            method_name = row['method'].replace('_', ' ').title()
            st.markdown(f"**{method_name}**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{row['precision_score']:.2%}")
            m2.metric("Recall", f"{row['recall_score']:.2%}")
            m3.metric("F1", f"{row['f1_score']:.2%}")
    
    st.divider()
    
    # ── Performance comparison chart ──
    st.subheader("Method Comparison")
    
    comparison_df = latest_methods.melt(
        id_vars=['method'],
        value_vars=['precision_score', 'recall_score', 'f1_score'],
        var_name='metric',
        value_name='value'
    )
    comparison_df['metric'] = comparison_df['metric'].str.replace('_score', '').str.title()
    
    fig = px.bar(
        comparison_df, x='method', y='value', color='metric',
        barmode='group',
        color_discrete_sequence=['#4F46E5', '#16A34A', '#EA580C'],
        labels={'value': 'Score', 'method': 'Detection Method'}
    )
    fig.update_layout(
        height=400, margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Ground truth analysis ──
    if not ground_truth.empty:
        st.subheader("Ground Truth Anomaly Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'anomaly_type' in ground_truth.columns:
                gt_types = ground_truth['anomaly_type'].value_counts().reset_index()
                gt_types.columns = ['type', 'count']
                fig = px.pie(
                    gt_types, values='count', names='type',
                    title="Injected Anomaly Types",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'severity' in ground_truth.columns:
                gt_sev = ground_truth['severity'].value_counts().reset_index()
                gt_sev.columns = ['severity', 'count']
                fig = px.pie(
                    gt_sev, values='count', names='severity',
                    title="Injected Anomaly Severities",
                    color_discrete_map=SEVERITY_COLORS
                )
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
    
    # ── Confusion matrix approximation ──
    st.subheader("Detection Coverage")
    
    total_events = len(pricing_df)
    gt_anomalies = pricing_df['is_anomaly'].sum() if 'is_anomaly' in pricing_df.columns else 0
    detected = len(anomalies_df) if not anomalies_df.empty else 0
    
    coverage_data = pd.DataFrame({
        'Metric': ['Total Events', 'Ground Truth Anomalies', 'Detected Anomalies',
                   'Detection Rate', 'Anomaly Prevalence'],
        'Value': [
            f"{total_events:,}",
            f"{gt_anomalies:,}",
            f"{detected:,}",
            f"{detected/max(gt_anomalies, 1)*100:.1f}%",
            f"{gt_anomalies/max(total_events, 1)*100:.2f}%"
        ]
    })
    st.table(coverage_data)


# =============================================================
# PAGE 5: Data Explorer
# =============================================================
elif page == "Data Explorer":
    st.header("Data Explorer")
    st.caption("Raw data access with filters — useful for investigation and debugging")
    
    tab1, tab2, tab3 = st.tabs([
        "Pricing Events", "Detected Anomalies", "Summary Stats"
    ])
    
    with tab1:
        st.subheader("Pricing Events")
        
        col1, col2 = st.columns(2)
        with col1:
            show_anomalies_only = st.checkbox("Show only anomalous events", value=False)
        with col2:
            max_rows = st.number_input("Max rows", 50, 5000, 500, 50)
        
        display_data = pricing_filtered.copy()
        if show_anomalies_only:
            display_data = display_data[display_data['is_anomaly'] == 1]
        
        st.dataframe(
            display_data.head(max_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                'final_price': st.column_config.NumberColumn("Price", format="$%.2f"),
                'base_price': st.column_config.NumberColumn("Base", format="$%.2f"),
                'surge_multiplier': st.column_config.NumberColumn("Surge", format="%.2fx"),
            }
        )
        
        st.caption(f"Showing {min(max_rows, len(display_data)):,} of {len(display_data):,} rows")
    
    with tab2:
        st.subheader("Detected Anomalies")
        
        if anomalies_filtered.empty:
            st.info("No detected anomalies in range.")
        else:
            st.dataframe(
                anomalies_filtered.head(500),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'anomaly_score': st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=1, format="%.3f"
                    ),
                    'final_price': st.column_config.NumberColumn("Price", format="$%.2f"),
                }
            )
    
    with tab3:
        st.subheader("Summary Statistics by Category")
        
        summary = pricing_filtered.groupby('category').agg(
            count=('final_price', 'count'),
            avg_price=('final_price', 'mean'),
            median_price=('final_price', 'median'),
            std_price=('final_price', 'std'),
            min_price=('final_price', 'min'),
            max_price=('final_price', 'max'),
            avg_surge=('surge_multiplier', 'mean'),
            max_surge=('surge_multiplier', 'max'),
            anomaly_count=('is_anomaly', 'sum'),
        ).round(2)
        
        summary['anomaly_rate_%'] = (summary['anomaly_count'] / summary['count'] * 100).round(2)
        
        st.dataframe(summary, use_container_width=True)
        
        # Box plot
        st.subheader("Price Distribution by Category")
        fig = px.box(
            pricing_filtered, x='category', y='final_price',
            color='category', notched=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False, yaxis_title="Price ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Region breakdown
        st.subheader("Anomalies by Region")
        if 'region' in pricing_filtered.columns:
            region_stats = pricing_filtered.groupby('region').agg(
                total=('event_id', 'count'),
                anomalies=('is_anomaly', 'sum'),
                avg_price=('final_price', 'mean'),
                avg_surge=('surge_multiplier', 'mean'),
            ).round(2)
            region_stats['anomaly_rate_%'] = (
                region_stats['anomalies'] / region_stats['total'] * 100
            ).round(2)
            st.dataframe(region_stats, use_container_width=True)
