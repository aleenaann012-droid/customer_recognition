import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Configure page layout and style
st.set_page_config(
    page_title="Analytics Dashboard", 
    page_icon="✨", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern UI (Glassmorphism, Dark Theme, Gradients)
st.markdown("""
<style>
    /* Main Background Override */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #f8fafc !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards (KPIs) */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    
    /* Metric Text Styling */
    div[data-testid="metric-container"] > div {
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
    }
    
    /* Sections */
    h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Table Styling overrides (as much as Streamlit allows) */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide top padding */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>✨ Insight Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time customer demographics and staff monitoring.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Connect to database securely
@st.cache_resource
def get_connection():
    db_path = os.path.join("database", "customers.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)

conn = get_connection()

def load_data():
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Visitor stats
    try:
        df_visitors = pd.read_sql_query("SELECT * FROM visitor_stats", conn)
        if 'visit_time' not in df_visitors.columns:
            df_visitors['visit_time'] = "Unknown"
        else:
            df_visitors['visit_time'] = df_visitors['visit_time'].fillna("Unknown")
    except Exception:
        df_visitors = pd.DataFrame(columns=["id", "age_group", "gender", "visit_time"])
        
    # 2. Staff Attendance
    try:
        query_staff = """
            SELECT s.name, s.role, sa.entry_time, sa.date
            FROM staff_attendance sa
            JOIN staff s ON sa.staff_id = s.staff_id
            ORDER BY sa.date DESC, sa.entry_time DESC
        """
        df_staff = pd.read_sql_query(query_staff, conn)
    except Exception:
        df_staff = pd.DataFrame(columns=["name", "role", "entry_time", "date"])
        
    return df_visitors, df_staff, today_str

df_visitors, df_staff, today_str = load_data()

# Calculate KPIs
total_visitors = len(df_visitors)

if not df_staff.empty and "date" in df_staff.columns:
    staff_present_today = len(df_staff[df_staff["date"] == today_str]["name"].unique())
else:
    staff_present_today = 0

# --- KPI Cards ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("Total Visitors", f"{total_visitors:,}")
with col2:
    st.metric("Staff Present", f"{staff_present_today:,}")
with col3:
    st.empty() # Spacer for layout

st.markdown("<br>", unsafe_allow_html=True)

# --- Charts Section ---
st.markdown("### Demographic Insights")
col_chart1, col_chart2 = st.columns(2)

# Disable plotly modebar for a cleaner look
chart_config = {'displayModeBar': False, 'responsive': True}

with col_chart1:
    if not df_visitors.empty:
        age_counts = df_visitors["age_group"].value_counts().reset_index()
        age_counts.columns = ["Age Group", "Count"]
        
        age_order = ["0-17", "18-25", "26-35", "36-45", "46-60", "60+"]
        
        fig_age = px.bar(
            age_counts, 
            x="Age Group", 
            y="Count",
            category_orders={"Age Group": age_order},
            text="Count"
        )
        
        # Premium Styling for Bar Chart
        fig_age.update_traces(
            marker_color='#38bdf8',
            marker_line_color='#0ea5e9',
            marker_line_width=1.5,
            opacity=0.85,
            textposition='outside',
            textfont=dict(color='#cbd5e1', size=14)
        )
        
        fig_age.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, l=0, r=0, b=0),
            xaxis=dict(showgrid=False, title="", tickfont=dict(color='#cbd5e1', size=12)),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="", tickfont=dict(color='#cbd5e1'), visible=False),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_age, use_container_width=True, config=chart_config)
    else:
        st.info("No visitor data available.")

with col_chart2:
    if not df_visitors.empty:
        gender_counts = df_visitors["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        
        fig_gender = px.pie(
            gender_counts, 
            names="Gender", 
            values="Count", 
            hole=0.6,
            color="Gender",
            color_discrete_map={"Male": "#38bdf8", "Female": "#f472b6"}
        )
        
        # Premium Styling for Donut Chart
        fig_gender.update_traces(
            textposition='outside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Visitors: %{value}<extra></extra>',
            marker=dict(line=dict(color='#0f172a', width=3))
        )
        
        fig_gender.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, l=0, r=0, b=0),
            showlegend=False,
            font=dict(color='#cbd5e1', size=13),
            height=300
        )
        
        st.plotly_chart(fig_gender, use_container_width=True, config=chart_config)
    else:
        st.info("No visitor data available.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Tables Section ---
col_table1, col_table2 = st.columns(2)

with col_table1:
    st.markdown("### Recent Visitors")
    if not df_visitors.empty:
        # Sort by id descending
        recent_visitors = df_visitors.sort_values(by="id", ascending=False).head(10)
        display_visitors = recent_visitors[["visit_time", "age_group", "gender"]].rename(
            columns={"visit_time": "Time", "age_group": "Age Bracket", "gender": "Gender"}
        )
        st.dataframe(
            display_visitors, 
            use_container_width=True, 
            hide_index=True,
            height=300
        )
    else:
        st.info("No recent visitors logged.")

with col_table2:
    st.markdown("### Staff Attendance")
    if not df_staff.empty:
        display_staff = df_staff.rename(
            columns={"name": "Name", "role": "Role", "entry_time": "Entry Time", "date": "Date"}
        )
        st.dataframe(
            display_staff.head(10), 
            use_container_width=True, 
            hide_index=True,
            height=300
        )
    else:
        st.info("No staff attendance records found.")