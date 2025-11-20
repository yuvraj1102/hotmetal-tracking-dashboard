import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import time
import requests
import numpy as np

# Hardcoded password protection (no secrets file needed)
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Define users and passwords directly in code
    users = {
        "admin": "admin123",
        "operator": "operator123", 
        "manager": "manager123",
        "yuvraj": "password123"
    }

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] in users
            and st.session_state["password"] == users[st.session_state["username"]]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.title("🔥 HOT Metal Tracking Dashboard - Login")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("🔥 HOT Metal Tracking Dashboard - Login")
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

# Check authentication
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Page configuration
st.set_page_config(
    page_title="HOT Metal Tracking Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Rest of your existing code continues here...
# [ALL YOUR EXISTING FUNCTIONS AND MAIN CODE REMAIN THE SAME]
# ... copy all the functions from your previous script starting from get_sheet_data()


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import time
import requests
import numpy as np

# Page configuration
st.set_page_config(
    page_title="HOT Metal Tracking Dashboard",
    page_icon="🔥",
    layout="wide"
)

def get_sheet_data(sheet_url, gid=0):
    """
    Extract data from Google Sheets using CSV export
    """
    try:
        # Convert Google Sheets URL to CSV export URL
        if '/edit#' in sheet_url:
            base_url = sheet_url.split('/edit#')[0]
        elif '/edit?' in sheet_url:
            base_url = sheet_url.split('/edit?')[0]
        else:
            base_url = sheet_url.replace('/edit', '')
        
        csv_url = base_url + '/export?format=csv'
        
        # Add gid if provided
        if gid != 0:
            csv_url += f'&gid={gid}'
            
        # Read CSV data
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"Error accessing sheet: {e}")
        return pd.DataFrame()

def process_bf_data(df):
    """
    Process Blast Furnace data
    """
    if df.empty:
        return df
    
    # Standardize column names (case insensitive)
    df.columns = df.columns.str.upper()
    
    # Calculate NET WEIGHT if not present
    if 'NET WEIGHT' not in df.columns and all(col in df.columns for col in ['GROSS WEIGHT', 'TARE WEIGHT']):
        df['NET WEIGHT'] = df['GROSS WEIGHT'] - df['TARE WEIGHT']
    
    # Process date column
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce').dt.date
    
    # Process shift column
    if 'SHIFT' in df.columns:
        df['SHIFT'] = df['SHIFT'].astype(str).str.upper().str.strip()
    
    return df

def process_sms_data(df):
    """
    Process SMS data and calculate NET WEIGHT
    """
    if df.empty:
        return df
    
    # Standardize column names (case insensitive)
    df.columns = df.columns.str.upper()
    
    # Calculate NET WEIGHT as sum of P1, P2, P3
    p_columns = []
    for col in df.columns:
        if 'P1' in col and 'WEIGHT' in col:
            p_columns.append('P1 WEIGHT')
        elif 'P2' in col and 'WEIGHT' in col:
            p_columns.append('P2 WEIGHT')
        elif 'P3' in col and 'WEIGHT' in col:
            p_columns.append('P3 WEIGHT')
    
    # Remove duplicates and ensure proper order
    p_columns = sorted(list(set(p_columns)))
    
    if len(p_columns) >= 1:
        df['NET WEIGHT'] = df[p_columns].sum(axis=1)
    
    # Process date column
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce').dt.date
    
    # Process shift column
    if 'SHIFT' in df.columns:
        df['SHIFT'] = df['SHIFT'].astype(str).str.upper().str.strip()
    
    return df

def filter_data_by_date(df, selected_date):
    """
    Filter data by selected date
    """
    if df.empty or 'DATE' not in df.columns:
        return df
    
    if selected_date == "All Dates":
        return df
    else:
        return df[df['DATE'] == selected_date]

def filter_data_by_shift(df, selected_shift):
    """
    Filter data by selected shift
    """
    if df.empty or 'SHIFT' not in df.columns or selected_shift == "All Shifts":
        return df
    else:
        return df[df['SHIFT'] == selected_shift]

def create_comparison_chart(bf_data, sms_data):
    """
    Create bar graph comparing BF net weight vs SMS P1,P2,P3 weights
    """
    if bf_data.empty or sms_data.empty:
        return None
    
    # Get common LADLE NOs between both datasets
    common_ladles = set(bf_data['LADLE NO'].dropna().astype(str)).intersection(
        set(sms_data['LADLE NO'].dropna().astype(str))
    )
    
    if not common_ladles:
        st.warning("No common LADLE NOs found between BF and SMS data")
        return None
    
    # Limit to last 10 common ladles for better visualization
    common_ladles = sorted(common_ladles, key=lambda x: int(x) if x.isdigit() else x)[-10:]
    
    fig = go.Figure()
    
    # Colors for different weight types
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']  # Blue, Green, Orange, Red, Purple
    
    for i, ladle_no in enumerate(common_ladles):
        # BF data
        bf_rows = bf_data[bf_data['LADLE NO'].astype(str) == ladle_no]
        if not bf_rows.empty:
            bf_row = bf_rows.iloc[0]
            bf_net_weight = bf_row['NET WEIGHT'] if 'NET WEIGHT' in bf_row and pd.notna(bf_row['NET WEIGHT']) else 0
        else:
            bf_net_weight = 0
        
        # SMS data
        sms_rows = sms_data[sms_data['LADLE NO'].astype(str) == ladle_no]
        if not sms_rows.empty:
            sms_row = sms_rows.iloc[0]
            # Get P weights
            p1_weight = sms_row.get('P1 WEIGHT', 0) if 'P1 WEIGHT' in sms_row and pd.notna(sms_row.get('P1 WEIGHT')) else 0
            p2_weight = sms_row.get('P2 WEIGHT', 0) if 'P2 WEIGHT' in sms_row and pd.notna(sms_row.get('P2 WEIGHT')) else 0
            p3_weight = sms_row.get('P3 WEIGHT', 0) if 'P3 WEIGHT' in sms_row and pd.notna(sms_row.get('P3 WEIGHT')) else 0
            sms_net_weight = sms_row.get('NET WEIGHT', 0) if 'NET WEIGHT' in sms_row and pd.notna(sms_row.get('NET WEIGHT')) else 0
        else:
            p1_weight, p2_weight, p3_weight, sms_net_weight = 0, 0, 0, 0
        
        # Create grouped bars for each ladle
        categories = ['BF Net', 'SMS P1', 'SMS P2', 'SMS P3', 'SMS Total']
        weights = [bf_net_weight, p1_weight, p2_weight, p3_weight, sms_net_weight]
        
        for j, (category, weight, color) in enumerate(zip(categories, weights, colors)):
            fig.add_trace(go.Bar(
                x=[f"Ladle {ladle_no}"],
                y=[weight],
                name=category if i == 0 else "",  # Show legend only for first ladle
                legendgroup=category,
                marker_color=color,
                hovertemplate=f"Ladle: {ladle_no}<br>Type: {category}<br>Weight: {weight:,.0f} tons<extra></extra>"
            ))
    
    fig.update_layout(
        title="HOT Metal Weight Comparison: BF Net Weight vs SMS Pour Weights",
        xaxis_title="Ladle Number",
        yaxis_title="Weight (Tons)",
        barmode='group',
        height=500,
        showlegend=True,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_efficiency_chart(bf_data, sms_data):
    """
    Create efficiency comparison chart
    """
    if bf_data.empty or sms_data.empty:
        return None
    
    common_ladles = set(bf_data['LADLE NO'].dropna().astype(str)).intersection(
        set(sms_data['LADLE NO'].dropna().astype(str))
    )
    
    if not common_ladles:
        return None
    
    efficiencies = []
    ladle_nos = []
    
    for ladle_no in common_ladles:
        # BF data
        bf_rows = bf_data[bf_data['LADLE NO'].astype(str) == ladle_no]
        sms_rows = sms_data[sms_data['LADLE NO'].astype(str) == ladle_no]
        
        if not bf_rows.empty and not sms_rows.empty:
            bf_row = bf_rows.iloc[0]
            sms_row = sms_rows.iloc[0]
            
            bf_net_weight = bf_row['NET WEIGHT'] if 'NET WEIGHT' in bf_row and pd.notna(bf_row['NET WEIGHT']) else 0
            sms_net_weight = sms_row['NET WEIGHT'] if 'NET WEIGHT' in sms_row and pd.notna(sms_row['NET WEIGHT']) else 0
            
            if bf_net_weight > 0:
                efficiency = (sms_net_weight / bf_net_weight) * 100
                efficiencies.append(efficiency)
                ladle_nos.append(f"Ladle {ladle_no}")
    
    if efficiencies:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ladle_nos,
            y=efficiencies,
            marker_color=['green' if eff >= 95 else 'orange' if eff >= 90 else 'red' for eff in efficiencies],
            hovertemplate="%{x}<br>Efficiency: %{y:.1f}%<extra></extra>"
        ))
        
        fig.update_layout(
            title="Transfer Efficiency by Ladle",
            xaxis_title="Ladle Number",
            yaxis_title="Efficiency (%)",
            height=400,
            xaxis={'tickangle': -45}
        )
        
        # Add target line
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target: 95%")
        
        return fig
    
    return None

def main():
    st.title("🔥 HOT Metal Live Tracking Dashboard")
    st.markdown("### Real-time Tracking: Blast Furnace to SMS")
    
    # Pre-configured sheet URLs
    BF_SHEET_URL = "https://docs.google.com/spreadsheets/d/1DGskeHMsMGdSS-8JRXYkHOJqpPPbiJfIIA6-aNb9EPI/edit?gid=0#gid=0"
    SMS_SHEET_URL = "https://docs.google.com/spreadsheets/d/1d-QJ3nXo0IHwZk8-VLAQ_K8kCfqbHlcU4WjnbstW_Cc/edit?gid=0#gid=0"
    
    # Configuration section
    st.sidebar.header("Configuration")
    
    # Display current URLs (read-only)
    st.sidebar.info("**Current Sheets:**")
    st.sidebar.text(f"BF: {BF_SHEET_URL.split('/d/')[1][:20]}...")
    st.sidebar.text(f"SMS: {SMS_SHEET_URL.split('/d/')[1][:20]}...")
    
    refresh_rate = st.sidebar.selectbox(
        "Refresh Rate", 
        [5, 10, 30, 60], 
        index=1,
        format_func=lambda x: f"{x} seconds"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Placeholder for data
    placeholder = st.empty()
    
    def update_dashboard():
        with placeholder.container():
            # Fetch and process data
            with st.spinner("Fetching data from Google Sheets..."):
                bf_raw_data = get_sheet_data(BF_SHEET_URL, 0)
                sms_raw_data = get_sheet_data(SMS_SHEET_URL, 0)
                
            bf_data = process_bf_data(bf_raw_data)
            sms_data = process_sms_data(sms_raw_data)
            
            if not bf_data.empty and not sms_data.empty:
                # Date filtering section
                st.sidebar.header("Date Filter")
                
                # Get available dates from both datasets
                bf_dates = sorted(bf_data['DATE'].dropna().unique()) if 'DATE' in bf_data.columns else []
                sms_dates = sorted(sms_data['DATE'].dropna().unique()) if 'DATE' in sms_data.columns else []
                all_dates = sorted(set(bf_dates) | set(sms_dates))
                
                # Date selector
                date_options = ["All Dates"] + all_dates
                selected_date = st.sidebar.selectbox(
                    "Select Date",
                    options=date_options,
                    index=0
                )
                
                # Shift filtering section
                st.sidebar.header("Shift Filter")
                
                # Get available shifts from both datasets
                bf_shifts = sorted(bf_data['SHIFT'].dropna().unique()) if 'SHIFT' in bf_data.columns else []
                sms_shifts = sorted(sms_data['SHIFT'].dropna().unique()) if 'SHIFT' in sms_data.columns else []
                all_shifts = sorted(set(bf_shifts) | set(sms_shifts))
                
                # Shift selector
                shift_options = ["All Shifts"] + all_shifts
                selected_shift = st.sidebar.selectbox(
                    "Select Shift",
                    options=shift_options,
                    index=0
                )
                
                # Filter data by selected date and shift
                if selected_date != "All Dates":
                    bf_filtered = filter_data_by_date(bf_data, selected_date)
                    sms_filtered = filter_data_by_date(sms_data, selected_date)
                    date_display = f"for {selected_date}"
                else:
                    bf_filtered = bf_data
                    sms_filtered = sms_data
                    date_display = "(All Dates)"
                
                if selected_shift != "All Shifts":
                    bf_filtered = filter_data_by_shift(bf_filtered, selected_shift)
                    sms_filtered = filter_data_by_shift(sms_filtered, selected_shift)
                    shift_display = f"Shift {selected_shift}"
                else:
                    shift_display = "All Shifts"
                
                # Display data info with date and shift filter status
                st.success(f"✅ BF Data: {len(bf_filtered)} records {date_display} | {shift_display}")
                st.success(f"✅ SMS Data: {len(sms_filtered)} records {date_display} | {shift_display}")
                
                # Calculate metrics
                total_bf_sent = bf_filtered['NET WEIGHT'].sum() if 'NET WEIGHT' in bf_filtered.columns else 0
                total_sms_received = sms_filtered['NET WEIGHT'].sum() if 'NET WEIGHT' in sms_filtered.columns else 0
                in_transit = total_bf_sent - total_sms_received
                efficiency = (total_sms_received / total_bf_sent * 100) if total_bf_sent > 0 else 0
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Total Sent (BF)",
                        value=f"{total_bf_sent:,.0f} Tons",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Total Received (SMS)",
                        value=f"{total_sms_received:,.0f} Tons",
                        delta=None
                    )
                
                with col3:
                    delta_color = "normal" if in_transit >= 0 else "inverse"
                    st.metric(
                        label="In Transit",
                        value=f"{in_transit:,.0f} Tons",
                        delta=f"{in_transit:+,.0f} Tons",
                        delta_color=delta_color
                    )
                
                with col4:
                    efficiency_color = "normal" if efficiency >= 95 else "off"
                    st.metric(
                        label="Transfer Efficiency",
                        value=f"{efficiency:.1f}%",
                        delta=None
                    )
                
                # Create the main comparison chart
                st.subheader("📊 Weight Comparison: BF Net Weight vs SMS Pour Weights")
                comparison_chart = create_comparison_chart(bf_filtered, sms_filtered)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
                else:
                    st.info("No matching data found for comparison chart. Make sure LADLE NOs match between BF and SMS sheets.")
                
                # Efficiency chart
                efficiency_chart = create_efficiency_chart(bf_filtered, sms_filtered)
                if efficiency_chart:
                    st.subheader("📈 Transfer Efficiency by Ladle")
                    st.plotly_chart(efficiency_chart, use_container_width=True)
                
                # Recent transactions with date and shift columns
                st.subheader("📋 Recent Transactions")
                
                col7, col8 = st.columns(2)
                
                with col7:
                    st.write("**🏭 Blast Furnace - Recent Entries**")
                    bf_display_cols = ['DATE', 'SHIFT', 'ID', 'TAPNO', 'LADLE NO', 'NET WEIGHT']
                    available_bf_cols = [col for col in bf_display_cols if col in bf_filtered.columns]
                    display_bf = bf_filtered[available_bf_cols].tail(10)
                    st.dataframe(display_bf, width='stretch')
                
                with col8:
                    st.write("**🏗️ SMS - Recent Entries**")
                    sms_display_cols = ['DATE', 'SHIFT', 'ID', 'LADLE NO']
                    p_cols = [col for col in sms_filtered.columns if any(p in col for p in ['P1', 'P2', 'P3']) and 'WEIGHT' in col]
                    sms_display_cols.extend(p_cols)
                    sms_display_cols.append('NET WEIGHT')
                    
                    available_sms_cols = [col for col in sms_display_cols if col in sms_filtered.columns]
                    display_sms = sms_filtered[available_sms_cols].tail(10)
                    st.dataframe(display_sms, width='stretch')
                
                # Data summary
                with st.expander("📊 Data Summary"):
                    col9, col10 = st.columns(2)
                    with col9:
                        st.write("**BF Data Summary:**")
                        st.write(f"- Total Records: {len(bf_filtered)}")
                        st.write(f"- Unique Ladles: {bf_filtered['LADLE NO'].nunique()}")
                        st.write(f"- Average Net Weight: {bf_filtered['NET WEIGHT'].mean():.1f} tons" if 'NET WEIGHT' in bf_filtered.columns else "- No NET WEIGHT data")
                        if 'DATE' in bf_filtered.columns:
                            st.write(f"- Date Range: {bf_filtered['DATE'].min()} to {bf_filtered['DATE'].max()}")
                        if 'SHIFT' in bf_filtered.columns:
                            st.write(f"- Available Shifts: {', '.join(map(str, bf_filtered['SHIFT'].unique()))}")
                    
                    with col10:
                        st.write("**SMS Data Summary:**")
                        st.write(f"- Total Records: {len(sms_filtered)}")
                        st.write(f"- Unique Ladles: {sms_filtered['LADLE NO'].nunique()}")
                        st.write(f"- Average Net Weight: {sms_filtered['NET WEIGHT'].mean():.1f} tons" if 'NET WEIGHT' in sms_filtered.columns else "- No NET WEIGHT data")
                        if 'DATE' in sms_filtered.columns:
                            st.write(f"- Date Range: {sms_filtered['DATE'].min()} to {sms_filtered['DATE'].max()}")
                        if 'SHIFT' in sms_filtered.columns:
                            st.write(f"- Available Shifts: {', '.join(map(str, sms_filtered['SHIFT'].unique()))}")
                
            else:
                if bf_data.empty:
                    st.error("⚠️ Could not fetch Blast Furnace data. Please check:")
                    st.error("- Sheet is shared publicly (Anyone with link can view)")
                    st.error("- URL is correct")
                    st.error("- Sheet has data in the correct format")
                
                if sms_data.empty:
                    st.error("⚠️ Could not fetch SMS data. Please check:")
                    st.error("- Sheet is shared publicly (Anyone with link can view)")
                    st.error("- URL is correct")
                    st.error("- Sheet has data in the correct format")
            
            # Last updated timestamp
            st.markdown(f"*🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Initial dashboard update
    update_dashboard()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()