import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import random
from utils import send_verification_email, is_token_valid, save_token_to_log, store_token_in_session, is_token_valid_session
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode


def safe_divide(numerator, denominator):
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0:
            return np.nan  # or "" for display
        return numerator / denominator
    except (ValueError, TypeError):
        return np.nan  # or ""

def calculate_total_patients_enrolled(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with the running total of enrolled patients.
    For 'Actual' rows, use 'Patients Enrolled'.
    For 'Projection' rows, use 'Patients to be Enrolled'.
    """
    cumulative = []
    total = 0
    for idx, row in df.iterrows():
        if row['Actual / Projection'] == 'Actual':
            value = pd.to_numeric(row.get('Patients Enrolled', 0), errors='coerce') or 0
        else:
            value = pd.to_numeric(row.get('Patients to be Enrolled', 0), errors='coerce') or 0
        total += value
        cumulative.append(total)
    return pd.Series(cumulative, index=df.index)

def get_month_list(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq='MS')

def create_step2_df(start_date, end_date):
    month_list = get_month_list(start_date, end_date)
    today_month = pd.Timestamp.today().replace(day=1)

    df = pd.DataFrame({"Month": month_list})
    df["Actual / Projection"] = df["Month"].apply(lambda m: "Actual" if m < today_month else "Projection")
    df["Sites Activated"] = 0
    df["Total Sites Activated"] = 0
    df["Patients Enrolled"] = 0
    df["Total Patients Enrolled (decimal)"] = 0

    # For display only: blank out "Patients Enrolled" in projections
    df["Patients Enrolled Display"] = df.apply(
        lambda row: row["Patients Enrolled"] if row["Actual / Projection"] == "Actual" else None,
        axis=1
    )

    return df

def update_cumulative_columns(df):
    df["Total Sites Activated"] = pd.to_numeric(df["Sites Activated"], errors="coerce").fillna(0).cumsum()
    df["Patients Enrolled"] = pd.to_numeric(df["Patients Enrolled"], errors="coerce").fillna(0)
    df["Total Patients Enrolled (decimal)"] = df["Patients Enrolled"].cumsum()
    return df

# ----------  CACHED ENROLLMENT FORECAST  ----------
# ----------  STEP-3 DATA BUILD (runs once per unique inputs)  ----------
@st.cache_data(show_spinner="⏳ Computing enrollment forecast …")
def build_step3_df(step2_df: pd.DataFrame,
                   global_psm: float,
                   enrollment_goal: int,
                   max_months: int = 240) -> pd.DataFrame:
    """
    Heavy, read-only calculation of all derived columns for Step 3.
    Streamlit reruns this only if the argument values change.
    """
    df = step2_df.copy().sort_values("Month").reset_index(drop=True)
    
    # 🆕  Ensure Month is Timestamp, not string
    df["Month"] = pd.to_datetime(df["Month"])
    df["Month"] = df["Month"].dt.floor("D")          # keep day=1 for safety


    # ------- ORIGINAL HEAVY BLOCK MOVED HERE -------
    df["Sites Activated"]   = pd.to_numeric(df["Sites Activated"]).fillna(0)
    df["Patients Enrolled"] = pd.to_numeric(df["Patients Enrolled"]).fillna(0)
    df["Total Sites Activated"] = df["Sites Activated"].cumsum()

    df["Patients to be Enrolled"] = 0.0
    for i in range(len(df)):
        if df.at[i, "Actual / Projection"] == "Projection":
            prev_sites = df.at[i-1, "Total Sites Activated"] if i > 0 else 0
            df.at[i, "Patients to be Enrolled"] = round(global_psm * prev_sites, 3)

    running_total = 0.0
    df["Total Patients Enrolled (decimal)"] = 0.0
    for i in range(len(df)):
        num = (df.at[i, "Patients Enrolled"]
               if df.at[i, "Actual / Projection"] == "Actual"
               else df.at[i, "Patients to be Enrolled"])
        running_total += num
        df.at[i, "Total Patients Enrolled (decimal)"] = round(running_total, 3)

    # Extend future months until goal reached (or cap)
    months_added = 0
    while df["Total Patients Enrolled (decimal)"].iloc[-1] < enrollment_goal and months_added < max_months:
        next_month   = (df["Month"].iloc[-1] + pd.DateOffset(months=1)).replace(day=1)
        prev_sites   = df["Total Sites Activated"].iloc[-1]
        pte          = round(global_psm * prev_sites, 3)
        running_total += pte
        df = pd.concat([df, pd.DataFrame({
            "Month":                     [next_month],
            "Actual / Projection":       ["Projection"],
            "Sites Activated":           [0],
            "Patients Enrolled":         [np.nan],
            "Total Sites Activated":     [prev_sites],
            "Patients to be Enrolled":   [pte],
            "Total Patients Enrolled (decimal)":[running_total],
        })], ignore_index=True)
        months_added += 1

    # Compute PSM
    df["PSM"] = None
    for i in range(len(df)):
        prev_sites = df.at[i-1, "Total Sites Activated"] if i > 0 else 0
        if prev_sites:
            num = (df.at[i, "Patients Enrolled"]
                   if df.at[i, "Actual / Projection"] == "Actual"
                   else df.at[i, "Patients to be Enrolled"])
            df.at[i, "PSM"] = round(num / prev_sites, 3)

    return df



st.set_page_config(page_title="Enrollment Projection App", layout="wide")

# Constants
ACCESS_LOG = "data/access_log.csv"
SUBMISSION_LOG = "data/user_submissions.csv"
SESSION_DURATION_HOURS = 24

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Session state init
def init_session():
    keys = {
        "step": 0,
        "verified": False,
        "user_name": "",
        "user_email": "",
        "token": "",
        "token_sent": False,
        "total_enrollment": None,
        "global_psm": None,
        "fsa_date": None,
        "lsa_date": None,
    }
    for key, value in keys.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()

# Step 0 – Email Verification
if st.session_state.step == 0:
    st.title("🔐 Enrollment Projection Tool")
    st.write("Please enter your name and email address to receive the Access Token.")

    with st.form("email_form"):
        name = st.text_input("Name", value=st.session_state.user_name)
        email = st.text_input("Email", value=st.session_state.user_email)
        submitted = st.form_submit_button("Send Verification Token")

    if submitted:
        if name and email:
            with st.spinner("Sending verification email..."):
                token = str(random.randint(100000, 999999))
                success = send_verification_email(name, email, token)
            if success:
                # save_token_to_log(email, token). ## <-- for local testing
                store_token_in_session(token)  ## <-- for streamlit.app deployment
                st.session_state.token = token
                st.session_state.user_email = email
                st.session_state.user_name = name
                st.session_state.token_sent = True
                st.success(f"Access Token sent to {email}. Enter it below.")
            else:
                st.error("Failed to send email.")
        else:
            st.warning("Enter both name and email.")

    if st.session_state.token_sent:
        # st.info("Enter the token received via email.")
        with st.form("verify_form"):
            token_input = st.text_input("Enter Access Token")
            verify_submitted = st.form_submit_button("Verify Token")
        if verify_submitted:
            # if is_token_valid(st.session_state.user_email, token_input, ACCESS_LOG, SESSION_DURATION_HOURS):
            if is_token_valid_session(token_input):
                st.session_state.verified = True
                st.session_state.step = 1
                st.rerun()
            else:
                st.error("Invalid or expired token.")

# Step 1 – Parameters Setup
elif st.session_state.step == 1:
    st.title("Step 1 – Input Parameters")
    st.markdown(
        """
        <p style='font-size:20px; text-align:left; margin:0;'>The following parameters are required to project enrollment:</p>
        <ul style='font-size:16px;'>
            <li><b>Total Enrollment Goal</b>: The total number of patients you aim to enroll in the study.</li>
            <li><b>Global PSM</b>: The patient per site per month enrollment rate expected globally (note: this can be derived from past studies if not available).</li>
            <li><b>First Site Activated Date</b>: The date when the first site is expected to be activated.</li>
            <li><b>Last Site Activated Date</b>: The date when the last site is expected to be activated.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.caption("(Contact alfonso.kondo@ctmx.io if you need help with these parameters.)")

    st.subheader("Enrollment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        total_enrollment = st.number_input("Total Enrollment Goal", min_value=1,
                                           value=st.session_state.total_enrollment or 1,
                                           step=1,
                                           help="Whole number only.")
    with col2:
        psm = st.number_input("Global PSM", min_value=0.001,
                              value=st.session_state.global_psm or 0.001,
                              step=0.001, format="%.3f")

    st.subheader("Site Activation Parameters")
    col3, col4 = st.columns(2)
    with col3:
        fsa_date = st.date_input("First Site Activated Date", value=st.session_state.fsa_date)
    with col4:
        lsa_date = st.date_input("Last Site Activated Date", value=st.session_state.lsa_date)

    if st.button("Next: Go to Step 2 ➡️"):
        if not float(total_enrollment).is_integer():
            st.error("Total Enrollment Goal must be a whole number.")
        elif fsa_date > lsa_date:
            st.error("First Site Activated Date cannot be after Last.")
        else:
            # Check if Step 1 inputs have changed vs snapshot
            current_snapshot = {
                "total_enrollment": int(total_enrollment),
                "global_psm": round(psm, 3),
                "fsa_date": fsa_date,
                "lsa_date": lsa_date
            }

            if "step1_snapshot" not in st.session_state or st.session_state.step1_snapshot != current_snapshot:
                # Step 1 changed → invalidate Step 2 cached data
                st.session_state.step2_df = None
                st.session_state.step1_snapshot = current_snapshot

            # Save values to session
            st.session_state.total_enrollment = int(total_enrollment)
            st.session_state.global_psm = round(psm, 3)
            st.session_state.fsa_date = fsa_date
            st.session_state.lsa_date = lsa_date

            st.session_state.step1_inputs = {
                "total_enrollment": int(total_enrollment),
                "global_psm": round(psm, 3),
                "fsa_date": fsa_date,
                "lsa_date": lsa_date
            }
            
            st.session_state.step = 2
            st.rerun()


# ---------------- STEP 2 ----------------
elif st.session_state.step == 2:
    # st.markdown("### 🏗️ Step 2 – Site Activation & Enrollment Inputs")
    st.title("Step 2 – Actual/Projected Site Activations & Enrollment")
    st.markdown(
        """
        <p style='font-size:20px; text-align:left; margin:0;'>As enrollment is driven by the number of sites activated in any given month, the site activation as well as the past enrollment information is required in the following table.</p>
        <ul style='font-size:16px;'>
            <li><u><strong>Sites Activated</strong> column</u>: enter the number of sites activated per for the past months (i.e., Actual) and the number of sites anticipated to be activated in the current/future months (i.e., Projection).</li>
            <li><u><strong>Patients Enrolled</strong> column</u>: enter the number of patients enrolled per month for the past months (i.e., Actual) to allow trending and correct placement of the projection curve.</li>
        </ul>
        <p style='font-size:16px; text-align:left; margin:0;'>NOTE: only completed months will be considered as "Actual", and current and future months will be considered as "Projection".</p>
        <br>
        """,
        unsafe_allow_html=True
    )

    # Define date range
    fsa = pd.to_datetime(st.session_state["fsa_date"]).replace(day=1)
    lsa = pd.to_datetime(st.session_state["lsa_date"]).replace(day=1)
    today = pd.Timestamp.today().normalize()
    current_month = today.replace(day=1)

    month_range = pd.date_range(start=fsa, end=max(lsa, current_month), freq="MS")

    # Initialize session df if needed
    if (
        "step2_df" not in st.session_state
        or not isinstance(st.session_state["step2_df"], pd.DataFrame)
        or not all(pd.to_datetime(st.session_state["step2_df"]["Month"]).dt.to_period("M") == month_range.to_period("M"))
    ):
        df = pd.DataFrame({"Month": month_range})
        df["Sites Activated"] = 0
        df["Actual / Projection"] = df["Month"].apply(lambda x: "Projection" if x >= current_month else "Actual")
        df["Patients Enrolled"] = df["Actual / Projection"].apply(lambda x: 0 if x == "Actual" else np.nan)
        st.session_state["step2_df"] = df.copy()

    # Load from session and re-compute logic
    df = st.session_state["step2_df"].copy()
    df["Month"] = pd.to_datetime(df["Month"]).dt.to_period("M").dt.to_timestamp()
    df["Actual / Projection"] = df["Month"].apply(lambda x: "Projection" if x >= current_month else "Actual")
    df["Patients Enrolled"] = [
        pd.to_numeric(val, errors="coerce") if proj == "Actual" else np.nan
        for val, proj in zip(df["Patients Enrolled"], df["Actual / Projection"])
    ]

    # Update session state with corrected columns
    st.session_state["step2_df"] = df.copy()

    # Enforce column order
    df = df[["Month", "Actual / Projection", "Sites Activated", "Patients Enrolled"]]

    # JS callbacks
    editable_patients = JsCode("""
        function(params) {
            return params.data["Actual / Projection"] === "Actual";
        }
    """)
    gray_if_projection = JsCode("""
        function(params) {
            if (params.data["Actual / Projection"] === "Projection") {
                return { color: 'black', backgroundColor: '#4a4a4a' };
            }
        }
    """)

    # AgGrid setup
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column(
        "Month", 
        header_name="Month", 
        editable=False, 
        type=["customDateTimeFormat"], 
        custom_format_string="MMM yy"
    )
    gb.configure_column(
        "Actual / Projection", 
        editable=False
    )
    gb.configure_column(
        "Sites Activated", 
        editable=True, 
        type=["numericColumn"],
        cellEditor="agTextCellEditor"
    )
    gb.configure_column(
        "Patients Enrolled", 
        editable=editable_patients, 
        type=["numericColumn"], 
        cellStyle=gray_if_projection,
        cellEditor="agTextCellEditor"
    )
    gb.configure_grid_options(forceFitColumns=True)
    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        use_container_width=True,
        fit_columns_on_grid_load=True,
        height=450,
        key=f"step2_grid_{hash(tuple(df['Month']))}"
    )

    if grid_response and "data" in grid_response:
        updated_df = pd.DataFrame(grid_response["data"])

        # Clean "Sites Activated"
        updated_df["Sites Activated"] = pd.to_numeric(updated_df["Sites Activated"], errors="coerce").fillna(0).astype(int)

        # Clean "Patients Enrolled" based on Actual/Projection
        updated_df["Patients Enrolled"] = [
            pd.to_numeric(val, errors="coerce") if proj == "Actual" else np.nan
            for val, proj in zip(updated_df["Patients Enrolled"], updated_df["Actual / Projection"])
        ]

        # Optional: enforce correct column order if needed
        expected_cols = ["Month", "Actual / Projection", "Sites Activated", "Patients Enrolled"]
        updated_df = updated_df[[col for col in expected_cols if col in updated_df.columns]]

        # Store entire DataFrame to session state (fix)
        st.session_state["step2_df"] = updated_df.copy()

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back: to Step 1"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Generate: Enrollment Forecast\u00A0\u00A0➡️"):
            st.session_state.step = 3
            st.rerun()


# ---------------- STEP 3 ----------------
elif st.session_state.step == 3:

    st.title("Enrollment Forecast Summary")

    if "step2_df" not in st.session_state:
        st.warning("Step 2 data not found. Please return to Step 2 to input site and enrollment data.")
        if st.button("⬅️ Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    
    # --- build/cached heavy dataframe ------------------------
    global_psm      = st.session_state.get("global_psm", 0.0)
    enrollment_goal = st.session_state.get("total_enrollment", 0)
    full_df = build_step3_df(
        st.session_state["step2_df"],
        global_psm,
        enrollment_goal
    )

    # --- table_df: static table (never affected by slider) ---
    table_df = full_df.copy().reset_index(drop=True)
    table_df["Month"] = table_df["Month"].dt.strftime("%b %Y")
    table_df["Total Patients Enrolled"] = (
        np.floor(table_df["Total Patients Enrolled (decimal)"])
          .astype(int)
    )
    table_df = table_df[
        [
            "Month",
            "Actual / Projection",
            "Sites Activated",
            "Total Sites Activated",
            "Patients Enrolled",
            "Patients to be Enrolled",
            "Total Patients Enrolled",
            "PSM",
        ]
    ]

    # Render the static table BEFORE the slider
    st.subheader("Enrollment Forecast (Full Data)")
    AgGrid(
        table_df,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
    )


    # --- Month slider (now filters the *pre-computed* dataframe) ---
    min_month = full_df["Month"].min().to_pydatetime()
    max_month = full_df["Month"].max().to_pydatetime()

    month_range = st.slider(
        "Select Month Range:",
        min_value=min_month,
        max_value=max_month,
        value=(min_month, max_month),
        format="MMM YYYY"
    )

    # df = full_df.loc[
    #     (full_df["Month"] >= pd.to_datetime(month_range[0])) &
    #     (full_df["Month"] <= pd.to_datetime(month_range[1]))
    # ].copy()

    # df = df.reset_index(drop=True)      # ensures labels 0…N-1

    # --- Month slider (filtered only for plots) ---
    filtered_df = full_df.loc[
        (full_df["Month"] >= pd.to_datetime(month_range[0])) &
        (full_df["Month"] <= pd.to_datetime(month_range[1]))
    ].reset_index(drop=True)

    # prepare Month & Total Enrolled for plotting
    filtered_df["Month"] = filtered_df["Month"].dt.strftime("%b %Y")
    filtered_df["Total Patients Enrolled"] = (
        np.floor(filtered_df["Total Patients Enrolled (decimal)"])
          .astype(int)
    )


    # Format Month for display
    filtered_df["Month"] = filtered_df["Month"].dt.strftime("%b %Y")

    # Round down cumulative decimal to create integer display version
    filtered_df["Total Patients Enrolled"] = np.floor(
        filtered_df["Total Patients Enrolled (decimal)"]
    ).astype(int)

    # Define display columns in the requested order
    display_df = filtered_df[
        [
            "Month",
            "Actual / Projection",
            "Sites Activated",
            "Total Sites Activated",
            "Patients Enrolled",
            "Patients to be Enrolled",
            "Total Patients Enrolled",  
            # "Total Patients Enrolled (decimal)", 
            "PSM"
        ]
    ]

    # Define JS formatters
    psm_formatter = JsCode("""
        function(params) {
            return (params.value === null || params.value === undefined) ? '' : params.value.toFixed(3);
        }
    """)

    patients_enrolled_formatter = JsCode("""
        function(params) {
            if (params.data["Actual / Projection"] === "Projection" && params.value === 0) {
                return '';
            }
            return params.value;
        }
    """)

    patients_to_be_enrolled_formatter = JsCode("""
        function(params) {
            if (params.data["Actual / Projection"] === "Actual" && params.value === 0) {
                return '';
            }
            return params.value;
        }
    """)

    # Build Grid Options
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_column("Patients Enrolled", valueFormatter=patients_enrolled_formatter)
    gb.configure_column("Patients to be Enrolled", valueFormatter=patients_to_be_enrolled_formatter)
    gb.configure_column("PSM", valueFormatter=psm_formatter)
    gb.configure_grid_options(domLayout='normal')
    # gb.configure_grid_options(domLayout='autoHeight')


    # ------------- DASHBOARD METRICS (FIXED) ------------- #
    if not display_df.empty:
        total_months = len(display_df)
        fsa_month = st.session_state.step1_inputs['fsa_date'].strftime('%b %Y')
        lpi_month = display_df["Month"].max()
        actual_sites = display_df[display_df["Actual / Projection"] == "Actual"]["Sites Activated"].sum()
        projected_sites = display_df[display_df["Actual / Projection"] == "Projection"]["Sites Activated"].sum()
        total_sites = actual_sites + projected_sites
        percent_activated = (actual_sites / total_sites) * 100 if total_sites > 0 else 0
        patients_enrolled = display_df[display_df["Actual / Projection"] == "Actual"]["Patients Enrolled"].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
        enrollment_goal = st.session_state.step1_inputs.get("total_enrollment", 1)
        pending_enrollment = enrollment_goal - patients_enrolled
        percent_enrolled = (patients_enrolled / enrollment_goal)*100 if enrollment_goal else 0

        st.markdown(
            f"<p style='font-size:20px; text-align:left; margin:0;'><strong>Projected Enrollment Period:</strong> {fsa_month} to {lpi_month} ({total_months} months)</p>",
            unsafe_allow_html=True
        )

    # ---------- DONUT CHARTS FOR STEP 3 DASHBOARD ---------- #
    total_sites = actual_sites + projected_sites
    pending_enrollment = max(enrollment_goal - patients_enrolled, 0)

    # Site Activation Donut
    fig_sites = go.Figure(data=[go.Pie(
        labels=["Activated", "Pending"],
        values=[actual_sites, projected_sites],
        hole=0.85,
        marker=dict(colors=["steelblue", "#B6B6B6"]),
        # marker=dict(colors=["#636EFA", "#B6B6B6"]),
        textinfo="label+value+percent"
    )])
    fig_sites.update_layout(
        showlegend=False,
        annotations=[
            dict(text=str(total_sites), x=0.5, y=0.56, font_size=44, showarrow=False),
            dict(text="Site Activation Goal", x=0.5, y=0.30, font_size=12, showarrow=False)
        ],
        margin=dict(t=40, b=40, l=40, r=40),
        height=250
    )

    # Patient Enrollment Donut
    fig_enroll = go.Figure(data=[go.Pie(
        labels=["Enrolled", "Remaining"],
        values=[patients_enrolled, pending_enrollment],
        hole=0.85,
        marker=dict(colors=["#24C354", "#B6B6B6"]),
        textinfo="label+value+percent"
    )])
    fig_enroll.update_layout(
        showlegend=False,
        annotations=[
            dict(text=str(int(enrollment_goal)), x=0.5, y=0.56, font_size=44, showarrow=False),
            dict(text="Enrollment Goal", x=0.5, y=0.30, font_size=12, showarrow=False)
        ],
        margin=dict(t=40, b=40, l=40, r=40),
        height=250
    )

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(fig_sites, use_container_width=True)
    with col_chart2:
        st.plotly_chart(fig_enroll, use_container_width=True)

    # Filter data
    # actual_df = display_df[display_df["Actual / Projection"] == "Actual"]
    # projected_df = display_df[display_df["Actual / Projection"] == "Projection"]
    actual_df    = filtered_df[filtered_df["Actual / Projection"] == "Actual"]
    projected_df = filtered_df[filtered_df["Actual / Projection"] == "Projection"]

    # Create traces
    fig = go.Figure()

    # Activation – Actual (Bar)
    fig.add_trace(go.Bar(
        x=actual_df["Month"],
        y=actual_df["Total Sites Activated"],
        name="Activation – Actual",
        marker_color="steelblue"
    ))

    # Activation – Projected (Bar with pattern)
    fig.add_trace(go.Bar(
        x=projected_df["Month"],
        y=projected_df["Total Sites Activated"],
        name="Activation – Projected",
        marker=dict(
            # color="lightgray",
            color="steelblue",
            pattern=dict(shape="/")  # diagonal stripes
        )
    ))

    # Enrollment – Projected (Dotted Line)
    fig.add_trace(go.Scatter(
        x=display_df["Month"],
        y=display_df["Total Patients Enrolled"],
        name="Enrollment – Projected",
        mode="lines+markers",
        # line=dict(color="#24C354", dash="dot"),
        # line=dict(color="#00CC96", dash="dot"),
        line=dict(color="#24C354", dash="dot", width=3.5),
        marker=dict(size=9)
    ))

    # Enrollment – Actual (Line)
    fig.add_trace(go.Scatter(
        x=actual_df["Month"],
        y=actual_df["Total Patients Enrolled"],
        name="Enrollment – Actual",
        mode="lines+markers",
        # line=dict(color="#00CC96"),
        line=dict(color="#00CC96", width=3.5),
        marker=dict(size=9)
    ))

    # Layout
    fig.update_layout(
        title="Site Activation & Patient Enrollment Over Time",
        barmode="stack",
        xaxis=dict(
            tickangle=315, 
            automargin=True,
        ),
        # yaxis_title="Count",
        legend_title="Legend",
        legend=dict(
            orientation="h",         # horizontal layout
            yanchor="top",
            y=-0.3,                  # place below the x-axis labels
            xanchor="center",
            x=0.5,                   # center it horizontally
            title=None              # hide legend title (optional)
        ),
        height=500
    )

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    # --------- PSM Step Chart ---------
    fig_psm = go.Figure()

    fig_psm.add_trace(go.Scatter(
        x=display_df["Month"],
        y=display_df["PSM"],
        mode="lines+markers",
        name="PSM",
        line_shape="hv",  # step-wise chart
        line=dict(color="grey", width=3),
        # line=dict(color="#79688A", width=3),
        marker=dict(size=6)
    ))

    fig_psm.update_layout(
        title="Monthly Enrollment Rate (PSM) Over Time",
        xaxis=dict(tickangle=315),
        height=200,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )

    st.plotly_chart(fig_psm, use_container_width=True)


    # Render grid
    # AgGrid(
    #     display_df,
    #     gridOptions=gb.build(),
    #     allow_unsafe_jscode=True,
    #     update_mode=GridUpdateMode.NO_UPDATE,
    #     height=450,
    #     use_container_width=True,
    #     fit_columns_on_grid_load=True, 
    # )


    show_table = st.toggle("📄 Show Enrollment Forecast Data Table", value=False)

    if show_table:
        # --- Download buttons ---
        csv = display_df.to_csv(index=False).encode("utf-8")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            display_df.to_excel(writer, sheet_name="Data", index=False)
            # writer.save()
        excel_data = excel_buffer.getvalue()

        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name="enrollment_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_xlsx:
            st.download_button(
                label="⬇️ Download as Excel",
                data=excel_data,
                file_name="enrollment_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        AgGrid(
            display_df,
            gridOptions=gb.build(),
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.NO_UPDATE,
            height=450,
            use_container_width=True,
            fit_columns_on_grid_load=True,
        )


    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back: to Step 2"):
            st.session_state.step = 2
            st.rerun()
