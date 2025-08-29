# services/frontend/app.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="FloatChat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid transparent;
        color: #ffffff;
        background-color: #0068c9;
    }
</style>
""", unsafe_allow_html=True)


# --- API Configuration ---
API_URL = "http://api:8000/query" # 'api' is the service name in docker-compose

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.clipartmax.com/png/full/204-2045059_ocean-waves-wave-logo-sea-water-tsunami-ocean-wave-logo-png.png", width=100)
    st.title("🌊 FloatChat")
    st.markdown("Welcome to FloatChat, your AI-powered conversational interface for exploring ARGO ocean data.")
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1.  **Ask a question** in the chat box.
    2.  **Get an answer** powered by AI.
    3.  **Explore** the data through interactive maps and tables.
    """)
    st.markdown("---")
    st.markdown("### Example Questions:")
    st.info("""
    - "Show me the 5 most recent argo profiles."
    - "Find all floats launched in 2023."
    - "What are the locations of profiles taken after January 1, 2024?"
    - "Show salinity and temperature for profile_id 15."
    """)

# --- Main Chat Interface ---
st.title("ARGO Ocean Data Explorer")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you explore the ARGO float data today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message and message["data"]:
            df = pd.DataFrame(message["data"])
            
            # Display map if lat/lon data is available
            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.subheader("Geospatial Visualization")
                fig_map = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="profile_id",
                                            color_discrete_sequence=["#0068c9"], zoom=1, height=400)
                fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)

            # Display profile plot if temp/psal/pres data is available
            if all(k in df.columns for k in ['temp', 'psal', 'pres']):
                st.subheader("Profile Data")
                # Handle single vs multiple profiles
                profile_ids = df['profile_id'].unique()
                selected_id = st.selectbox("Select a Profile to view:", profile_ids)
                
                profile_data = df[df['profile_id'] == selected_id].iloc[0]
                
                fig_profile = go.Figure()
                # Temperature vs Pressure
                fig_profile.add_trace(go.Scatter(x=profile_data['temp'], y=profile_data['pres'], mode='lines+markers', name='Temperature (°C)'))
                # Salinity vs Pressure
                fig_profile.add_trace(go.Scatter(x=profile_data['psal'], y=profile_data['pres'], mode='lines+markers', name='Salinity (PSU)', xaxis='x2'))
                
                fig_profile.update_layout(
                    title=f"Temperature & Salinity Profile for ID: {selected_id}",
                    yaxis=dict(title='Pressure (dbar)', autorange='reversed'),
                    xaxis=dict(title='Temperature (°C)'),
                    xaxis2=dict(title='Salinity (PSU)', overlaying='x', side='top'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_profile, use_container_width=True)

            # Display data table
            st.subheader("Tabular Data")
            st.dataframe(df)

        if "sql" in message and message["sql"]:
            with st.expander("Generated SQL Query"):
                st.code(message["sql"], language="sql")


# Accept user input
if prompt := st.chat_input("Ask about ARGO floats..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Call the backend API
            response = requests.post(API_URL, json={"question": prompt})
            response.raise_for_status()  # Raise an exception for bad status codes
            api_response = response.json()

            # Display the final response
            message_placeholder.markdown(api_response["natural_language_response"])

            # Store the full response in session state
            assistant_message = {
                "role": "assistant",
                "content": api_response["natural_language_response"],
                "sql": api_response.get("sql_query"),
                "data": api_response.get("data")
            }
            st.session_state.messages.append(assistant_message)
            
            # Rerun to display the new data visualizations
            st.rerun()

        except requests.exceptions.RequestException as e:
            error_message = f"Could not connect to the backend API. Please ensure it is running. Error: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
