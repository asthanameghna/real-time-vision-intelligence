import pandas as pd
import requests
import streamlit

streamlit.title("Real-Time Vision Intelligence Dashboard")
streamlit.write("Dashboard is running")

streamlit.subheader("Live Stream")
streamlit.markdown(
    '<img src="http://127.0.0.1:8000/stream" style="max-width:100%;" />',
    unsafe_allow_html=True,
)

streamlit.subheader("Runtime Metrics")
try:
    _m = requests.get("http://127.0.0.1:8000/metrics", timeout=2).json()
except Exception as exc:
    streamlit.error(f"metrics unavailable: {exc}")
else:
    streamlit.write(f"processed_frames: {_m.get('processed_frames')}")
    streamlit.write(f"fps: {_m.get('fps')}")
    streamlit.write(f"active_tracks: {_m.get('active_tracks')}")
    streamlit.write(f"total_events: {_m.get('total_events')}")

streamlit.subheader("Recent Events")
try:
    _e = requests.get("http://127.0.0.1:8000/events", timeout=2).json()
except Exception as exc:
    streamlit.error(f"events unavailable: {exc}")
else:
    streamlit.table(pd.DataFrame(_e))
