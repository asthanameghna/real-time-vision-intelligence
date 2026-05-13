import streamlit

streamlit.title("Real-Time Vision Intelligence Dashboard")
streamlit.write("Dashboard is running")

streamlit.subheader("Live Stream")
streamlit.markdown(
    '<img src="http://127.0.0.1:8000/stream" style="max-width:100%;" />',
    unsafe_allow_html=True,
)
