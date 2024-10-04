import streamlit as st
import requests
import json
import sseclient
import time
from streamlit_lottie import st_lottie

# Set page config
st.set_page_config(page_title="Research Dashboard", layout="wide", page_icon="🔬")


# Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_research = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_tno6cg2w.json"
)


def stream_research(inputs, progress_bar, result_container):
    url = "http://localhost:8000/research"
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}

    try:
        with requests.post(url, json=inputs, headers=headers, stream=True) as response:
            client = sseclient.SSEClient(response)
            full_content = ""
            for event in client.events():
                data = json.loads(event.data)
                if "markdown_chunk" in data:
                    full_content += data["markdown_chunk"]
                    result_container.markdown(full_content, unsafe_allow_html=True)
                    progress_bar.progress(
                        min(len(full_content) / 5000, 1.0)
                    )  # Adjust denominator as needed
                elif data.get("status") == "complete":
                    progress_bar.progress(1.0)
                    st.success("Research complete!")
                    break
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    st.title("🔬 Research Dashboard")

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")
        with st.form("research_form"):
            product = st.text_input("Product", "Nothing Phone 2A")
            region = st.text_input("Region", "India")

            # Advanced Options Expander
            with st.expander("Advanced Options"):
                target_market = st.text_input(
                    "Target Market", "Tech-savvy consumers aged 18-35"
                )
                competitors = st.text_input("Competitors", "Apple, Samsung, Xiaomi")
                pricing_strategy = st.text_input(
                    "Pricing Strategy", "Value-based pricing"
                )
                context = st.text_area(
                    "Context",
                    "The product is a new smartphone model with advanced camera features and long battery life.",
                )

            submit_button = st.form_submit_button("Start Research")

        if submit_button:
            inputs = {
                "product": product,
                "region": region,
                "target_market": target_market,
                "competitors": competitors,
                "pricing_strategy": pricing_strategy,
                "context": context,
            }

            # Display the Lottie animation
            with st.spinner("Initiating research..."):
                st_lottie(lottie_research, height=200, key="research_animation")
                time.sleep(2)  # Simulate initialization time

            progress_bar = st.progress(0)
            result_container = col2.empty()

            stream_research(inputs, progress_bar, result_container)


if __name__ == "__main__":
    main()
