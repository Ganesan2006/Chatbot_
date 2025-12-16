import streamlit as st
from huggingface_hub import InferenceClient

# Set your API key and model
HF_API_KEY = os.environ["HF_TOKEN"]
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

client = InferenceClient(provider="auto", api_key=HF_API_KEY)

# Streamlit UI
st.set_page_config(page_title="LLM Chat", layout="centered")
st.title("🤖 LLM Chat with Hugging Face")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Generate Response"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = completion.choices[0].message["content"]
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
