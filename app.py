# app.py

import streamlit as st
import os
import tempfile
from PIL import Image
from main import generate_images  # Ensure main.py is in the same directory

st.set_page_config(layout="wide")  # Set layout to wide for a better UI experience

st.title("Diffusion Posterior Sampling UI")

# Sidebar for parameter inputs
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", "stabilityai/stable-diffusion-2-base")
    scale = st.slider("Scale", 1.0, 10.0, 4.8)
    algo = st.selectbox("Algorithm", ["dps", "dsg"])
    operator = st.selectbox("Operator", ["srx8", "gdb", "mdb"])
    nstep = st.number_input("Number of Steps", value=500, step=1)
    fdm_c1 = st.number_input("FreeDOM c1", value=100, step=1)
    fdm_c2 = st.number_input("FreeDOM c2", value=250, step=1)
    fdm_k = st.number_input("FreeDOM k", value=2, step=1)
    psld_gamma = st.slider("PSLD Gamma", 0.01, 1.0, 0.1)
    prompt = st.text_input("Prompt", "Enter your prompt here")

# Main area for image input and output
uploaded_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Generate Images"):
    if uploaded_images:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            for uploaded_image in uploaded_images:
                # Save each uploaded image to the temporary directory
                image_path = os.path.join(tmpdir, uploaded_image.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                # Call the generate function for each image
                generate_images(
                    model=model,
                    data_path=tmpdir,
                    out_path=output_dir,
                    scale=scale,
                    algo=algo,
                    operator=operator,
                    nstep=nstep,
                    fdm_c1=fdm_c1,
                    fdm_c2=fdm_c2,
                    fdm_k=fdm_k,
                    psld_gamma=psld_gamma,
                    prompt=prompt
                )

            # Display results
            for uploaded_image in uploaded_images:
                st.write(f"**Results for {uploaded_image.name}**")
                col1, col2 = st.columns(2)  # Define two columns for output images
                with col1:
                    st.image(uploaded_image, caption="Uploaded Image")

                with col2:
                    for filename in ["source", "low_res", "recon", "recon_low_res"]:
                        image_file = os.path.join(output_dir, filename, uploaded_image.name)
                        if os.path.exists(image_file):
                            st.write(f"Results from {filename}:")
                            st.image(Image.open(image_file), caption=filename)

            st.success("Generation completed!")

    else:
        st.warning("Please upload at least one image before generating.")