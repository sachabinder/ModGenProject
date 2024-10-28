# app.py

import streamlit as st
import os
import tempfile
from PIL import Image
from main import generate_images  # Ensure main.py is in the same directory

st.title("Diffusion Posterior Sampling UI")

# User input options for parameters
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

uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if st.button("Generate Image"):
    if uploaded_image is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded image to the temporary directory
            image_path = os.path.join(tmpdir, "input_image.png")
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Define output directory within the temporary directory
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            # Call the generate function
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

            # Load and display generated images
            st.write("Processing...")
            for filename in ["source", "low_res", "recon", "recon_low_res"]:
                image_file = os.path.join(output_dir, filename, "input_image.png")
                if os.path.exists(image_file):
                    st.write(f"Results from {filename}:")
                    st.image(Image.open(image_file), caption=filename)

            st.write("Generation completed!")

    else:
        st.warning("Please upload an image before generating.")