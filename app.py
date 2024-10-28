import streamlit as st
import os
from main import run_pipeline  # Import the function from main.py
from PIL import Image

# Set up the main page
st.title("Diffusion Posterior Sampling UI")
st.write("Upload an image, adjust parameters, and view the generated results.")

# Sidebar for parameters
model = st.sidebar.text_input("Model", "stabilityai/stable-diffusion-2-base")
scale = st.sidebar.slider("Scale", min_value=1.0, max_value=10.0, value=4.8)
algo = st.sidebar.selectbox("Algorithm", ["dps", "dsg"])
operator = st.sidebar.selectbox("Operator", ["srx8", "gdb", "mdb"])
nstep = st.sidebar.number_input("Number of Steps", min_value=100, max_value=1000, value=500)
fdm_c1 = st.sidebar.number_input("FreeDOM c1", min_value=1, max_value=500, value=100)
fdm_c2 = st.sidebar.number_input("FreeDOM c2", min_value=1, max_value=500, value=250)
fdm_k = st.sidebar.number_input("FreeDOM k", min_value=1, max_value=10, value=2)
psld_gamma = st.sidebar.slider("PSLD Gamma", min_value=0.01, max_value=1.0, value=0.1)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

# Directory for temporary storage
output_folder = "./temp_output"
os.makedirs(output_folder, exist_ok=True)

# Run pipeline when image is uploaded and button is clicked
if uploaded_file:
    input_image_path = os.path.join(output_folder, uploaded_file.name)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(input_image_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Image"):
        st.write("Processing...")

        # Run the pipeline
        output_dirs = run_pipeline(
            input_image_path=output_folder,
            output_folder=output_folder,
            model=model,
            scale=scale,
            algo=algo,
            operator=operator,
            nstep=nstep,
            fdm_c1=fdm_c1,
            fdm_c2=fdm_c2,
            fdm_k=fdm_k,
            psld_gamma=psld_gamma
        )
        print("output_folder: {}".format(output_folder))

        # Display the results
        for folder_name in output_dirs:
            st.write(f"Results from {folder_name.split('/')[-1]}:")
            for img_file in os.listdir(folder_name):
                img_path = os.path.join(folder_name, img_file)
                image = Image.open(img_path)
                st.image(image, caption=f"{folder_name}", use_column_width=True)