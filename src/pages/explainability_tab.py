import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

import torchvision.transforms as transforms

from src.utils.mlflow_utils import get_experiments, get_runs, load_model
from src.utils.gradcam import GradCAM


def overlay_heatmap(img, heatmap):

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * img + 0.5 * heatmap
    overlay = overlay / 255.0

    return overlay


def explainability_tab():

    st.title("Prediction + Grad-CAM + Upload Image")

    # ---------------- MLflow ---------------- #

    experiments = get_experiments()

    exp_name = st.selectbox(
        "Experiment",
        [e.name for e in experiments],
        key="exp_select"
    )

    exp = next(e for e in experiments if e.name == exp_name)

    runs = get_runs(exp.experiment_id)

    run_id = st.selectbox(
        "Run",
        runs["run_id"],
        key="run_select"
    )

    model = load_model(run_id)
    model.eval()

    # ---------------- Grad-CAM ---------------- #

    target_layer = model.model.conv2 if hasattr(model, "model") else model.conv2
    cam = GradCAM(model, target_layer)

    # ---------------- UPLOAD ---------------- #

    uploaded_file = st.file_uploader(
        "Upload Image (CIFAR-like)",
        type=["png", "jpg", "jpeg"]
    )

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        img_np = np.array(image.resize((32, 32)))

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

        st.image(image, caption="Uploaded Image", width=200)

        st.write(f"Predicted Class: {pred}")

        # ---------------- Grad-CAM ---------------- #

        heatmap = cam.generate(input_tensor, class_idx=pred)

        overlay = overlay_heatmap(img_np, heatmap)

        st.image(
            overlay,
            caption="Grad-CAM Heatmap",
            width=250
        )