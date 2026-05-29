import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.utils.mlflow_utils import get_experiments, get_runs, load_model


def error_analysis_tab():

    st.header("Error Analysis")

    experiments = get_experiments()

    experiment_names = [exp.name for exp in experiments]

    selected_exp = st.selectbox(
        "Experiment",
        experiment_names,
        key="error_exp"
    )

    experiment = next(e for e in experiments if e.name == selected_exp)

    runs = get_runs(experiment.experiment_id)

    if len(runs) == 0:
        st.warning("No runs found")
        return

    selected_run = st.selectbox(
        "Run",
        runs["run_id"],
        key="error_run"
    )

    if not st.button("Analyze Errors"):
        return

    model = load_model(selected_run)
    model.eval()

    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=False,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_pred, confidence_list = [], [], []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            confidence, pred = torch.max(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())
            confidence_list.extend(confidence.numpy())

    # ---------------- CONFUSION MATRIX ---------------- #

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, cmap="Blues", ax=ax)

    st.pyplot(fig)

    # ---------------- MISCLASSIFIED ---------------- #

    st.subheader("Misclassified Samples")

    shown = 0

    for idx in range(len(dataset)):

        if y_true[idx] != y_pred[idx]:

            image, _ = dataset[idx]

            image = image.permute(1, 2, 0).numpy()

            st.image(image, width=120)

            st.write(f"True: {dataset.classes[y_true[idx]]}")
            st.write(f"Predicted: {dataset.classes[y_pred[idx]]}")
            st.write(f"Confidence: {confidence_list[idx]:.3f}")

            st.divider()

            shown += 1

        if shown == 10:
            break