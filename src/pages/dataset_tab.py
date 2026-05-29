import streamlit as st
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter


def dataset_tab():

    st.header("📊 Dataset Exploration (CIFAR-10)")

    transform = transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=False,
        transform=transform
    )

    classes = dataset.classes
    labels = dataset.targets

    # ---------------- STATS ---------------- #

    st.subheader("Dataset Statistics")

    st.write(f"Number of samples: {len(dataset)}")
    st.write(f"Number of classes: {len(classes)}")

    split_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Samples": [30000, 10000, 10000]
    })

    st.dataframe(split_df)

    # ---------------- DISTRIBUTION ---------------- #

    counts = Counter(labels)

    fig, ax = plt.subplots()

    ax.bar(classes, [counts[i] for i in range(len(classes))])
    ax.set_title("Class Distribution")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ---------------- FILTER ---------------- #

    selected_class = st.selectbox(
        "Filter by class",
        ["All"] + classes,
        key="dataset_filter"
    )

    if selected_class == "All":
        filtered_indices = list(range(len(dataset)))
    else:
        class_idx = classes.index(selected_class)
        filtered_indices = [i for i, l in enumerate(labels) if l == class_idx]

    if len(filtered_indices) == 0:
        st.warning("No samples found")
        return

    sample_position = st.slider(
        "Sample index",
        0,
        len(filtered_indices) - 1,
        0,
        key="dataset_slider"
    )

    idx = filtered_indices[sample_position]

    image, label = dataset[idx]

    image_np = image.permute(1, 2, 0).cpu().numpy()

    st.image(
        image_np,
        width=250,
        caption=f"Class: {classes[label]}"
    )

    st.write("### Sample Info")
    st.write(f"Label index: {label}")
    st.write(f"Class name: {classes[label]}")