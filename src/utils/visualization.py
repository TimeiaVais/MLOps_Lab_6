import matplotlib.pyplot as plt


def plot_probabilities(
        probs
):

    fig, ax = plt.subplots()

    ax.bar(
        range(len(probs)),
        probs
    )

    return fig