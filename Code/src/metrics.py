import matplotlib.pyplot as plt


def plot_activation(X1, X2, inverse_indices, n_samples,unique_activations):
    z = inverse_indices.numpy().reshape(n_samples, n_samples)

    # Create the 2D plot
    fig, ax = plt.subplots()

    # Contour plot
    contour = ax.contourf(
        X1, X2, z,
        cmap='viridis'
    )

    # Show the plot
    ax.set_title(f"Unique activations: {unique_activations.shape[0]}")
    plt.show()