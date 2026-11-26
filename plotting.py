import matplotlib.pyplot as plt


def plot_depth(grid, state):

    plt.figure(figsize=(7, 5))

    plt.contourf(
        grid.X,
        grid.Y,
        state.D,
        levels=30
    )

    plt.colorbar(label="Fluid depth D")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fluid depth at final time")

    plt.tight_layout()
    plt.show()