import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return (x - 3) ** 2 + (y + 2) ** 2


def df_dx(x, y):
    return 2 * (x - 3)


def df_dy(x, y):
    return 2 * (y + 2)


def gradient_descent(x_init, y_init, learning_rate=0.05, max_iterations=500, tolerance=1e-10):
    x = float(x_init)
    y = float(y_init)

    history = {
        "x": [x],
        "y": [y],
        "loss": [f(x, y)]
    }

    for _ in range(max_iterations):
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        loss = f(x, y)
        history["x"].append(x)
        history["y"].append(y)
        history["loss"].append(loss)

        if len(history["loss"]) > 1:
            if abs(history["loss"][-1] - history["loss"][-2]) < tolerance:
                break

    return history, x, y


def create_plots(history):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131)
    ax1.plot(history["loss"], linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = fig.add_subplot(132)
    x_range = np.linspace(-1, 5, 100)
    y_range = np.linspace(-4, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(history["x"], history["y"], ".-", markersize=5, label="Path")
    ax2.plot(history["x"][0], history["y"][0], "o", markersize=8, label="Start")
    ax2.plot(3, -2, "*", markersize=14, label="Minimum")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Optimization Trajectory")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133, projection="3d")
    x_plot = np.linspace(-1, 5, 50)
    y_plot = np.linspace(-4, 2, 50)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    Z_plot = f(X_plot, Y_plot)

    ax3.plot_surface(X_plot, Y_plot, Z_plot, alpha=0.7)
    ax3.plot(history["x"], history["y"], history["loss"], ".-", markersize=4)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("f(x, y)")
    ax3.set_title("Surface and Optimization Path")

    plt.tight_layout()
    plt.savefig("q1_gradient_descent.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    x_init = 0
    y_init = 0
    learning_rate = 0.05
    max_iterations = 500

    history, x_final, y_final = gradient_descent(
        x_init=x_init,
        y_init=y_init,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )

    print("Question 1: Gradient Descent")
    print("Function: f(x, y) = (x - 3)^2 + (y + 2)^2")
    print("df/dx = 2(x - 3)")
    print("df/dy = 2(y + 2)")
    print(f"Initial point: ({x_init}, {y_init})")
    print(f"Final point: ({x_final:.6f}, {y_final:.6f})")
    print(f"Final loss: {f(x_final, y_final):.10f}")
    print("Theoretical minimum: (3, -2), f(x, y) = 0")

    create_plots(history)
    print("Saved plot: q1_gradient_descent.png")


if __name__ == "__main__":
    main()
