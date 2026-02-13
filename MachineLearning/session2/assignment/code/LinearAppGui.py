import tkinter as tk
import numpy as np
from ClassLinear import LinearRegressionGD


class LinearRegressionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AMIT - Machine Learning Diploma")
        self.root.geometry("700x500")

        # -----------------------------
        # Dataset
        # -----------------------------
        self.X = np.array([50, 60, 70, 80, 90]).reshape(-1, 1)
        self.y = np.array([150, 180, 210, 240, 270])

        # -----------------------------
        # Train Model
        # -----------------------------
        self.model = LinearRegressionGD(
            learning_rate=0.000001,
            n_iters=10000,
            normalize=False
        )
        self.model.fit(self.X, self.y)

        self.create_widgets()

    def create_widgets(self):

        # =============================
        # Header
        # =============================
        header = tk.Label(
            self.root,
            text=" Price of house predict",
            bg="blue",
            fg="white",
            font=("Arial", 24, "bold")
        )
        header.pack(fill=tk.X)

        # =============================
        # Sidebar
        # =============================
        sidebar = tk.Frame(self.root, bg="lightgrey", width=200)
        sidebar.pack(fill=tk.Y, side=tk.LEFT)

        project_labels = [
            "Linear Regression",
            "Show Parameters",
            "Show Plot"
        ]

        for label in project_labels:
            lbl = tk.Label(
                sidebar,
                text=label,
                bg="lightgrey",
                anchor="w",
                padx=15,
                font=("Arial", 14)
            )
            lbl.pack(fill=tk.X, padx=5, pady=5)

        # =============================
        # Main Section
        # =============================
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=40)

        title = tk.Label(
            main_frame,
            text="House Price Prediction",
            font=("Arial", 22)
        )
        title.pack(pady=10)

        # Input
        label = tk.Label(
            main_frame,
            text="Enter House Size (m²):",
            font=("Arial", 16)
        )
        label.pack()

        self.size_entry = tk.Entry(main_frame, font=("Arial", 16))
        self.size_entry.pack()

        # Predict Button
        execute_button = tk.Button(
            main_frame,
            text="Predict",
            command=self.predict_price,
            bg="grey",
            fg="black",
            font=("Arial", 16)
        )
        execute_button.pack(pady=10)

        # Show Parameters Button
        param_button = tk.Button(
            main_frame,
            text="Show Model Parameters",
            command=self.show_parameters,
            bg="orange",
            font=("Arial", 14)
        )
        param_button.pack(pady=5)

        # Show Plot Button
        plot_button = tk.Button(
            main_frame,
            text="Show Training Plot",
            command=self.show_plot,
            bg="lightblue",
            font=("Arial", 14)
        )
        plot_button.pack(pady=5)

        # Result Label
        self.result_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 18, "bold")
        )
        self.result_label.pack(pady=20)

    # =============================
    # Prediction
    # =============================
    def predict_price(self):

        try:
            size = float(self.size_entry.get())

            prediction = self.model.predict([[size]])

            self.result_label.config(
                text=f"Predicted Price: {prediction[0]:.2f} (thousands)",
                fg="green"
            )

        except ValueError:
            self.result_label.config(
                text="Please enter a valid number",
                fg="red"
            )

    # =============================
    # Show Parameters
    # =============================
    def show_parameters(self):

        self.result_label.config(
            text=f"θ0 (Bias): {self.model.bias:.4f}\nθ1 (Weight): {self.model.weight[0]:.4f}",
            fg="black"
        )

    # =============================
    # Show Plot
    # =============================
    def show_plot(self):
        self.model.plot_training(self.X, self.y)


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()
