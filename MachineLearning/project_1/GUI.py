# ================================
# Import Required Libraries
# ================================

import tkinter as tk
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class BostonHousingApp:
    """
    BostonHousingApp creates a graphical user interface (GUI)
    that predicts Boston housing prices using a Decision Tree
    Regressor machine learning model.

    The application performs the following tasks:
    - Loads the housing dataset
    - Trains a Decision Tree regression model
    - Allows the user to input house features
    - Predicts the house price based on those features
    - Displays a model complexity plot
    """

    def __init__(self, root):
        """
        Initialize the application.

        Parameters
        ----------
        root : Tk
            The main Tkinter window.

        This method:
        - Loads the dataset
        - Extracts features and target values
        - Trains the Decision Tree model
        - Saves the trained model
        - Builds the graphical interface
        """

        self.root = root
        self.root.title("Boston Housing Prediction")
        self.root.geometry("750x500")

        # =====================
        # Load Dataset
        # =====================

        data = pd.read_csv("MachineLearning/project_1/housing.csv")

        # Target variable (house prices)
        self.prices = data['MEDV']

        # Selected features used for prediction
        self.features = data[['RM', 'LSTAT', 'PTRATIO']]

        # =====================
        # Train Decision Tree Model
        # =====================

        self.model = DecisionTreeRegressor(max_depth=3)
        self.model.fit(self.features, self.prices)

        # Save trained model using pickle
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """
        Create and arrange all graphical interface elements.

        The interface contains:
        - A header title
        - A sidebar with model options
        - Input fields for house features
        - A button to make predictions
        - A label to display the result
        """

        # =====================
        # Header
        # =====================

        header = tk.Label(
            self.root,
            text="Boston House Price Predictor",
            bg="blue",
            fg="white",
            font=("Arial", 24, "bold")
        )
        header.pack(fill=tk.X)

        # =====================
        # Sidebar
        # =====================

        sidebar = tk.Frame(self.root, bg="lightgrey", width=200)
        sidebar.pack(fill=tk.Y, side=tk.LEFT)

        tk.Label(
            sidebar,
            text="Decision Tree Model",
            bg="lightgrey",
            font=("Arial", 14),
            anchor="w",
            padx=10
        ).pack(fill=tk.X, pady=5)

        # Button to show complexity plot
        tk.Button(
            sidebar,
            text="Show Complexity Plot",
            command=self.show_complexity_plot,
            font=("Arial", 12)
        ).pack(fill=tk.X, pady=10)

        # =====================
        # Main Input Section
        # =====================

        main = tk.Frame(self.root)
        main.pack(pady=40)

        tk.Label(
            main,
            text="Enter House Features",
            font=("Arial", 22)
        ).pack(pady=10)

        # RM Input
        tk.Label(main, text="RM (Number of Rooms)", font=("Arial", 14)).pack()
        self.rm_entry = tk.Entry(main, font=("Arial", 14))
        self.rm_entry.pack()

        # LSTAT Input
        tk.Label(main, text="LSTAT (% lower class)", font=("Arial", 14)).pack()
        self.lstat_entry = tk.Entry(main, font=("Arial", 14))
        self.lstat_entry.pack()

        # PTRATIO Input
        tk.Label(main, text="PTRATIO (Student/Teacher)", font=("Arial", 14)).pack()
        self.ptratio_entry = tk.Entry(main, font=("Arial", 14))
        self.ptratio_entry.pack()

        # =====================
        # Predict Button
        # =====================

        tk.Button(
            main,
            text="Predict Price",
            command=self.predict,
            bg="green",
            font=("Arial", 14)
        ).pack(pady=15)

        # Label to display prediction result
        self.result = tk.Label(
            main,
            text="",
            font=("Arial", 18, "bold")
        )
        self.result.pack()

    # =====================
    # Prediction Function
    # =====================

    def predict(self):
        """
        Predict the house price using user input.

        The function:
        - Reads values entered by the user
        - Converts them to numerical values
        - Sends them to the trained model
        - Displays the predicted price

        If invalid data is entered, an error message is shown.
        """

        try:
            # Read user input
            rm = float(self.rm_entry.get())
            lstat = float(self.lstat_entry.get())
            ptratio = float(self.ptratio_entry.get())

            # Create dataframe with correct feature names
            data = pd.DataFrame(
                [[rm, lstat, ptratio]],
                columns=['RM', 'LSTAT', 'PTRATIO']
            )

            # Predict house price
            prediction = self.model.predict(data)[0]

            # Display result
            self.result.config(
                text=f"Predicted Price: ${prediction*1000:,.2f}",
                fg="green"
            )

        except ValueError:
            # Handle invalid input
            self.result.config(
                text="Invalid Input",
                fg="red"
            )

    # =====================
    # Model Complexity Plot
    # =====================

    def show_complexity_plot(self):
        """
        Display a model complexity plot.

        This function:
        - Splits the dataset into training and testing sets
        - Trains multiple Decision Tree models with different depths
        - Calculates the R² score for each model
        - Plots model performance versus tree depth
        """

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.prices,
            test_size=0.2,
            random_state=42
        )

        depths = list(range(1, 11))
        scores = []

        # Train models with different tree depths
        for d in depths:

            model = DecisionTreeRegressor(max_depth=d)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

            scores.append(score)

        # Plot results
        plt.figure()
        plt.plot(depths, scores, marker='o')
        plt.title("Decision Tree Model Complexity")
        plt.xlabel("Max Depth")
        plt.ylabel("R2 Score")
        plt.grid(True)
        plt.show()


# =====================
# Run Application
# =====================

if __name__ == "__main__":

    root = tk.Tk()
    app = BostonHousingApp(root)
    root.mainloop()