import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression App")

        self.load_data_button = ttk.Button(root, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.linear_regression_button = ttk.Button(root, text="Linear Regression (scikit-learn)", command=self.linear_regression)
        self.linear_regression_button.pack(pady=10)

        self.nn_regression_button = ttk.Button(root, text="Neural Network Regression (TensorFlow/Keras)", command=self.nn_regression)
        self.nn_regression_button.pack(pady=10)

        self.scatter_plot_button = ttk.Button(root, text="Scatter Plot", command=self.scatter_plot)
        self.scatter_plot_button.pack(pady=10)

        self.quit_button = ttk.Button(root, text="Quit", command=root.destroy)
        self.quit_button.pack(pady=10)

        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.linear_model = None
        self.nn_model = None

    def load_data(self):
        # Replace 'your_data.csv' with the path to your dataset
        file_path = tk.filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            tk.messagebox.showinfo("Info", "Data loaded successfully.")

    def split_data(self):
        if self.data is not None:
            features = self.data.drop(columns=['Target'])
            target = self.data['Target']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            tk.messagebox.showinfo("Info", "Data split successfully.")

    def linear_regression(self):
        self.split_data()
        if self.X_train is not None:
            self.linear_model = LinearRegression()
            self.linear_model.fit(self.X_train, self.y_train)
            tk.messagebox.showinfo("Info", "Linear Regression model trained successfully.")

    def nn_regression(self):
        self.split_data()
        if self.X_train is not None:
            self.nn_model = Sequential([
                Dense(8, input_dim=self.X_train.shape[1], activation='relu'),
                Dense(1)
            ])
            self.nn_model.compile(optimizer='adam', loss='mean_squared_error')
            self.nn_model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)
            tk.messagebox.showinfo("Info", "Neural Network Regression model trained successfully.")

    def scatter_plot(self):
        if self.data is not None:
            plt.scatter(self.data['Feature'], self.data['Target'], alpha=0.5)
            plt.title("Scatter Plot")
            plt.xlabel("Feature")
            plt.ylabel("Target")

            if self.linear_model is not None:
                plt.plot(self.data['Feature'], self.linear_model.predict(self.data[['Feature']]), color='red', linewidth=2, label="Linear Regression")

            if self.nn_model is not None:
                plt.plot(self.data['Feature'], self.nn_model.predict(self.data[['Feature']]), color='green', linewidth=2, label="Neural Network Regression")

            plt.legend()
            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()
