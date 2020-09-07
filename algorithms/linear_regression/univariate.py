# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utilities import Utils


# Variable Declaration


# Code
class UnivariateLR:
    def __init__(self, data, label_index=-1):
        self.data = data
        self.label_index = label_index
        self.m, self.X, self.y, self.theta = self.get_X_y(data)
        self.validate_costs = []

    def visualize_dataset(self):
        print("Plotting data...")
        ax = sns.scatterplot(x=self.data.columns[0], y=self.data.columns[self.label_index],
                             data=self.data)
        ax.set_title("{} vs {} chart".format(self.data.columns[0], self.data.columns[1]))
        plt.show()

    def get_X_y(self, data):
        print("Splitting dependent and independent columns...")
        m = data.shape[0]
        X = np.append(np.ones((m, 1)), data[data.columns[0]].values.reshape(m, 1), axis=1)
        y = data[data.columns[self.label_index]].values.reshape(m, 1)
        theta = np.zeros((data.shape[1], 1))
        return m, X, y, theta

    def compute_cost(self):
        y_pred = self.X.dot(self.theta)
        error = np.sum(np.square(y_pred - self.y)) * (0.5 / self.m)
        return error

    def gradient_descent(self, alpha, iterations):
        print("Starting Gradient Descent...")
        for i in range(1,iterations+1):
            y_pred = self.X.dot(self.theta)
            error_gradient = np.dot(self.X.transpose(), y_pred - self.y)
            self.theta = self.theta - ((alpha / self.m) * error_gradient)
            self.validate_costs.append(self.compute_cost())
            if i % 100==0:
                print("Cost value after {} iterations is: {}".format(i, self.validate_costs[-1]))

    def visualize_cost_function(self):
        print("Plotting Cost function to validate the modelling process...")
        assert len(self.validate_costs) > 0, "Costs not yet computed."
        plt.plot(self.validate_costs)
        plt.xlabel("Iterations")
        plt.ylabel("$Cost - J(\Theta)$")
        plt.title("Cost function VS Iterations")
        plt.show()

    def plot_regression_fit(self):
        print("Plotting regression fit over data...")
        t = np.squeeze(self.theta)
        sns.scatterplot(x=self.data.columns[0], y=self.data.columns[self.label_index],
                             data=self.data)
        x_value = [x for x in range(int(min(self.data[self.data.columns[0]])), int(max(self.data[self.data.columns[0]])))]
        y_value = [x*t[1]+t[0] for x in x_value]

        sns.lineplot(x_value, y_value)
        plt.xlabel(self.data.columns[0])
        plt.ylabel(self.data.columns[self.label_index])
        plt.show()

    def predict(self, x):
        return np.dot(self.theta.transpose(), x)

    def pipeline(self, alpha, iterations):
        self.visualize_dataset()
        self.gradient_descent(alpha=alpha, iterations=iterations)
        self.visualize_cost_function()
        self.plot_regression_fit()