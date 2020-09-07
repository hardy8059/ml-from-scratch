# Import Packages
from algorithms.linear_regression.univariate import UnivariateLR
from src.utilities import Utils

# Variable Declaration
utils = Utils("bike_sharing_data.txt")
u_lr = UnivariateLR(utils.load_data())

# Code
if __name__ == "__main__":
    # Get dataset details
    # utils.get_data_details()

    # Visualize the dataset
    # u_lr.visualize_dataset()

    # Run gradient Descent
    # u_lr.gradient_descent(alpha=0.01, iterations=2000)

    # Visualize Cost Function
    # u_lr.compute_cost()

    # Visualize Regressio Fit
    # u_lr.plot_regression_fit()

    # Run whole pipeline in one go
    u_lr.pipeline(alpha=0.01, iterations=2000)
