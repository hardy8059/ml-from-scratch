from src.get_path import path_to_dataset_folder
import pandas as pd
import os


class Utils:
    def __init__(self, dataset_name):
        self.path_to_dataset = os.path.join(path_to_dataset_folder,dataset_name)
        assert (os.path.exists(self.path_to_dataset) == True), "Invalid path to dataset."
        self.data = pd.read_csv(self.path_to_dataset)

    def load_data(self):
        """
        Load the dataset in memory
        :return: A pandas DataFrame with csv
        """
        try:
            return self.data
        except Exception as e:
            return e

    def get_data_details(self, info=True, describe=True, display_rows=True, row_count=5):
        """
        Get details about data like row and column information and display the data rows
        :param info: True, if you want to print the details of rows and columns in the data
        :param describe: True, if you want to print the average, max, minimum etc. details of the columns
        :param display_rows: True, if you want to print the rows of data
        :param row_count: int, number of rows to print from starting
        :return:
        """
        try:
            print("****************************************************")
            if info:
                print("Row and column details are...")
                print("****************************************************")
                print(self.data.info())
                print("****************************************************")
            if describe:
                print("Statistical information for columns are...")
                print("****************************************************")
                print(self.data.describe())
                print("****************************************************")
            if display_rows:
                print("First {} rows are...".format(row_count))
                print("****************************************************")
                print(self.data.head(row_count))
                print("****************************************************")
        except Exception as e:
            return e