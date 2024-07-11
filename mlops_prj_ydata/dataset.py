from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict


from mlops_prj_ydata.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()




class CreditDataPreprocessor:
    def __init__(self, verbose=False):
        self.names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
                      'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
                      'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
                      'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
        
        self.num_vars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
                         'existingcredits', 'peopleliable']
        
        self.cat_vars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
                         'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
                         'telephone', 'foreignworker']
        
        self.X_clean = None
        self.y_clean = None
        self.verbose = verbose

    def fetch_data(self):
        """ Fetch dataset from UCI repository """
        data = fetch_ucirepo(id=144)
        X = data.data.features
        y = data.data.targets

        if self.verbose:        
            print(data.metadata)
            print(data.variables)
        
        X.columns = self.names[:-1]
        y.replace([1, 2], [1, 0], inplace=True)  # Binarize the output
        
        return X, y

    def preprocess_data(self, X, y):
        """ Preprocess data: encoding, scaling, and merging data """
        # Standardization of numerical variables
        num_data_std = pd.DataFrame(StandardScaler().fit_transform(X[self.num_vars]), columns=self.num_vars)
        
        # Encoding categorical variables
        d = defaultdict(LabelEncoder)
        le_cat_data = X[self.cat_vars].apply(lambda x: d[x.name].fit_transform(x))
        
        if self.verbose:
            for var in self.cat_vars:
                print(f"{var}: Original -> {X[var].unique()}")
                print(f"{var}: Encoded -> {le_cat_data[var].unique()}")
        
        # One-hot encoding for categorical variables
        dummy_vars = pd.get_dummies(X[self.cat_vars], columns=self.cat_vars)
        
        # Concatenating all processed features
        data_clean = pd.concat([num_data_std, dummy_vars], axis=1)
        
        return data_clean, y

    def prepare_data(self):
        """ Main method to prepare data """
        X, y = self.fetch_data()
        self.X_clean, self.y_clean = self.preprocess_data(X, y)
        return self.X_clean, self.y_clean

def german_credit_data():
    preprocessor = CreditDataPreprocessor()
    return preprocessor.prepare_data()

