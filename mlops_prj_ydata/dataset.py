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
    def __init__(self, verbose=False, save=True):
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
        self.save = save

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

        if self.save:
            df = pd.concat([self.X_clean, self.y_clean], axis=1)
            df.to_csv("../data/processed/german_credit_score.csv", index=False)

        return self.X_clean, self.y_clean

def german_credit_data(save=True):
    try:
        df = pd.read_csv("../data/processed/german_credit_score.csv")
        return df.iloc[:, :-1], df.iloc[:, -1:]
    except:
        preprocessor = CreditDataPreprocessor(save=save)
        return preprocessor.prepare_data()




import pandas as pd

class BankDataProcessor:
    def __init__(self, filepath="../data/raw/bank.csv", save=True):
        """
        Initialize the BankDataProcessor with the path to the CSV file.
        """
        self.filepath = filepath
        self.data = None
        self.data_X = None
        self.data_y = None
        self.save = save

    def load_data(self):
        """
        Load data from a CSV file.
        """
        self.data = pd.read_csv(self.filepath, delimiter=";", header='infer')
        return self.data.head()

    def preprocess_data(self):
        """
        Preprocess the data by converting categorical columns to dummy variables
        and encoding the target variable.
        """
        columns_to_dummy = ['job', 'marital', 'education', 'default', 'housing', 
                            'loan', 'contact', 'month', 'poutcome']
        self.data = pd.get_dummies(self.data, columns=columns_to_dummy)
        self.data['y'].replace(('yes', 'no'), (1, 0), inplace=True)

        if self.save:
            self.data.to_csv("../data/processed/bank_marketing.csv", index=False)


    def get_features_and_target(self):
        """
        Public method to get the preprocessed features and target.
        Split the data into features (X) and target (y).
        """
        self.load_data()
        self.preprocess_data()
        self.data_y = pd.DataFrame(self.data['y'])
        self.data_X = self.data.drop(['y'], axis=1)

        return self.data_X, self.data_y


def bank_marketing(save=True):
    try:
        df = pd.read_csv("../data/processed/bank_marketing.csv.csv")
        return df.iloc[:, :-1], df.iloc[:, -1:]
    except:
        processor = BankDataProcessor(save=save)
        return processor.get_features_and_target()

