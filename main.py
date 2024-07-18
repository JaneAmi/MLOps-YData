import logging
import warnings
from sklearn.model_selection import train_test_split

# Custom modules
from mlops_prj_ydata.dataset import german_credit_data, bank_marketing
from models.models import XGBClassifierWrapper, XGBClassifierWrapperSimple
from xai_compare.comparison import FeatureElimination

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Adjust SHAP logging level to WARNING or ERROR
logging.getLogger("shap").setLevel(logging.WARNING)


def process_dataset(dataset_func, model_class, params, test_size=0.3, random_state=2, stratify=None):
    """
    Process a dataset by loading data, splitting it, training a model, and evaluating the results.
    
    Parameters:
        dataset_func (function): Function to load the dataset.
        model_class (class): The model class to be used for training.
        params (dict): Parameters for the FeatureElimination class.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        stratify (array-like): Stratify splits by this array.
    """

    # Get the data
    logger.info("Loading dataset...")
    X, y = dataset_func()

    # Split into training and test sets
    logger.info("Splitting dataset...")
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Create and fit the model
    logger.info("Training model...")
    model = model_class()
    model.fit(X_train_clean, y_train_clean)

    # Evaluate the model
    logger.info("Evaluating model...")
    model.evaluate(X_test_clean, y_test_clean)

    # Update params with the current model and data
    params.update({'model': model, 'data': X, 'target': y})

    # Perform feature elimination
    logger.info("Performing feature elimination...")
    feature_elim = FeatureElimination(**params)
    feature_elim.best_result()

def main():
    """
    Main function to process multiple datasets with their respective models and parameters.
    """
    # Bank marketing dataset
    logger.info("Processing Bank Marketing dataset...")
    X_bank, y_bank = bank_marketing()
    bank_params = {
        'custom_explainer': None,
        'mode': 'classification',
        'metric': 'Accuracy',
        'default_explainers': ['shap', 'permutations'],
        'verbose': True,
        'random_state': 2,
        'threshold': 0.6
    }
    process_dataset(lambda: (X_bank, y_bank), XGBClassifierWrapperSimple, bank_params, test_size=0.3, random_state=2, stratify=y_bank)

    # German credit risk dataset
    logger.info("Processing German Credit Risk dataset...")
    X_german, y_german = german_credit_data()
    german_params = {
        'custom_explainer': None,
        'mode': 'classification',
        'metric': 'AUC',
        'default_explainers': ['shap', 'permutations'],
        'verbose': True,
        'random_state': 1,
        'threshold': 0.6
    }
    process_dataset(lambda: (X_german, y_german), XGBClassifierWrapper, german_params, test_size=0.2, random_state=1)

if __name__ == "__main__":
    main()

