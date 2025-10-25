import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PRUNED_MODEL_FILE = "pruned_model.pkl"

def train_and_save_model(max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features='sqrt'):
    # Load dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=RANDOM_STATE
    )
    
    # Train pruned model
    pruned_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE
    )
    pruned_model.fit(X_train, y_train)

    # Save everything needed by the app in one dictionary
    save_dict = {
        "model": pruned_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": data.feature_names,
        "target_names": data.target_names
    }
    
    # Save the dictionary to a single file
    joblib.dump(save_dict, PRUNED_MODEL_FILE)
    print(f"Pruned model + test data saved as '{PRUNED_MODEL_FILE}'")

if __name__ == "__main__":
    train_and_save_model()