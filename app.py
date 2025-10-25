import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

PRUNED_MODEL_FILE = "pruned_model.pkl"

st.set_page_config(page_title="CART Pruned Model App", layout="wide")
st.title("CART Pruned Model Explorer")

# Use caching to load the model and data only ONCE
@st.cache_resource
def load_model_data():
    """
    Loads the model and test data from the .pkl file.
    This function is cached so it only runs once.
    """
    try:
        data_dict = joblib.load(PRUNED_MODEL_FILE)
        return data_dict
    except FileNotFoundError:
        st.error(f"Model file '{PRUNED_MODEL_FILE}' not found.")
        st.error("Please run 'python train_model.py' first to create the model file.")
        st.stop()

# Load the data
data_dict = load_model_data()

# Unpack the data from the dictionary
model = data_dict["model"]
X_test = data_dict["X_test"]
y_test = data_dict["y_test"]
feature_names = data_dict["feature_names"]
target_names = data_dict["target_names"]

# --- Model Evaluation ---
st.header("Pruned Model Evaluation")
st.markdown("This section shows the model's performance on the unseen test data.")

y_pred = model.predict(X_test)
st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

with st.expander("Show Detailed Classification Report & Confusion Matrix"):
    st.text("Classification Report:")
    # Use st.code to display text in a monospaced block
    st.code(classification_report(y_test, y_pred, target_names=target_names))
    
    st.text("Confusion Matrix:")
    st.code(confusion_matrix(y_test, y_pred))

# --- Tree Visualization ---
st.header("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    model, 
    filled=True, 
    feature_names=feature_names, 
    class_names=target_names, 
    fontsize=10
)
st.pyplot(fig)

# --- Classify a New Sample ---
st.header("Classify a New Sample")
st.markdown("Use the slider to pick a sample from the test set and see the model's prediction.")

sample_index = st.slider("Select a test sample index:", 0, len(X_test) - 1, 0)

# Get the single sample
new_sample = X_test[sample_index].reshape(1, -1)
actual_label = target_names[y_test[sample_index]]
pred_label = target_names[model.predict(new_sample)[0]]

# Display prediction
col1, col2 = st.columns(2)
col1.metric("Actual Class", actual_label)
col2.metric("Predicted Class", pred_label)

if pred_label == actual_label:
    st.success("✅ Correct prediction")
else:
    st.error("❌ Incorrect prediction")