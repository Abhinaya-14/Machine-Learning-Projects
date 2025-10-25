import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set a consistent random state
RANDOM_STATE = 42

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="CART & Pruning Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Caching Functions (for performance)
# ---------------------------------

@st.cache_data
def load_data():
    """Loads and splits the breast cancer dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, data

@st.cache_resource
def train_unpruned_model(X_train, y_train):
    """Trains the full, unpruned CART model."""
    dt_classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

@st.cache_resource
def train_pruned_model(_X_train, _y_train, max_depth, min_samples_split, min_samples_leaf, max_features):
    """Trains the pruned CART model based on sidebar parameters."""
    pruned_dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE
    )
    pruned_dt_classifier.fit(_X_train, _y_train)
    return pruned_dt_classifier

# ---------------------------------
# Sidebar for User Input
# ---------------------------------
st.sidebar.title("Pruning Parameters")
st.sidebar.markdown("""
Adjust the hyperparameters to control the tree's complexity and prevent overfitting.
""")

param_max_depth = st.sidebar.slider(
    "1. Maximum Depth (`max_depth`)", 
    min_value=2, max_value=20, value=5, step=1,
    help="Restricts the maximum depth of the tree. A smaller value reduces complexity."
)

param_min_samples_split = st.sidebar.slider(
    "2. Minimum Samples Split (`min_samples_split`)",
    min_value=2, max_value=50, value=10, step=1,
    help="The minimum number of samples required to split an internal node."
)

param_min_samples_leaf = st.sidebar.slider(
    "3. Minimum Samples Per Leaf (`min_samples_leaf`)",
    min_value=1, max_value=50, value=5, step=1,
    help="The minimum number of samples required to be at a leaf node."
)

param_max_features = st.sidebar.selectbox(
    "4. Maximum Features (`max_features`)",
    options=['sqrt', 'log2', None],
    index=0,
    help="The number of features to consider when looking for the best split."
)

# ---------------------------------
# Main Application
# ---------------------------------
st.title("CART Decision Tree & Pruning Explorer")
st.markdown("Explore how pruning affects CART (Decision Tree) performance on the Breast Cancer dataset.")

# Load data
X_train, X_test, y_train, y_test, data = load_data()

# Train models
unpruned_model = train_unpruned_model(X_train, y_train)
pruned_model = train_pruned_model(
    X_train, y_train, 
    param_max_depth, 
    param_min_samples_split, 
    param_min_samples_leaf, 
    param_max_features
)

# --- 1. Dataset Overview ---
with st.expander("1. Dataset Overview (Breast Cancer)"):
    st.write("The Breast Cancer Wisconsin dataset is a classic classification dataset.")
    st.markdown(f"**Features:** {X_train.shape[1]} (e.g., `{data.feature_names[0]}`, `{data.feature_names[1]}`...)")
    st.markdown(f"**Target Classes:** {data.target_names[0]} (0), {data.target_names[1]} (1)")
    st.markdown(f"**Training Samples:** {X_train.shape[0]}")
    st.markdown(f"**Testing Samples:** {X_test.shape[0]}")

# --- 2. Unpruned Model ---
st.header("2. Unpruned Model (Overfit)")
st.markdown("This is the default CART model with no restrictions. It will likely overfit the training data.")

y_train_pred_unpruned = unpruned_model.predict(X_train)
y_test_pred_unpruned = unpruned_model.predict(X_test)
train_acc_unpruned = accuracy_score(y_train, y_train_pred_unpruned)
test_acc_unpruned = accuracy_score(y_test, y_test_pred_unpruned)

col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc_unpruned:.4f}")
col2.metric("Test Accuracy", f"{test_acc_unpruned:.4f}")

if train_acc_unpruned == 1.0 and test_acc_unpruned < 0.98:
    st.warning("Overfitting Detected: Training accuracy is 100% while test accuracy is lower.")

with st.expander("Show Unpruned Model Report & Tree"):
    st.text("Test Classification Report (Unpruned):")
    st.text(classification_report(y_test, y_test_pred_unpruned, target_names=data.target_names))
    
    st.subheader("Unpruned Tree Visualization")
    fig, ax = plt.subplots(figsize=(25, 12))
    plot_tree(
        unpruned_model, 
        filled=True, 
        feature_names=data.feature_names, 
        class_names=data.target_names,
        fontsize=8
    )
    st.pyplot(fig)

# --- 3. Pruned Model ---
st.header("3. Pruned Model (With Parameters)")
st.markdown("This model uses the parameters set in the sidebar to reduce overfitting.")

y_train_pred_pruned = pruned_model.predict(X_train)
y_pred_pruned = pruned_model.predict(X_test)
train_acc_pruned = accuracy_score(y_train, y_train_pred_pruned)
test_acc_pruned = accuracy_score(y_test, y_pred_pruned)

col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc_pruned:.4f}")
col2.metric("Test Accuracy", f"{test_acc_pruned:.4f}")

if test_acc_pruned > test_acc_unpruned:
    st.success("✅ Pruning Improved Generalization!")
elif test_acc_pruned == test_acc_unpruned:
    st.info("ℹ️ Pruning Maintained Accuracy: simpler model, similar test performance.")
else:
    st.warning("⚠️ Over-pruning Detected: test accuracy decreased. Adjust parameters.")

with st.expander("Show Pruned Model Report & Tree"):
    st.text("Classification Report (Pruned):")
    st.code(classification_report(y_test, y_pred_pruned, target_names=data.target_names))
    
    st.text("Confusion Matrix (Pruned):")
    st.code(confusion_matrix(y_test, y_pred_pruned))
    
    st.text("Accuracy Score (Pruned):")
    st.code(f"Accuracy: {accuracy_score(y_test, y_pred_pruned):.4f}")

    st.subheader("Pruned Tree Visualization")
    fig_pruned, ax_pruned = plt.subplots(figsize=(20, 10))
    plot_tree(
        pruned_model, 
        filled=True, 
        feature_names=data.feature_names, 
        class_names=data.target_names,
        fontsize=10,
        max_depth=param_max_depth
    )
    st.pyplot(fig_pruned)

# --- 4. Classify a New Sample ---
st.header("4. Classify a New Sample")
st.markdown("Select a test sample to see how both models classify it.")

sample_index = st.slider(
    "Select a test sample index:", 
    min_value=0, 
    max_value=len(X_test) - 1, 
    value=0, 
    step=1
)

new_sample = X_test[sample_index].reshape(1, -1)
actual_label_index = y_test[sample_index]
actual_label_name = data.target_names[actual_label_index]

st.subheader(f"Classifying Test Sample #{sample_index}")
st.markdown(f"**Actual Class:** `{actual_label_name}`")

pred_unpruned = unpruned_model.predict(new_sample)
pred_pruned = pruned_model.predict(new_sample)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Unpruned Model")
    pred_name = data.target_names[pred_unpruned[0]]
    st.markdown(f"**Predicted:** `{pred_name}`")
    if pred_unpruned[0] == actual_label_index:
        st.success("Correct")
    else:
        st.error("Incorrect")

with col2:
    st.markdown("#### Pruned Model")
    pred_name = data.target_names[pred_pruned[0]]
    st.markdown(f"**Predicted:** `{pred_name}`")
    if pred_pruned[0] == actual_label_index:
        st.success("Correct")
    else:
        st.error("Incorrect")
