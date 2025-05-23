# COVID-19 Bayesian Network Diagnostic Model

## Overview
This repository contains the implementation of an optimized Bayesian Network for COVID-19 diagnosis leveraging Dirichlet priors and BDeu scoring for accurate inference. The model achieves over 94% accuracy in predicting COVID-19 test outcomes based on clinical symptoms and demographic information.

## Features
- **Bayesian Network Construction**: Directed Acyclic Graph (DAG) modeling symptom-disease relationships
- **Advanced Parameter Estimation**: Integration of Dirichlet priors with BDeu scoring metric
- **Multiple Inference Methods**: Implementation and comparison of Likelihood Weighting, Rejection Sampling, and Gibbs Sampling
- **Comprehensive Evaluation**: Multiple distance metrics and performance assessments
- **Clinical Decision Support**: Probabilistic inference for COVID-19 diagnosis

## File Structure

```bash
covid-19-bayesian-network/
├── covid(thirdpaper).ipynb      # Jupyter Notebook with all analysis and code
├── data/
│   ├── corona_tested_individuals_ver_0083.english.csv
│   └── corona_tested_individuals_ver_006.english.csv
└── README.md
```

## Requirements
- Python 3.7.15
- pgmpy 0.1.12
- NumPy 1.21.6
- Pandas 1.3.5
- Matplotlib 3.5.3
- Seaborn 0.12.1
- Scikit-learn 1.0.2
- SciPy 1.7.3
- NetworkX 2.6.3
- Statsmodels 0.13.5

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/username/COVID-19-Bayesian-Network-Diagnostic-Model.git
cd COVID-19-Bayesian-Network-Diagnostic-Model
```

### Step 2: Create Conda Environment
```bash
# Using the provided environment file
conda env create -f thirdpaper_covid.yml
conda activate thirdpaper
```

**Alternative Manual Installation:**
```bash
conda create -n thirdpaper python=3.7.15
conda activate thirdpaper
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preprocessing
```python
# Handle missing values
df_test['age_60_and_above'].fillna('No', inplace=True)
updated_df['age_60_and_above'].fillna('No', inplace=True)

# Define categorical encodings
cleanup_nums = {
    "corona_result": {
        "other": 2,
        "negative": 0,
        "positive": 1
    },
    "age_60_and_above": {
        "Yes": 1,
        "No": 0
    },
    "gender": {
        "male": 1,
        "female": 0
    },
    "test_indication": {
        "Contact with confirmed": 1,
        "Abroad": 2,
        "Other": 3
    }
}

# Apply encodings
updated_df1 = updated_df.replace(cleanup_nums)
df_test = df_test.replace(cleanup_nums)
```

### 2. Build Bayesian Network
```python
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination, BayesianModelSampling, GibbsSampling
from IPython.core.display import display, HTML

# Define the structure of the Bayesian Network
# Each tuple represents a directed edge: (parent, child)
architecture = [
    ("test_date", "corona_result"),
    ("cough", "corona_result"),
    ("fever", "corona_result"),
    ("sore_throat", "corona_result"),
    ("shortness_of_breath", "corona_result"),
    ("head_ache", "corona_result"),
    ("age_60_and_above", "corona_result"),
    ("test_indication", "corona_result"),
    ("gender", "corona_result")
]

# Initialize the Bayesian Model with the given structure
model = BayesianModel(architecture)

# Improve notebook display: prevent line wrapping in output cells (optional)
display(HTML("<style>.output_area pre {white-space: pre;}</style>"))

# Fit the model to the dataset using a Bayesian Estimator
# - BDeu prior: suitable for sparse data
# - Equivalent Sample Size controls the influence of prior vs. data
# - complete_samples_only=False allows incomplete rows in the training data
model.fit(
    data=updated_df1,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
    complete_samples_only=False
)

# Validate the model to ensure there are no logical inconsistencies
print(f'Check model: {model.check_model()}\n')

# Print the Conditional Probability Distributions (CPDs) of each node
for cpd in model.get_cpds():
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')

# Create inference engines for querying the model
# 1. Exact Inference
inference = VariableElimination(model)

# 2. Approximate Inference Methods
BMS_inference = BayesianModelSampling(model)  # Likelihood weighting and rejection
GS_inference = GibbsSampling(model)           # Gibbs sampling

```

### 3. Perform Inference
```python
def show_active_trail(model, start, end, evidences={}, trail_to_show=[]):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from pgmpy.inference import VariableElimination

    # Build a readable title string for the inference
    str_evidences = '| ' if evidences else ''
    for evidence, value in evidences.items():
        str_evidences += f"{evidence}={value}"
        if evidence != list(evidences.keys())[-1]:
            str_evidences += ', '
    title_inference = f'P({end} {str_evidences})'

    # Set up figure layout with two subplots: inference table and network graph
    fig = plt.figure(figsize=(20, 10))
    ax1, ax2 = fig.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1.2, 1.8]})

    # Perform Bayesian inference
    inference = VariableElimination(model)
    query = inference.query(variables=[end], evidence=evidences, show_progress=False)

    # Format and display the inference table
    probabilities = [f'{value:.4f}' for value in query.values]
    table = np.column_stack((query.state_names[query.variables[0]], probabilities))
    font_size = 18
    bbox = [0, 0, 1, 0.70]
    mpl_table = ax1.table(cellText=table,
                          bbox=bbox,
                          cellLoc='center',
                          colLabels=[f'{end} State', f'P({end} | Evidence)'],
                          colWidths=[1.5, 2],
                          loc='center',
                          colColours=['#d9d9d9', '#d9d9d9'])
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    ax1.axis('off')
    ax1.set_title(title_inference, fontsize=22)

    # Check and display whether the trail is active
    obs = list(evidences.keys()).copy()
    if start in obs:
        obs.remove(start)
    active = model.is_active_trail(start=start, end=end, observed=obs)
    title_graph = f"Active Trail ({start} → {end}): {'Active' if active else 'Inactive'}"

    # Filter edges and configure colors and sizes
    trail_edges = [edge for edge in model.edges if edge[0] in trail_to_show and edge[1] in trail_to_show]
    node_colors = ['#f4d03f' if node in evidences else '#ffffff' for node in trail_to_show]
    node_sizes = [1200 if node in [start, end] else 900 for node in trail_to_show]
    edge_colors = ['#e74c3c' if (edge[0] in evidences and evidences[edge[0]] == 1) else 
                   '#3498db' if (edge[0] in evidences and evidences[edge[0]] == 0) else '#7f8c8d' 
                   for edge in trail_edges]
    edge_widths = [3.2 if edge[0] in evidences else 2.2 for edge in trail_edges]

    # Draw the network graph
    pos = nx.spring_layout(model, seed=42)
    nx.draw_networkx_edges(model, pos, edgelist=trail_edges, edge_color=edge_colors, width=edge_widths, ax=ax2, arrows=True)
    nx.draw_networkx_nodes(model, pos, nodelist=trail_to_show, node_color=node_colors, edgecolors='black', node_size=node_sizes, ax=ax2)
    nx.draw_networkx_labels(model, pos, ax=ax2, font_size=16)

    # Annotate start and end nodes
    ax2.annotate('Start Node', xy=pos[start], xycoords='data',
                 xytext=(-70, 40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax2.annotate('End Node', xy=pos[end], xycoords='data',
                 xytext=(70, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # Add legend for color and edge meaning
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Evidence Node (Yellow)', 
                                  markerfacecolor='#f4d03f', markersize=12, markeredgecolor='black'),
                       plt.Line2D([0], [0], marker='o', color='w', label='Other Nodes (White)', 
                                  markerfacecolor='#ffffff', markersize=12, markeredgecolor='black'),
                       plt.Line2D([0], [0], color='#e74c3c', lw=3.2, label='Edge (Symptom Present - Red)'),
                       plt.Line2D([0], [0], color='#3498db', lw=3.2, label='Edge (Symptom Absent - Blue)'),
                       plt.Line2D([0], [0], color='#7f8c8d', lw=2.2, label='Edge (Neutral - Gray)')]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=14, bbox_to_anchor=(1.15, 1))

    ax2.set_title(title_graph, fontsize=22)
    plt.tight_layout()
    plt.show()

show_active_trail(
    model,
    start='test_date',
    end='corona_result',
    evidences={'sore_throat': 1, 'fever': 1},
    trail_to_show=[
        'test_date', 'cough', 'fever', 'sore_throat', 'shortness_of_breath',
        'head_ache', 'corona_result', 'age_60_and_above', 'gender', 'test_indication'
    ]
)
show_active_trail(
    model,
    start='test_date',
    end='corona_result',
    evidences={'sore_throat': 0, 'fever': 0},
    trail_to_show=[
        'test_date', 'cough', 'fever', 'sore_throat', 'shortness_of_breath',
        'head_ache', 'corona_result', 'age_60_and_above', 'gender', 'test_indication'
    ]
)


```
![Fig2](Fig2.png)
![Fig3](Fig3.png)
### 4. Comparing Conditional Probability Tables (CPTs) with Visual Labels
```python
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
import pandas as pd
import matplotlib.patches as mpatches

def compare_cpts_with_labels(model, variable, result_labels):
    """
    Compare the CPTs (conditional probability tables) of a variable across multiple evidence sets.

    Parameters:
    - model: pgmpy BayesianModel instance
    - variable: str, the variable to query (e.g., 'corona_result')
    - result_labels: list of str, labels for the variable's states (e.g., ['negative', 'positive', 'other'])
    """
    inference = VariableElimination(model)

    # Define distinct evidence sets to compare
    evidence_sets = [
        {'fever': 1, 'sore_throat': 1, 'cough': 1, 'head_ache': 1},
        {'fever': 0, 'sore_throat': 0, 'cough': 0, 'head_ache': 0},
        {'fever': 1, 'sore_throat': 0, 'cough': 1, 'head_ache': 0},
        {'fever': 0, 'sore_throat': 1, 'cough': 0, 'head_ache': 1},
    ]

    cpts = {}
    set_labels = []

    colors = ['#0072B2', '#D55E00', '#CC79A7', '#009E73']  # Blue, Orange, Pink, Teal
    hatch_patterns = ['..', '--', 'xx', '++']  # Distinct and clear hatch styles

    # Query CPTs for each evidence set
    for i, evidence in enumerate(evidence_sets):
        query = inference.query(variables=[variable], evidence=evidence, show_progress=False)
        labeled_query = {result_labels[i]: prob for i, prob in enumerate(query.values)}
        cpts[f'Set {i+1}'] = [labeled_query[label] for label in result_labels]
        evidence_desc = ', '.join([f'{k}={v}' for k, v in evidence.items()])
        set_labels.append(f'Set {i+1}: {evidence_desc}')

    # Convert to DataFrame for easy plotting
    cpts_df = pd.DataFrame(cpts, index=result_labels)
    print(cpts_df)

    # Plot the CPTs
    fig, ax = plt.subplots(figsize=(18, 18), dpi=150)
    bars = cpts_df.plot(kind='bar', ax=ax, color=colors, width=0.8)

    # Add hatching and styling to bars
    for i, bar in enumerate(bars.containers):
        for patch in bar:
            patch.set_hatch(hatch_patterns[i])
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

    ax.set_ylabel("Probability", fontsize=20, fontweight='bold')
    ax.set_title(f'CPTs for {variable} Across Evidence Sets', fontsize=24, fontweight='bold', pad=30)
    ax.set_xticklabels(result_labels, rotation=0, fontsize=18, fontweight='bold')

    # Custom legend with colored and hatched patches
    handles = []
    for i, (color, hatch) in enumerate(zip(colors, hatch_patterns)):
        patch = mpatches.Patch(
            facecolor=color,
            hatch=hatch,
            edgecolor='black',
            linewidth=2,
            label=set_labels[i]
        )
        handles.append(patch)

    ax.legend(
        handles=handles,
        title="Evidence Set",
        loc='upper right',
        bbox_to_anchor=(1, 1),
        fontsize=18,
        title_fontsize=20,
        edgecolor='black',
        handleheight=3,
        handlelength=4,
    )

    plt.tight_layout()
    plt.show()

```
![Fig4](Fig4.png)

## Evaluating Model Performance with Metrics and Confusion Matrix

This function calculates the predictive accuracy, precision, recall, F1 score, and confusion matrix
for a given Bayesian Network model on a test dataset using pgmpy and sklearn.metrics.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics_and_confusion_matrix(model, data, actual_var):
    """
    Calculate accuracy, precision, recall, F1 score, and confusion matrix for model predictions.

    Parameters:
    - model: pgmpy BayesianModel (trained)
    - data: pandas DataFrame containing test samples
    - actual_var: string, name of the target variable column

    Returns:
    - accuracy, precision, recall, f1 (percentages)
    - confusion matrix (numpy array)
    """
    from pgmpy.inference import VariableElimination

    inference = VariableElimination(model)
    correct_predictions = 0
    true_labels = []
    predicted_labels = []

    # Iterate over each test sample row
    for _, row in data.iterrows():
        # Extract evidence dictionary excluding the target variable
        evidence = row.drop(actual_var).to_dict()
        actual = row[actual_var]

        # Perform MAP query to predict the target variable given evidence
        query = inference.map_query(variables=[actual_var], evidence=evidence, show_progress=False)

        true_labels.append(actual)
        predicted_labels.append(query[actual_var])

        # Count correct predictions
        if query[actual_var] == actual:
            correct_predictions += 1

    # Calculate metrics
    accuracy = correct_predictions / len(data)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print metrics as percentages
    print(f'Predictive Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    return accuracy * 100, precision * 100, recall * 100, f1 * 100, cm


# Example usage:
# Sample a small portion of the test data (0.1%) for faster computation
df_test_sample = df_test.sample(frac=0.001, random_state=42)

# Calculate metrics on the sample test data
accuracy, precision, recall, f1, cm = calculate_metrics_and_confusion_matrix(model, df_test_sample, 'corona_result')


```
### Predictive Accuracy: 95.34%
### Precision: 94.49%
### Recall: 95.34%
### F1 Score: 94.05%

###  Draw Model Performance
```python
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Metrics to plot
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

# Set up the figure for the performance metrics
fig, ax = plt.subplots(figsize=(12, 10), dpi=150)  # High resolution for clear images

# Use a color-blind friendly palette
colors = sns.color_palette("colorblind")

# Plot the metrics as a bar chart
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette=colors, ax=ax)

# Customize title and labels for better readability in research papers
ax.set_title("Model Performance Metrics", fontsize=24, fontweight='bold', pad=20)
ax.set_ylabel("Percentage (%)", fontsize=18)
ax.set_xlabel("Metrics", fontsize=18)
ax.set_ylim(0, 100)  # Set the y-axis from 0 to 100%

# Annotate the bars with the percentage values
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points', 
                fontsize=16, fontweight='bold')

# Improve grid visibility
ax.grid(True, linestyle='--', alpha=0.6)

# Save the image automatically with high resolution
output_file = "model_performance_metrics.png"
fig.savefig(output_file, bbox_inches='tight', dpi=150)
print(f"Saved as {output_file} (150 DPI, 12x10 inches)")

plt.show()

```
![Fig5](Fig5.png)
###  Plotting the Confusion Matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for the confusion matrix (Higher resolution and colorblind-friendly)
fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

# Use a colorblind-friendly palette (cividis)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar=False,
            xticklabels=['Negative', 'Positive', 'Other'], 
            yticklabels=['Negative', 'Positive', 'Other'],
            linewidths=0.8, linecolor='black', annot_kws={"size": 16, "fontweight": 'bold'})

# Enhance titles and labels for better clarity in research papers
ax.set_title("Confusion Matrix of COVID-19 Test Prediction", fontsize=24, fontweight='bold', pad=30)
ax.set_xlabel("Predicted Label", fontsize=20, fontweight='bold')
ax.set_ylabel("True Label", fontsize=20, fontweight='bold')

# Customize tick parameters for better readability
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)

# Remove grid for a cleaner look
plt.grid(False)

# Save the figure with high resolution
plt.savefig("confusion_matrix_high_res.png", bbox_inches='tight', dpi=150)

plt.tight_layout()
plt.show()
```
![Fig6](Fig6.png)
### Symptom Importance and ROC Curve Analysis using Bayesian Network for COVID-19 Diagnosis
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Define the complete Bayesian Network structure
architecture = [
    ("test_date", "corona_result"),
    ("cough", "corona_result"),
    ("fever", "corona_result"),
    ("sore_throat", "corona_result"),
    ("shortness_of_breath", "corona_result"),
    ("head_ache", "corona_result"),
    ("age_60_and_above", "corona_result"),
    ("test_indication", "corona_result"),
    ("gender", "corona_result")
]

# Create and train the Bayesian Network
model = BayesianModel(architecture)
model.fit(updated_df1, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
inference = VariableElimination(model)

# Function to compute and plot the importance of each symptom in predicting a positive test result
def plot_symptom_importance(model, target="corona_result"):
    importance_scores = {}
    
    # Get all the parent nodes (symptoms) that directly influence the target node (corona_result)
    symptom_nodes = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]
    
    for var in symptom_nodes:
        prob_positive_given_symptom = inference.query(variables=[target], evidence={var: 1}, show_progress=False).values[1]
        prob_positive_given_no_symptom = inference.query(variables=[target], evidence={var: 0}, show_progress=False).values[1]
        importance = prob_positive_given_symptom - prob_positive_given_no_symptom
        importance_scores[var] = importance

    sorted_scores = {k: v for k, v in sorted(importance_scores.items(), key=lambda item: abs(item[1]), reverse=True)}

    # Plotting the importance
    plt.figure(figsize=(12, 12), dpi=150)
    sns.barplot(x=list(sorted_scores.keys()), y=list(sorted_scores.values()), palette="viridis")
    plt.title("Symptom Importance in Predicting Positive COVID-19 Test", fontsize=24, fontweight='bold', pad=20)
    plt.xlabel("Symptom", fontsize=20, fontweight='bold')
    plt.ylabel("Importance Score (Difference in Probability)", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig("symptom_importance_high_res.png", bbox_inches='tight', dpi=150)
    plt.show()

# Run and plot the symptom importance
plot_symptom_importance(model)

# Function to compute and plot ROC curves for each class in a multiclass problem
def plot_roc_curves(inference, data, target="corona_result"):
    plt.figure(figsize=(12, 12), dpi=150)
    
    actuals = label_binarize(data[target], classes=[0, 1, 2])  # Convert the target to binary format
    symptom_nodes = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]
    markers = ['o', 's', '^']  # Distinct markers for each class
    line_styles = ['-', '--', '-.']
    
    for i, (class_label, marker, style) in enumerate(zip(["Negative", "Positive", "Other"], markers, line_styles)):
        probs = []
        for _, row in data.iterrows():
            evidence = {col: row[col] for col in row.index if col != target}
            prob = inference.query(variables=[target], evidence=evidence, show_progress=False).values[i]
            probs.append(prob)
        
        fpr, tpr, _ = roc_curve(actuals[:, i], probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})', linestyle=style, marker=marker, markersize=6)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2)
    plt.title("ROC Curves for Different Classes of COVID-19 Test Result", fontsize=24, fontweight='bold', pad=20)
    plt.xlabel("False Positive Rate", fontsize=20, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig("roc_curves_high_res.png", bbox_inches='tight', dpi=150)
    plt.show()

# Sample a manageable portion of df_test to create the ROC curves
df_test_sample = df_test.sample(frac=0.001, random_state=42)

# Run and plot the ROC curves
plot_roc_curves(inference, df_test_sample)

```
![Fig7](Fig7.png)
![Fig8](Fig8.png)
### Evaluate Model Performance
```python
from src.evaluation_metrics import evaluate_model

# Comprehensive model evaluation
metrics = evaluate_model(bn_model, test_data)
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Precision: {metrics['precision']:.2f}%")
print(f"Recall: {metrics['recall']:.2f}%")
print(f"F1-Score: {metrics['f1_score']:.2f}%")
```
## Usage Examples

### Example 1: Single Patient Diagnosis
```python
# Define patient symptoms
patient_symptoms = {
    'fever': 1,          # Present
    'cough': 1,          # Present
    'sore_throat': 0,    # Absent
    'shortness_of_breath': 0,  # Absent
    'head_ache': 1,      # Present
    'age_60_and_above': 0,     # Under 60
    'gender': 1,         # Male
    'test_indication': 1  # Contact with confirmed
}

# Get diagnosis probabilities
diagnosis = bn_model.predict_proba(patient_symptoms)
print(f"Negative: {diagnosis[0]:.3f}")
print(f"Positive: {diagnosis[1]:.3f}")
print(f"Other: {diagnosis[2]:.3f}")
```

### Example 2: Batch Processing
```python
# Process multiple patients
patient_data = pd.read_csv('new_patients.csv')
predictions = bn_model.predict_batch(patient_data)
patient_data['predicted_covid_prob'] = predictions[:, 1]  # Positive probability
```

### Example 3: Symptom Importance Analysis
```python
from src.bayesian_network import calculate_symptom_importance

# Calculate importance scores for each symptom
importance_scores = calculate_symptom_importance(bn_model)
for symptom, score in importance_scores.items():
    print(f"{symptom}: {score:.4f}")
```

## Inference Methods Comparison

The repository includes three inference methods with performance comparison:

1. **Likelihood Weighting**: Best balance of accuracy and computational efficiency
2. **Rejection Sampling**: High accuracy but computationally intensive
3. **Gibbs Sampling**: MCMC method with convergence challenges in this context

### Running Inference Comparison
```python
from notebooks.inference_comparison import compare_inference_methods

# Compare all three methods
results = compare_inference_methods(
    model=bn_model,
    sample_sizes=[1000, 5000, 10000, 50000],
    evidence={'fever': 1, 'cough': 1}
)
```

## Model Performance

Achieved performance metrics on test dataset:
- **Accuracy**: 95.34%
- **Precision**: 94.49%
- **Recall**: 95.34%
- **F1-Score**: 94.05%

## Data Format

The model expects input data with the following format:

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| cough | int | 0, 1 | 0=Absent, 1=Present |
| fever | int | 0, 1 | 0=Absent, 1=Present |
| sore_throat | int | 0, 1 | 0=Absent, 1=Present |
| shortness_of_breath | int | 0, 1 | 0=Absent, 1=Present |
| head_ache | int | 0, 1 | 0=Absent, 1=Present |
| age_60_and_above | int | 0, 1 | 0=Under 60, 1=60+ |
| gender | int | 0, 1 | 0=Female, 1=Male |
| test_indication | int | 1, 2, 3 | 1=Contact, 2=Abroad, 3=Other |
| corona_result | int | 0, 1, 2 | 0=Negative, 1=Positive, 2=Other |

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wagan2024optimized,
  title={Optimized Bayesian Network for COVID-19 Diagnosis Leveraging Dirichlet Priors and BDeu Scoring for Accurate Inference},
  author={Wagan, Asif Ali and Talpur, Shahnawaz and Narejo, Sanam and Alqhtani, Samar M and Alqazzaz, Ali and Al Reshan, Mana Saleh and Shaikh, Asadullah},
  journal={PeerJ Computer Science},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Corresponding Author**: Samar M. Alqhtani (smalqhtani@nu.edu.sa)
- **First Author**: Asif Ali Wagan
- **Repository Maintainer**: [Your GitHub username]

## Acknowledgments

- Dataset source: [COVID-19 Dataset for Prediction](https://github.com/nshomron/covidpred)
- Special thanks to the open-source community for the tools and libraries used
- Funding support: Deanship of Graduate Studies and Scientific Research at the University of Bisha
