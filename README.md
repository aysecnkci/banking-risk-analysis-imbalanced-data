# ML-Based Customer Risk Analysis for Banking: Achieving 80% Recall on Imbalanced Data

## Project Overview

This project focuses on customer risk prediction in a banking environment using machine learning models. The primary challenge was dealing with imbalanced data where the minority class represents customers with a higher risk of default. Our goal was to achieve high recall, specifically targeting 80%, while maintaining acceptable levels of precision and accuracy.

## Objectives

- **Goal**: Achieve 80% recall on high-risk customers.
- **Dataset**: Imbalanced banking dataset.
- **Models Used**: CART, Random Forest (RF), Gradient Boosting Machine (GBM), LightGBM, and BalancedRandomClassifier.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.

## Approach

1. **Data Preprocessing**:
   - Handled missing data.
   - Performed feature scaling and encoding for categorical variables.
   - Addressed the class imbalance using specialized techniques.

2. **Imbalanced Data Handling**:
   - Implemented **RandomUnderSampler** to reduce majority class size.
   - Used **TomekLinks** to remove overlapping data points and further refine the dataset.
   
3. **Modeling**:
   - Tried several machine learning models: **CART**, **RF**, **GBM**, and **LightGBM**.
   - The most effective model for handling imbalance was **BalancedRandomClassifier**.

4. **Hyperparameter Optimization**:
   - Applied hyperparameter tuning to the BalancedRandomClassifier using grid search to optimize performance.

5. **Model Evaluation**:
   - The BalancedRandomClassifier provided the best results.
     

## Dataset

The dataset contains anonymized banking data with several features such as:

- **Customer Demographics**: Age, gender, occupation, etc.
- **Financial Indicators**: Credit history, balance, transaction patterns.
- **Target Variable**: Customer risk level.

For privacy reasons, the dataset is not included in this repository.

## Results

- **Best Model**: BalancedRandomClassifier
- **Final Performance Metrics**:
  - **Accuracy**: 74%
  - **Precision**: 72%
  - **Recall**: 80%
  - **F1-Score**: 76%

## How to Run

### Prerequisites

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### Running the Code

1. Clone the repository:

```bash
git clone https://github.com/aysecnkci/banking-risk-analysis-imbalanced-data.git
```

2. Run the Jupyter notebook to preprocess the data, train the model, and evaluate it:

```bash
jupyter notebook risk_analysis_banking_imbalanced.ipynb
```

### Repository Structure

```plaintext
├── README.md
├── requirements.txt
├── notebooks/
│   └── risk_analysis_banking_imbalanced.ipynb
```


## Future Work

- Experiment with deep learning models to improve recall.
- Further tune hyperparameters to explore better performance.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the contributors and the machine learning community for resources and support.
