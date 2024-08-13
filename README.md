# Loan Default Prediction Project

This repository contains the work related to predicting loan defaults using various machine learning models. The project was part of the CS610 Applied Machine Learning course, and it aims to enhance the accuracy of risk assessments for banks and lenders.

## Project Overview

Banks have been facing significant challenges in accurately assessing the risk of loan defaults. The goal of this project is to improve these assessments by applying advanced machine learning models to predict the likelihood of a loan default. This will help financial institutions make better-informed decisions, fostering a more stable and inclusive financial ecosystem.

## Repository Structure

The repository contains the following files and notebooks:

- **`AML Report.pdf`**: The final report detailing the business problem, dataset, preprocessing steps, model selection, and results.
- **`KNN_full.ipynb`**: Implementation and analysis using the K-Nearest Neighbors (KNN) model.
- **`LGBM_Full_Clean.ipynb`**: Implementation and analysis using the Light Gradient Boosting Machine (LGBM) model, which was selected as the final model for deployment.
- **`Logi_Regression_Clean.ipynb`**: Implementation and analysis using Logistic Regression.
- **`SVM_full.ipynb`**: Implementation and analysis using the Support Vector Machine (SVM) model.
- **`TabNetClassifier_full.ipynb`**: Implementation and analysis using the TabNet Classifier.
- **`XGboost_full.ipynb`**: Implementation and analysis using XGBoost.

## Dataset

The dataset used in this project includes:

1. **Application Train/Test**: Primary dataset containing static data for all applications.
2. **Bureau**: Records of preceding credits associated with clients, reported to the Credit Bureau.
3. **Previous Application**: Records of prior applications for Home Credit loans.

### Preprocessing Steps

1. **Handling Missing Values**: 
   - Automatic handling by LightGBM during training.
   - Median imputation for other models.
   
2. **Feature Engineering**: 
   - New features were created to capture various financial aspects of the applicants, such as `CREDIT_INCOME_PERCENT`, `ANNUITY_INCOME_PERCENT`, `CREDIT_TERM`, and more.
   
3. **Handling Unrealistic Values**: 
   - Identified and replaced unrealistic values with `NaN` to maintain data integrity.

## Models and Evaluation

### Models Implemented:
- **Logistic Regression**: Known for its interpretability in binary classification.
- **K-Nearest Neighbors (KNN)**: Effective in capturing non-linear relationships.
- **Support Vector Machine (SVM)**: Handles high-dimensional spaces and non-linear decision boundaries.
- **XGBoost**: Known for its performance in extensive datasets and robust regularization techniques.
- **TabNetClassifier**: Utilizes decision layers and attention mechanisms for feature selection.
- **Light Gradient Boosting Machine (LGBM)**: The final model selected for its robust performance, handling of categorical features, and computational efficiency.

### Model Selection and Final Decision

The final selection was based on the ROC-AUC score, which evaluates the model's ability to distinguish between classes. LGBM was chosen as the final model due to its balanced performance, computational efficiency, and compatibility with the dataset's characteristics.

## Results

- **LGBM** achieved the best balance between training and validation AUC scores, demonstrating good generalization capability.

## Future Considerations

- **Data Integration**: Incorporate additional data sources like social media activity and utility payments.
- **Advanced Feature Engineering**: Use techniques such as polynomial features, PCA, and interaction features to improve prediction accuracy.
- **Model Improvement**: Explore CatBoost and model ensembling techniques to further enhance predictive power.

## Conclusion

This project successfully applied machine learning techniques to improve loan default predictions. The use of LGBM, combined with robust preprocessing and feature engineering, resulted in a model that is both accurate and efficient for financial risk assessments.
