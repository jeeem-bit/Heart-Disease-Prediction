# SC1015 Mini Project

School of Computer Science & Engineering \
Nanyang Technology University \
Lab Group: A140 \
Team: 8

Contributors
1. Aaron Jerome Lim Li Yang
2. Brendan Yap Ming Thye
3. Chia Jia En

---
### Project Description
This project aims to predict the presence of heart disease in individuals using the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). This repository contains the source code (Jupyter Notebook `.ipynb` files), dataset, data description text file, and the final video presentation.

---
### Table of Contents
1. Problem Definition
2. Data Exploration
3. Exploratory Data Analysis (EDA)
4. Data Preparation
5. Model Selection
6. Model Prediction
7. Data Driven Insights & Recommendations
8. References

---
### 1. Problem Definition
According to the World Health Organization, heart disease is the leading cause of death worldwide, causing 17.9 million deaths annually. Similarly, cardiovascular disease is prevalent in Singapore, as 21 people die from cardiovascular disease every day, accounting for 32% of deaths in 2021, according to the Singapore Heart Foundation. Like all other diseases, early detection of disease allows for early intervention and a higher chance of survival. As such, our group aims to accurately predict heart disease in any individual and fight against this silent killer.

**Dataset:** [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) \
**Problem Statement:** Predicting a Patient's Propensity towards developing Heart Disease

---
### 2. Data Exploration
We first performed a simple data exploration on the Heart Disease dataset to get a glimpse of the 14 features and 1025 samples we are working with. 

From the `data_description.txt` file, the features can be defined as follows:
> 1. `age`: age in years
> 2. `sex`: sex
> 3. `cp`: chest pain type
> 4. `trestbps`: resting blood pressure (in mm Hg on admission to the hospital)
> 5. `chol`: serum cholestoral in mg/dl
> 6. `fbs`: fasting blood sugar in mg/dl
> 7. `restecg`: resting electrocardiographic results
> 8. `thalach`: maximum heart rate achieved
> 9. `exang`: exercise induced angina
> 10. `oldpeak`: ST depression induced by exercise relative to rest
> 11. `slope`: the slope of the peak exercise ST segment
> 12. `ca`: number of major vessels colored by flourosopy
> 13. `thal`: the presence of the blood disorder thalassemia
> 14. `target`: diagnosis of heart disease (angiographic disease status)
    * 0: < 50% diameter narrowing (no heart disease)
    * 1: > 50% diameter narrowing (heart disease)

The presence of heart disease in a patient is defined using the response variable `target` which takes the angiographic disease status determining the narrowing of any major heart vessels to diagnose heart disease using binary labels. As such, the problem can be defined as a **Binary Classification** problem.

---
### 3. Exploratory Data Analysis (EDA)
In performing EDA, we observe if there exists any relationships and/or patterns between the features and the response variable `target` which may provide an indication of which features may contribute more to the model's predictive power during the classification stage. Additionally, EDA assists us with making decisions for data cleaning later on e.g., Is there a class imbalance in the response variable which would necessitate data resampling? Does the skewness of the data distribution for the features necessitate data normalization?

For our EDA, we performed the following:
1. **Univariate Analysis**
    * Response variable `target`: Count plot to determine if class imbalance exists.
    * Continuous variables: Histogram w/ KDE plot to compare distributions and determine if normalization is required.
    * Categorical variables: Count plots to check for unknown classes which may require data cleaning.
2. **Bivariate Analysis**
    * Continuous variables against `target`: box plot analysis
    * Categorical variables against `target`: count plot analysis
3. **Correlation matrices**
    * Continuous variables against `target`: Point-Biserial Correlation
    * Categorical variables against `target`: Chi-Square Test

---
### 4. Data Preparation
