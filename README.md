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
The data preparation stage helps to improve model accuracy and reduces the delay in training the model since irrelevant features and erroneous samples are removed. From the insights gleaned during EDA, we have gained a better understanding of the dataset such that we know how to clean and preprocess the dataset to prepare it for the binary classifier model later on.

For data preparation, we performed the following:
1. **Data Cleaning**: using `KNNImputer` to fill in the missing data for the variable `thal` and removing samples with undefined values for `ca`.
2. **Fitting to a Normal Distribution**: Fitting skewed data distributions of continuous variables to a Gaussian distribution.
3. **Data Standardization + One-Hot Encoding**: Rescaling all continuous variables to work on the same scale such that they are equally considered by the predictive model during training, and encoding the categorical variables into a numerical form (binary labels) such that it can be processed by a predictive model.
4. **Feature Selection**: Using a `Random Forest Classifier` to determine the importances of features in contributing to the predictive model's decisions, such that we can decide which features to drop from our moderately high-dimensional dataset with '28' predictors.

---
### 5. Model Selection
First, we trained a basic `Decision Tree Classifier` model on the cleaned dataset to determine the performance of a basic single-tree classifier, evaluated using Precision and Recall derived from a confusion matrix. Unfortunately, the decision tree classifier failed to meet our conditions for a "good" classifier; that is, Recall = 1.000, such that any patient with heart disease is not misidentified as having no heart disease. This is especially crucial in a medical setting, where people's lives can change drastically over a simple diagnosis.

Using the preliminary decision tree classifier as an example, we employed `K-Fold Cross Validation` to compare the predictive performance of the many binary classification models out there on unseen data and pick only the few best ones with the highest mean accuracy score in making predictions. The Top 3 classifier models obtained throgh this method are unsurprisingly all implementations of ensemble learning techniques, which aggregate the outputs of multiple decision trees to make more precise predictions that have reduced errors and/or biases. These models include:
1. Random Forest Classifier
2. Bagging Classifier
3. XG Boost Classifier

After narrowing down our choice of classifier model, we then used `GridSearchCV()` to iteratively calculate for an optimal set of hyperparamters for each of the three models such that these hyperparameters can maximize the predictive quality of the classifier models. This is referred to as "fine-tuning" the classifier models, and after finding the best set of hyperparameters for each model, the fine-tuned models were evaluated against each other using the F1-score metric.

As a result, the classifier model chosen after much deliberation was the `XG Boost` classifier.

---
### 6. Model Prediction
Our chosen model 'XG Boost' was first performed on the train dataset, earning a perfect classification accuracy which sowed initial suspicions of overfitting. It was then tested on the test dataset, and was able to maintain a similar high prediction accuracy, which cleared doubts as our model was able to prove its reliability on unseen data as well.

We then further check the effectiveness of the 'XG Boost' model by utilising the Receiver Operator Characteristic (ROC) curve alongside the Area Under the Curve (AUC) plot to evaluate the separability of binary classification models, that is, the model's ability to tell the different classes apart.

---
### 7. Data-driven Insights & Recommendations
With an accurate and reliable model, we can now analyse the importance of each predictor, providing us some insight into the contributions each feature has towards heart disease. Using feature importance, we plot a barplot, and condensed our findings into two main points:

1. Patients are much more susceptible to heart disease if:
      * more of the heart vessels, which supply blood flow to the heart, are blocked (due to cholesterol build-up in the walls of the vessels),
      * they experience chest pain not only during exercise, but even in a resting state.

2. Patients are much less susceptible to heart disease if:
      * there is no prolonged or frequent chest pain in daily life
      * if chest pain is ONLY experienced during physical activity, then it is less likely to be an indicator of heart-related problems and can be construed to mean more pressure is being put on the heart during exercise, which is normal.
      * for patients with thalessemia, if it is of the reversible defect type, then they are at lower risk of heart disease as there is normal blood flow.
      
      
Finally, we settled on two main recommendations for our project:

1. Our project may have benefitted from basing off a less dated dataset. However we counter with our reasoning that as the dataset only includes biological characteristics which remains more stable over the course of time.
2. Our project may have benefitted from the inclusion of other lifestyle features in the dataset such frequency of excercise or eating habits which has been understood through general knowledge to affect the probability of heart disease as well: Our model may have been more well-rounded with more varied predictors in other areas.

---
### Contributions
1. @paaronoia (Aaron Lim) - Script, Video Editor, Presenter
2. @Brenz2000 (Brendan Thye) - Data Exploration, EDA, Data Preparation, Model Selection, Model Prediction, Data-driven Insights and Reccomendations
3. @jjeennn (Chia Jia En) - EDA, Model Selection, Script, Slides

---
### References
