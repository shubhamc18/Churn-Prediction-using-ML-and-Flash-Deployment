# Problem Setting
To develop a model that helps them retain customers of a telecom provider as customers are quick to switch to different providers based on the benefits offered by many other providers

# Methodology
Data cleaning and exploratory analysis 
Feature selection using data exploration and visualization
Model building to predict churning of customers
Model deployment using Flask API

# Dataset details
Churn.- Customers who left within the last month 
Details of services used by customers — phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV
Customer account information - contract, payment method, paperless billing, monthly charges, and total charges.
Demographics information about customers — gender, age range, and if they have partners and dependents.

# Feature Selection of attributes for model building
We use sampling technique called SMOTE(Synthetic Minority Oversampling Technique) to create synthetic data points for minority class (Yes) which is also used for feature selection
We use one-hot encoding for our feature selection because the models don’t interpret categorical data. Hence, we need to convert them into non-categorical data and we assign numbers using the function:
replaceStruct = {"Churn":     {"No": 0, "Yes": 1 }  }
oneHotCols = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
replaced_data=df.replace(replaceStruct, inplace=True)
replaced_data=pd.get_dummies(df, columns=oneHotCols)
replaced_data.head()

# Model Building
We have built models on the SMOTE and standardized data 
We are using the 30 features from the 47 features
We will build 4 different models – Decision Trees, Logistic Regression , Adaboost and Logistic Regression

# Model Insights
Decision Tree Model model has a fairly low accuracy in determining churning rate of the customer
Logistic regression model, which is overall slightly better performer in terms of predicting churners but it's miss-classification rate is too high.
Random Forest Model whose miss-classification rate for predicting churners is 3rd lowest and also overall mis-classification rate is also low.
Ada Boost Model whose miss-classification rate for predicting churners is 2nd lowest is the best model

