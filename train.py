import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 
 
DATA_PATH =r"data\credit_risk_dataset.csv" 
MODEL_PATH ="models\\model.joblib" 
REPORT_PATH = "reports\\comparison.csv" 
 
FEATURES =["annual_income", 
           "credit_score", 
           "loan_amount", 
           "existing_debt", 
           "late_payments", 
           "previous_defaults"] 
 
TARGET="credit_risk" 

df=pd.read_csv(DATA_PATH) 
print(df.head()) 
 
print("Rows in dataset:", len(df)) 
 
print("Risk Classes:", df[TARGET].unique()) 
 
X=df[FEATURES] 
 
y=df[TARGET] 
 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42) 
 
models={"Logistic Regression": LogisticRegression(), 
        "Decision Tree": DecisionTreeClassifier()} 

results = []
     
for name, classifier in models.items(): 
        pipeline =Pipeline([ 
                ("imputer" , SimpleImputer(strategy="median")), 
                ("scaler",StandardScaler()), 
                ("classifier",classifier) 
        ]) 
         
        pipeline.fit(X_train,y_train) 
        y_pred=pipeline.predict(X_test) 
        accuracy = accuracy_score(y_test,y_pred) 
 
        results.append({"models" : name , "accuracy" :round(accuracy,3)}) 
 
result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False) 
 
print(result_df.to_string(index=False))