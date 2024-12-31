import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from google.cloud import storage
import json
from google.cloud import bigquery
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

storage_client = storage.Client()
bucket = storage_client.bucket("bank-loan-mlops")



def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Data')
    return df

def clean_df(df):
  df = df.drop(['ZIP Code'], axis = 1)
  df['Experience'] = df['Experience'].apply(abs)
  df['CCAvg'] = df['CCAvg']*12
  X = df.drop(['Personal Loan'], axis=1)
  y = df['Personal Loan']

  return X, y

def preprocess_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
  cat_features =  ['CD Account','Education','Family','Securities Account','Online','Securities Account']
  num_features =  ['Age','Experience','Income','CCAvg','Mortgage']
  numeric_transformer = StandardScaler()
  oh_transformer = OneHotEncoder(drop='first')

  preprocessor = ColumnTransformer(
      [
          ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features)
      ]
  )

  X_train=preprocessor.fit_transform(X_train)
  X_test=preprocessor.transform(X_test)

  return X_train, X_test, y_train, y_test

def train_model(model_name, X_train, y_train):
    if model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',n_estimators=300,max_depth=5,learning_rate=0.1,
                           colsample_bytree=0.5)


    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline    

def get_classification_report(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name, pipeline):
    artifact_name = model_name+'_model.joblib'
    dump(pipeline, artifact_name)
    # Uncomment below lines for cloud execution
    model_artifact = bucket.blob('bank_model_artifact/'+artifact_name)
    model_artifact.upload_from_filename(artifact_name)


def load_model_artifact(file_name):
    blob = bucket.blob("ml-artifacts/" + file_name)
    blob.download_to_filename(file_name)
    return load(file_name)

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    client = bigquery.Client()
    table_id = "beaming-botany-436322-f7.ML_OPS.bank_loan_model_metrics"
    table = bigquery.Table(table_id)

    row = {"algo_name": algo_name, "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'), "model_metrics": json.dumps(model_metrics)}
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)


def main():
    input_data_path = "gs://bank-loan-mlops/Bank_Personal_Loan_Modelling.xlsx"
    model_name='xgboost'
    df = load_data(input_data_path)
    X,y = clean_df(df)
    X_train, X_test, y_train, y_test = preprocess_data(X,y) 
    pipeline = train_model(model_name, X_train, y_train)
    accuracy_metrics = get_classification_report(pipeline, X_test, y_test)
    training_time = datetime.now()
    write_metrics_to_bigquery(model_name, training_time, accuracy_metrics)
    save_model_artifact(model_name,pipeline)

if __name__ == "__main__":
    main()

# main()