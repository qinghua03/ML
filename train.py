import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


seed=7

df_train=pd.read_csv("./data/train.csv",sep=",",index_col=[0])
#print(df_train)
df_test=pd.read_csv("./data/test.csv",sep=",",index_col=[0])
#print(df_test)


X_train=df_train.drop(columns="satisfaction")
X_test=df_test.drop(columns="satisfaction")
y_train=df_train["satisfaction"]
y_test=df_test["satisfaction"]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=seed, n_estimators=300,
              max_depth=None,max_features="sqrt")


rf_model.fit(X_train, y_train)

print("---------")
fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('model'+ str(fecha) + ".model" , 'wb') as archivo_salida:
    pickle.dump(rf_model, archivo_salida)

print("-----------")
with open('model/my_model.model', "rb") as archivo_entrada:
    modelo_importada = pickle.load(archivo_entrada)

print(modelo_importada)

modelo_importada_pred = modelo_importada.predict(X_test)

print(classification_report(y_test, modelo_importada_pred))

