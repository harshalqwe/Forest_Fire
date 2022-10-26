import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df=pd.read_csv('Fire_forest.csv')

df=df.drop(['X','Y','FFMC','Rain'],axis=1)

def preprocessing(df,task):
    df=df.copy()
    if task=='Regression':
        Y=df['Fire Occurrence']
    elif task=='Classification':
        Y=df['Fire Occurrence'].apply(lambda x: 1 if x>0 else 0)

    X=df.drop('Fire Occurrence',axis=1)

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.80,shuffle=True,random_state=0)

    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train=pd.DataFrame(scaler.transform(X_train),columns=X.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)

    return X_train,X_test,Y_train,Y_test
X=X.astype('int')
Y=Y.astype('int')


X_train,X_test,Y_train,Y_test=preprocessing(df,task='Classification')


nn_classifier_model=MLPClassifier(activation='relu',hidden_layer_sizes=(16,16),n_iter_no_change=100,solver='adam')
nn_classifier_model.fit(X_train,Y_train)

inputt=[int(x) for x in "45 32 44 3 27".split('')]
final=[np.array(inputt)]

b=nn_classifier_model.predict_proba(final)

pickle.dump(nn_classifier_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))