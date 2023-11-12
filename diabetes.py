import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from joblib import dump, load


df=pd.read_csv('./diabetes.csv')
#sns.pairplot(df,hue='Outcome')

#Isı grafikleri
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()

# Verisetinde toplam diyabet olan ve olmayan kişi sayıları
#df.Outcome.value_counts().plot(kind='bar')
#plt.xlabel("Diabetes or Not")
#plt.ylabel("Count")
#plt.title("Outcome ")
#plt.show()

X=df.drop('Outcome',axis=1)
y=df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

# Standart Scaler kısmı
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

#Farklı sayıda denemelele ideal k katsayısının 18 olduğunu bulduk.
knn=KNeighborsClassifier(n_neighbors=18,metric='euclidean',p=2)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

#KNN skoru:
#print(knn.score(X_test,y_test))

target_names = ['Diabetes', 'Normal']
print(classification_report(y_test, y_pred, target_names=target_names))
dump(knn, 'knn.pkl')