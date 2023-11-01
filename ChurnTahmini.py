import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as mt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


csv = pd.read_csv("Churn_Modelling.csv",sep=";",dtype={"EstimatedSalary":float})   #Veri setini okudum.

df = pd.DataFrame(csv)


pd.set_option('display.max_columns',20)                                           #Verilerin konsolda düzgün gözükmesi için ayarlar yaptım.
pd.set_option('display.width',1000)
print("Veri setinin ilk 5 satırı.")
print(df.head())                                                                  #Veri setinin ilk 5 satırını bastırdım.


df = df.drop(columns=["RowNumber","CustomerId","Surname"])                        #Kullanılmayacak değişkenleri çıkardım.
print("\n\nModele girmeyecek olan değişkenler çıkarıldı.")
print(df.head())                                                                  #Kullanılmayacak veriler çıktıktan sonra DataFrame'in ilk 5 satırı.

print("\n\nVeri setinde null veri var mı yok mu kontrolü.")
print(pd.DataFrame(df.isnull().sum(),columns=["Null Veri Sayısı"]))               #Veri setinde null veriler var mı kontrol ettim.
"""
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',linewidths=.2)               #Değişkenler arası korelasyon ilişkisi.
plt.title("\n\nChurn Modelling Korelasyon Heatmap")
plt.show()
"""
bagimsiz_degiskenler = df.iloc[:,0:10]                                            #Bağımsız değişkenler ve hedef değişkenin birbirinden ayrılması.
hedef_degisken = df.iloc[:,10:]

bagimsiz_degiskenler["isMale"] = bagimsiz_degiskenler["Gender"].map({"Male":1,"Female":1})
                                                                                                           #Veri setinde bulunan string değişkenlerin modele katılabilmesi
bagimsiz_degiskenler[["France","Germany","Spain"]] = pd.get_dummies(bagimsiz_degiskenler["Geography"])     #için sayılara çevirdim.
bagimsiz_degiskenler = bagimsiz_degiskenler.drop(columns=["Gender","Geography"])
print("\n\nString değişkenler modele katılabilmesi için sayılara dönüştürüldü. ")
print(bagimsiz_degiskenler.head())

print("\n\nSkalası geniş olan değişkenlere normalizasyon uygulanarak 0 ve 1 arasında indirgendi.")
normalization = lambda x:(x-x.min()) / (x.max()-x.min())                                                   #Veri setinde skalası geniş olan Balance, EstimatedSalary,
transformColumns = bagimsiz_degiskenler[["Balance","EstimatedSalary","CreditScore"]]                       #CreditScore değişkenlerine normalizasyon uygulayarak 0 ve 1 
bagimsiz_degiskenler[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)        #arasına indirgedim.
print(bagimsiz_degiskenler.head())


x_train,x_test,y_train,y_test = train_test_split(bagimsiz_degiskenler,hedef_degisken,test_size=0.25,random_state=0) #Veri setini train ve test olarak ayırdım.


kararAgaci = DecisionTreeClassifier()                                    #Karar Ağacı(Decision Tree) algoritması.
kararAgaci.fit(x_train,y_train)                                          #Modelin eğitilmesi.
kararAgaci_tahmin = kararAgaci.predict(x_test)                           #Eğitilen modelin churn tahmini.
kararAgaci_skor = accuracy_score(y_test,kararAgaci_tahmin)               #Doğruluk skoru.

lojistikRegresyon = LogisticRegression()                                 #Lojistik Regresyon(Logistic Regression) algoritması.
lojistikRegresyon.fit(x_train,y_train)                                   #Modelin eğitilmesi.
lojistikRegresyon_tahmin = lojistikRegresyon.predict(x_test)             #Eğitilen modelin churn tahmini.
lojistikRegresyon_skor = accuracy_score(y_test,lojistikRegresyon_tahmin) #Doğruluk skoru.

gnb = GaussianNB()                                                       #Gaussian Naive Bayes algoritması.
gnb.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
gnb_tahmin = gnb.predict(x_test)                                         #Eğitilen modelin churn tahmini.
gnb_skor = accuracy_score(y_test,gnb_tahmin)                             #Doğruluk skoru.

knn = KNeighborsClassifier( metric='minkowski')                          #K-En Yakın Komşu(KNN) algoritması.
knn.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
knn_tahmin = knn.predict(x_test)                                         #Eğitilen modelin churn tahmini.
knn_skor = accuracy_score(y_test,knn_tahmin)                             #Doğruluk skoru.

rastgeleOrman = RandomForestClassifier()                                 #Rastgele Ormanlar(Random Forest) algoritması.
rastgeleOrman.fit(x_train,y_train)                                       #Modelin eğitilmesi.
rastgeleOrman_tahmin = rastgeleOrman.predict(x_test)                     #Eğitilen modelin churn tahmini.
rastgeleOrman_skor = accuracy_score(y_test,rastgeleOrman_tahmin)         #Doğruluk skoru.

svm = SVC()                                                              #Destek Vektör Makineleri(SVM) algoritması.
svm.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
svm_tahmin = svm.predict(x_test)                                         #Eğitilen modelin churn tahmini.
svm_skor = accuracy_score(y_test,svm_tahmin)                             #Doğruluk skoru.

xgboost = xgb.XGBClassifier()                                            #XGBoost algoritması.
xgboost.fit(x_train, y_train)                                            #Modelin Eğitilmesi.
xgboost_tahmin = xgboost.predict(x_test)                                 #Eğitilen modelin churn tahmini.
xgboost_skor = xgboost.score(x_test,xgboost_tahmin)                      #Doğruluk skoru.

#Doğruluk değerlerinin yazdırılması.
print("\n\nAlgoritmaların Doğruluk Değerleri")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "Doğruluk":[kararAgaci_skor, lojistikRegresyon_skor, gnb_skor, knn_skor, rastgeleOrman_skor,svm_skor,xgboost_skor]}))


#Cross validation yaparak overfitting kontrolü.
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGBoost', XGBClassifier()))

results_boxplot = []
names = []
results_mean = []
results_std = []
p,t = bagimsiz_degiskenler.values, hedef_degisken.values.ravel()
for name, model in models:
    cv_results = cross_val_score(model, p,t, cv=10)                       #Veri setini 10 parçaya ayırarak doğruluk skoru.
    results_boxplot.append(cv_results)
    results_mean.append(cv_results.mean())                                #Doğruluk skorlarının ortalaması.
    results_std.append(cv_results.std())                                  #Doğruluk skorlarının standart sapması.
    names.append(name)

#Algoritmaların doğruluklarının ortalaması ve standart sapmasını yazdırdım.
print("\n\n Cross Validation sonrası algoritmaların doğruluk ortalaması ve doğruluk standart sapması.")
print(pd.DataFrame({"Algoritmalar":names,"Doğruluk Ortalaması":results_mean,
              "Doğruluk Standart Sapması":results_std}))


kararAgaci_hassasiyet = mt.precision_score(y_test,kararAgaci_tahmin)
lojistikRegresyon_hassasiyet = mt.precision_score(y_test,lojistikRegresyon_tahmin)     #Algoritmaların hassasiyet değerlerinin hesaplanması.
gnb_hassasiyet = mt.precision_score(y_test,gnb_tahmin)
knn_hassasiyet = mt.precision_score(y_test,knn_tahmin)
rastgeleOrman_hassasiyet = mt.precision_score(y_test,rastgeleOrman_tahmin)
svm_hassasiyet = mt.precision_score(y_test,svm_tahmin)
xgboost_hassasiyet = mt.precision_score(y_test,xgboost_tahmin)

print("\n\nAlgoritmaların hassasiyet(precision) değerleri.")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "Hassasiyet(Precision)":[kararAgaci_hassasiyet, lojistikRegresyon_hassasiyet, gnb_hassasiyet, knn_hassasiyet, rastgeleOrman_hassasiyet,svm_hassasiyet,xgboost_hassasiyet]}))


kararAgaci_duyarlilik = mt.recall_score(y_test,kararAgaci_tahmin)
lojistikRegresyon_duyarlilik = mt.recall_score(y_test,lojistikRegresyon_tahmin)        #Algoritmaların duyarlılık değerlerinin hesaplanması.
gnb_duyarlilik = mt.recall_score(y_test,gnb_tahmin)
knn_duyarlilik = mt.recall_score(y_test,knn_tahmin)
rastgeleOrman_duyarlilik = mt.recall_score(y_test,rastgeleOrman_tahmin)
svm_duyarlilik = mt.recall_score(y_test,svm_tahmin)
xgboost_duyarlilik = mt.recall_score(y_test,xgboost_tahmin)

print("\n\nAlgoritmaların duyarlılık(recall) değerleri.")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "Duyarlılık(Recall)":[kararAgaci_duyarlilik, lojistikRegresyon_duyarlilik, gnb_duyarlilik, knn_duyarlilik, rastgeleOrman_duyarlilik,svm_duyarlilik,xgboost_duyarlilik]}))

kararAgaci_f1 = mt.f1_score(y_test,kararAgaci_tahmin)
lojistikRegresyon_f1 = mt.f1_score(y_test,lojistikRegresyon_tahmin)                   #Algoritmaların f1 değerlerinin hesaplanması.
gnb_f1 = mt.f1_score(y_test,gnb_tahmin)
knn_f1 = mt.f1_score(y_test,knn_tahmin)
rastgeleOrman_f1 = mt.f1_score(y_test,rastgeleOrman_tahmin)
svm_f1 = mt.f1_score(y_test,svm_tahmin)
xgboost_f1 = mt.f1_score(y_test,xgboost_tahmin)

print("\n\nAlgoritmaların f1 skoru değerleri.")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "F1 Skoru":[kararAgaci_f1, lojistikRegresyon_f1, gnb_f1, knn_f1, rastgeleOrman_f1,svm_f1,xgboost_f1]}))

tn, fp, fn, tp = mt.confusion_matrix(y_test,kararAgaci_tahmin).ravel()
kararAgaci_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,lojistikRegresyon_tahmin).ravel()         #Algoritmaların özgüllük değerlerinin hesaplanması.
lojistikRegresyon_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,gnb_tahmin).ravel()
gnb_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,knn_tahmin).ravel()
knn_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,rastgeleOrman_tahmin).ravel()
rastgeleOrman_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,svm_tahmin).ravel()
svm_ozgulluk = tn / (tn+fp)

tn, fp, fn, tp = mt.confusion_matrix(y_test,xgboost_tahmin).ravel()
xgboost_ozgulluk = tn / (tn+fp)

print("\n\nAlgoritmaların özgüllük(specificity) değerleri.")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "Özgüllük(Specificity)":[kararAgaci_ozgulluk, lojistikRegresyon_ozgulluk, gnb_ozgulluk, knn_ozgulluk, rastgeleOrman_ozgulluk,svm_ozgulluk,xgboost_ozgulluk]}))