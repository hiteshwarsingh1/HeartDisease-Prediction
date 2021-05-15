import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("data.csv")
Xn = df[['age','sex','cp','rbp','chol','fbs','recg','mhr','exang','oldpeak','slope']]
scaler = StandardScaler()
X = scaler.fit_transform(Xn)
# X = preprocessing.normalize(X_n, norm='max')
Y = df.target
prediction = []
split =[]
from sklearn.neighbors import KNeighborsClassifier
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
for sr in range(1,5):
    sr=sr/10
    split.append(sr)
    i=1
    iteration=[]
    p = []
    while(i<=500):
        iteration.append(i)
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=sr)
        gnb = KNeighborsClassifier()
        gnb.fit(X_train, Y_train)
        acc = gnb.score(X_test,Y_test)*100
        p.append(acc)
        i=i+1
    prediction.append(p)
plt.scatter(iteration,prediction[0])
plt.scatter(iteration,prediction[1])
plt.scatter(iteration,prediction[2])
plt.scatter(iteration,prediction[3])
plt.xlabel("iterations")
plt.ylabel("acuraccy")
plt.legend(["0.1","0.2","0.3","0.4"],title="Test Set Ratio",loc ="lower right")

plt.show()

plt.plot(iteration,prediction[1])
plt.show()
