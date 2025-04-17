import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('heart.csv')

print(df.head())


X = np.array(df.loc[:, df.columns != "output"])
y = np.array(df["output"])

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#neural network

class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_train_scaled.shape[1])
        self.b = np.random.randn(1)
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []
        
        
    def activation(self, x):
        #sigmoid
        return 1/(1+np.exp(-x))
    
    def activation_derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))
    
    def forward(self, X):
        
        hidden_1 = np.dot(X, self.w) + self.b
        activated_1 = self.activation(hidden_1)
        return activated_1
    
    def backward(self, X, y_true):
        
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.activation_derivative(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = X
        
        dL_db = dL_dpred + dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        
        return dL_db, dL_dw
    
    def optimizer(self, dL_db, dL_dw):
        self.w -= self.LR * dL_dw
        self.b -= self.LR * dL_db
        
    def train(self, epochs):
        for epoch in range(epochs):
            random_pos = np.random.randint(0, len(self.X_train))
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])
            
            L = np.sum((y_train_pred - y_train_true) ** 2)
            self.L_train.append(L)
            
            dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])
            
            self.optimizer(dL_db, dL_dw)
            
            L_sum = 0
            for i in range(len(self.X_test)):
                y_test_pred = self.forward(self.X_test[i])
                L_sum += np.sum((y_test_pred - self.y_test[i]) ** 2)
            
            self.L_test.append(L_sum)
            
        return "Training complete"





LR = 0.1
EPOCHS= 1000


nn = NeuralNetworkFromScratch(LR, X_train_scaled, y_train, X_test_scaled, y_test)
nn.train(EPOCHS)

print(nn.L_train[0], nn.L_train[-1])
print(nn.L_test[0], nn.L_test[-1])
        
sns.lineplot(x=list(range(len(nn.L_test))), y=nn.L_test, label="Test Loss")

import matplotlib.pyplot as plt
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


total = X_test_scaled.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_predicted = nn.forward(X_test_scaled[i])
    y_pred = np.round(y_predicted)
    y_preds.append(y_pred)
    if y_pred == y_true:
        correct += 1
print("Accuracy: ", correct/total)
print("Confusion Matrix: ")
cm = confusion_matrix(y_test, y_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
