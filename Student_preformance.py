#import libraries
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\python\student performance predictor\study_performance.csv")
#for displaying top 5 rows
print(data.head())
print(data["test_preparation_course"].nunique())
#here mapping values categorical to numeric
data["test_preparation_course"]=data["test_preparation_course"].map({"none":0,"completed":1})
data["lunch"]=data["lunch"].map({"standard":0,"free/reduced":1})

#for spliting data into training,testing we importing train-test-split module
from sklearn.model_selection import train_test_split
X=data[["reading_score","writing_score","test_preparation_course","lunch"]]
y=data["math_score"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Implementing LinearRegression model
from sklearn.linear_model import LinearRegression
model=LinearRegression()#model creation
model.fit(X_train,y_train)#model training
y_pred=model.predict(X_test)#prediction
#print(y_pred)
#For evaluting our model we importing MAE &R2_score
from sklearn.metrics import mean_absolute_error,r2_score
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean Absolute error:",mae)
print("R2 Score:",r2)
#plot for ml model
plt.figure(figsize=(6,6))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual marks")
plt.ylabel("predicted marks")
plt.title("Actual vs Predicted")
plt.show()
#Implementing decision tree
from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor()
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
#for evaluating DecisionTree 
mae2=mean_absolute_error(y_test,y_pred2)
r2_=r2_score(y_test,y_pred2)
print("mean Absolute error in DecisionTree :",mae2)
print("R2 Score in DecisionTree :",r2_)
print("Feature importance:",model2.feature_importances_)
#ploting begins...
plt.figure(figsize=(6,6))
plt.bar(data["reading_score"],data["writing_score"])
plt.xlabel("reading score")
plt.ylabel("writing score")
plt.show()

##############
import pickle
pickle.dump(model,open("student_model.pkl","wb"))

