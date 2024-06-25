import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib



st.set_page_config(page_title="Titanic Survival Prediction",page_icon="🚢",layout='centered')

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Titanic Survival Prediction 🚢</h1>", unsafe_allow_html=True)

st.write("Developing a machine learning model aimed at predicting the survival of passengers on the Titanic. For this purpose, I am utilizing a Decision Tree classifier. This involves analyzing various features of the passengers, such as age, gender, ticket class, and more, to determine their likelihood of survival.")

data = pd.read_csv('titanic.csv').drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis="columns")
df = pd.read_csv('df.csv')
df = df.drop(df.columns[0],axis='columns')
# st.table(df)
col1,col2 = st.columns([1,1])

col1.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Data_Set Structure</h1>", unsafe_allow_html=True)
col1.table(data.head())
col1.write('''
The dataset for this project includes these columns:

Survived: Indicates if a passenger survived (1) or not (0).
Pclass: Passenger's ticket class (1st, 2nd, 3rd).
Sex: Passenger's gender (male or female).
Age: Passenger's age in years.
Fare: Fare paid for the ticket.''')
col2.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Data descreption</h1>", unsafe_allow_html=True)
col2.table(data.describe())

co1,co2 = st.columns([1,1])


x_train, x_test, y_train, y_test = train_test_split(df.drop(['Survived'],axis='columns'),df['Survived'], test_size=0.2,random_state=10)

model = joblib.load("model.jb")
# label = joblib.load("label.jb")
y_pred = model.predict(x_test)


co1.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Confusion Matrix</h1>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test,y_pred),cmap='Blues',annot=True,xticklabels=['Not_survive','Survive'],yticklabels=['Not_survive','Survive'])
ax.set_title("Confusion matrix heat map representation")
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
co1.pyplot(fig)

co2.markdown("<h1 style='text-align: center; color: #5A4FCF;'>Model</h1>", unsafe_allow_html=True)
str = "Model accuracy: "+ str(model.score(x_test,y_test))

co2.code(str)
pclass = co2.number_input(label="Pclass",min_value=1,max_value=3,step=1)
sex = co2.selectbox(label = "Gender",options=['Male',"female"])

if sex == 'Male':
    sex=1
else:
    sex = 0

age = co2.slider(label="Age",min_value=1,max_value=100,step=1)
fare = co2.number_input(label="Fare",step=1)

run = co2.button(label="Predict")
if run:
    value = model.predict([[pclass,sex,age,fare]])
    if value == [1]:
        co2.code("Survives")
    else:
        co2.code('Not Survives')
