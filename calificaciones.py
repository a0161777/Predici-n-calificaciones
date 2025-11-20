import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificación  ''') # el titulo que se agrega a la app
st.image("Examen_25.jpg", caption="Predicción de calificación en examen.") # pie de figura
# nombre en el cual se guardo la imagen
st.header('Datos') # encabezado

def user_input_features():
  # Entrada
  Horas_de_estudio= st.number_input("Horas de estudio:", min_value=0, max_value=100, value = 0)
  Horas_de_sueño = st.number_input('Horas de sueño',  min_value=0, max_value=100, value = 0)
  Asistencia = st.number_input('Asistencia:', min_value=0, max_value=100, value = 0)
  Calificaciones_previas = st.number_input('Calificaciones previas:', min_value=0, max_value=100, value = 0)

  user_input_data = {'hours_studied': Horas_de_estudio,
                     'sleep_hours': Horas_de_sueño,
                     'attendance_percent': Asistencia, 
                     'previous_scores':  Calificaciones_previas
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()


datos =  pd.read_csv('student_exam_scores_prediction.csv', encoding='latin-1')

datos.columns = datos.columns.str.strip()

X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613777)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1= LR.coef_
b0= LR.intercept_
prediction=b0+b1[0]*df['hours_studied']+b1[1]*df['sleep_hours']+b1[2]*df['attendance_percent']+b1[3]*df['previous_scores']



st.subheader('Calculo de calificación')
st.write('La calificación de la persona es: ', prediction) 
