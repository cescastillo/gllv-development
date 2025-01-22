
from openpyxl import reader, load_workbook, Workbook
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import joblib


# Para correr el programa localmente usar streamlit run main.py
st.set_page_config(page_icon="💬",layout="centered", page_title="Tax Code Encoder")

st.title("Encode Tax Code")
st.subheader("Instrucciones de uso")
st.write("Paso 1: Selección del modelo del impuesto:")
st.write("En el panel lateral de la aplicación encontrarás un menú desplegable con el título Selecciona el impuesto. Aquí deberás seleccionar el tipo de impuesto para el cual deseas realizar la predicción del código. Las opciones disponibles son: 1120, 1065, 1040 y 1120S")
st.write("")
st.write("Paso 2: Cargar el archivo de Datos")
st.write("Debajo del menú de selección de impuesto, verás un botón que dice Carga el archivo. Aquí deberás subir el archivo de datos financieros. La aplicación SOLO acepta archivos en formato Excel .xlsx")
st.write("Consideraciones importantes: Asegúrate de que el archivo no exceda los 200 MB, El archivo debe contener una hoja llamada Sheet1 o Trial Balance, donde se encuentran los datos relevantes")
st.write("")
st.write("Paso 3: Visualización de la Predicción")
st.write("Una vez que hayas subido el archivo y seleccionado el impuesto, la aplicación procesará los datos y aplicará el modelo correspondiente. Luego de unos segundos, se mostrará en pantalla una tabla con las siguientes columnas:")
st.text("Predicted Tax Code: El código fiscal predicho.")
st.text("Account Number: El número de cuenta.")
st.text("Account Description: La descripción de la cuenta.")
st.text("Debit: La cantidad debitada.")
st.text("Credit: La cantidad acreditada.")
st.text("Probability: La probabilidad de acierto que tiene la predicción.")



#DEFININEDO VARIABLES Y FUNCIONES
archivo_excel = "SUPERMERCADO TALPA 7, INC_2023_FINANCIAL STATEMENTS_TRIAL BALANCE 2023.xlsx"
max_file_size = 10 * 1024 * 1024 #10MB
logoGllv = "./imgs/GLLV Logo.png"
df_completo = None

def probablity_color(val):
    color = 'red' if val < 0.50 else 'white'
    return f'background-color: {color}'

def classify_ctg(account_number):
   if pd.isna(account_number) or not account_number[0].isdigit():
      return "Unknown"
   category_map = {
        '1': 'CA',
        '2': 'CL',
        '3': 'EQ',
        '4': 'INC',
        '7': 'INC',
        '5': 'EXP',
        '6': 'EXP',
        '8': 'EXP',
        '9' : ''

   }
   return category_map.get(account_number[0], "Unknown")


       
#Usar para generar los modelos comprimidos
# joblib.dump(model1,'model_1120S_compress.pkl', compress=3)
# print("Modelo 1120S comprimido")


st.sidebar.title("Por favor carga el archivo y selecciona el modelo del impuesto")


add_selectbox = st.sidebar.selectbox('Selecciona el impuesto',('1120','1065','1040','1120S'))
print(f"{add_selectbox}")

uploaded_file = st.sidebar.file_uploader("Carga el archivo", type={"xlsx","csv","txt"})


if uploaded_file is not None:
 try:
     df_completo = pd.read_excel(uploaded_file, sheet_name='Sheet1')
     print("Hoja 'Sheet1' cargada exitosamente.")
 except ValueError:
     #Si falla, intentar con la hoja 'Hoja1'
     try:
        df_completo = pd.read_excel(uploaded_file, sheet_name='Hoja1')
        print("Hoja 'Hoja1' cargada exitosamente.")
     except ValueError:
        #Si tambíen falla, intentar con la hoja 'Trial Balance'
        try:
           df_completo = pd.read_excel(uploaded_file, sheet_name='Trial Balance')
           print("Hoja 'Trial Balance' cargada exitosamente.")
        except ValueError:
           #Si ninguna hoja se encuentra, lanzar un error
           raise ValueError("No se pudo encontrar 'Sheet1', 'Hoja1' ni 'Trial Balance' en el archivo Excel")
         
   

#Mostrar las columnas disponibles
if isinstance(df_completo, pd.DataFrame):
 print("Columnas del archivo subido: ", df_completo.columns.to_list())

 

 
 def load_model_from_dropbox(url):
   # Descargar el archivo desde dropbox
   response = requests.get(url)
   response.raise_for_status() # Verificar que no haya errores en la solicitud
   # Cargar el modelo desde los bytes descargados
   return joblib.load(io.BytesIO(response.content))
 
 # URLs de Dropbox con el formato dl=1 para descarga directa
 dropbox_model1 = 'https://www.dropbox.com/scl/fi/ikzc34rqejxgtwqvjwqpm/TD-IDFmodel_1120S.pkl?rlkey=v7s2jhr80p1g4fa61t776xqde&st=9ihfjgl7&dl=1'
 dropbox_model2 = 'https://www.dropbox.com/scl/fi/uvf77fiwodtggyalcf82m/TF-IDFmodel_1120.pkl?rlkey=3h8x49cnlog9j9eyt1hpz43zq&st=6cr79321&dl=1'
 dropbox_model3 = 'https://www.dropbox.com/scl/fi/4qowk4sksqbfd4cltf90i/TF-IDFmodel_1065.pkl?rlkey=owsrnsmygrqj705n6mpvck872&st=s2r2xg5o&dl=1'
 dropbox_model4 = 'https://www.dropbox.com/scl/fi/rnffy8ax4njzftm9jrc72/TF-IDFmodel_1040.pkl?rlkey=f8y7w4fqgl0kstkh7u3742kl7&st=5gtwosyh&dl=1'

 

 # Cargar los modelos y encoders desde Dropbox
 model1 = load_model_from_dropbox(dropbox_model1)
 model2 = load_model_from_dropbox(dropbox_model2)
 model3 = load_model_from_dropbox(dropbox_model3)
 model4 = load_model_from_dropbox(dropbox_model4)
 
 #Definir entity_tax_models como un diccionario
 entity_tax_models = {
    '1120': model2, # Puedes reemplazar 'Modelo 1120S' con el modelo real o función
    '1065': model3,
    '1040': model4,
    '1120S': model1
     # Agrega más casos según sea necesario
 }
 
 

 #Seleccionar las filas que contienen los datos (a partir de la fila 1)
 df_completo = df_completo.iloc[1:].reset_index(drop=True)
 #Renombrar las columnas a los datos que tenemos

 if len(df_completo.columns) == 5:
    df_completo.columns =['Unnamed','Account Description','Debit','Unnamed_3','Credit']

    #Eliminar columnas innecesarias
    df_completo = df_completo[['Account Description','Debit','Credit']]

    #Hacer split a las columnas del archivo
    df_completo['Account Number'] = df_completo['Account Description'].str.split('·').str[0]
    df_completo['Account Description'] = df_completo['Account Description'].str.split('·').str[1]

 elif len(df_completo.columns) == 3:
    df_completo.columns = ['Account Description', 'Debit', 'Credit']

    #Hacer split a las columnas del archivo
    df_completo['Account Number'] = df_completo['Account Description'].str.split().str[0]
    df_completo['Account Description'] = df_completo['Account Description'].str.split(n=1).str[1]

 else:
    raise ValueError("El archivo subido tiene un formato inesperado, por favor revisa las columnas")

 

 #Define las columnas que deseas extraer
 columnas_deseadas = ['Account Number','Account Description','Debit','Credit']

 #Verificamos que las columnas existan en el archivo
 for columna in columnas_deseadas:
    if columna not in df_completo.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en el archivo de Excel.")

 #Seleccionamos las columnas deseadas
 df = df_completo[columnas_deseadas].copy()

 # Opcional: Manejo de valores faltantes
 df['Debit'] = df['Debit'].fillna(0)
 df['Credit'] = df['Credit'].fillna(0)

 #Asegurar que 'Debit' y 'Credit' sean de tipo numérico
 df['Debit'] = pd.to_numeric(df['Debit'],errors='coerce').fillna(0)
 df['Credit'] = pd.to_numeric(df['Credit'],errors='coerce').fillna(0)

 df['Entity Tax'] = f"{add_selectbox}"


 df = df.dropna(subset=['Account Number', 'Account Description'], how='all')
 new_data = df

 # Convertir las columnas de los nuevos datos a los tipos correctos
 new_data['Account Description'] = new_data['Account Description'].str.strip()
 new_data['Account Number'] = new_data['Account Number'].str.strip().str.lower()
 new_data['XAccount'] =  new_data['Account Number'] +" "+ new_data['Account Description'].str.lower()
 #Clasificar categorías
 new_data['Category'] = new_data['Account Number'].apply(classify_ctg)   
 print(new_data)


 



 # Predicción
 predictions = []
 prediction_proba = []
 

 for _, row in new_data.iterrows():
    entity_tax = row['Entity Tax']

    # Seleccionar el modelo correcto basado en Entity Tax
    if entity_tax in entity_tax_models:
        model = entity_tax_models[entity_tax]

        # Seleccionar las características necesarias
        x_account = row['XAccount']
        if pd.isna(x_account):
         x_account = ""  # Usa una cadena vacía o un valor predeterminado
        else:
         x_account = str(x_account)

        # Realizar la predicción
        prediction = model.predict([x_account])[0]

        #Confianza
        proba = model.predict_proba([x_account])[0].max()

        # Almacenar la predicción
        predictions.append(prediction)
        prediction_proba.append(proba)
    else:
        predictions.append(None)  # Si no hay modelo para el Entity Tax, devolver None
        prediction_proba.append(None)  # Si no hay modelo para el Entity Tax, devolver None

 # Añadir las predicciones al DataFrame
 new_data['Predicted Tax Code'] = predictions
 new_data['Probability'] = prediction_proba

 #Filtrar las filas que tienen categoría distinta a Unknown
 new_data_filtered = new_data[new_data['Category'] != 'Unknown']
 
 # Seleccionar las columnas que se mostrarán en la tabla
 new_data_selected = new_data_filtered[['Predicted Tax Code','Account Number', 'Account Description', 'Debit', 'Credit','Probability', 'Category']]


 st.write(f"## Predicción de Tax Code {add_selectbox}")
 st.dataframe(new_data_selected.style.applymap(probablity_color, subset=['Probability']))
 st.success(" Se ha generado la predicción", icon="✅")
 st.text("Para obtener soporte adicional, comunícate con el desarrollador de la aplicación.")
else:
 

 st.info('Siga los pasos para generar la predicción',  icon="ℹ️")

st.logo(logoGllv)




