# Importamos la librerías 

from fastapi import FastAPI
import pandas as pd
import json
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
# ----------------

app = FastAPI()
# ----------------
df_f1_2= pd.read_csv(r'C:/Users/Mary/Desktop/Pimlops/df_f1_2.csv')
df_f3_4_5= pd.read_csv(r'C:/Users/Mary/Desktop/Pimlops/df_f3_4_5.csv')
# ----------------
df_f3_4_5.drop(columns='Unnamed: 0',inplace=True)
df_f3_4_5.drop(columns='title',inplace=True)
df_f3_4_5.drop(columns='release_date',inplace=True)

df_f3_4_5['genres'] = df_f3_4_5['genres'].apply(lambda x: x.replace("'", "").strip("[]"))

df_f3_4_5.dropna(inplace=True)
lista=[]
for i in range(0,len(df_f3_4_5)):
    string = df_f3_4_5.iloc[i][5]

    try:
      b = int(string[-5:-1])
    except ValueError:
      b = float('nan') 

    lista.append(b)

df_f3_4_5['posted_year'] = lista
df_f3_4_5.dropna(inplace=True)
df_f3_4_5['posted_year'] = df_f3_4_5['posted_year'].astype('int')

df_f3_4_5.drop(columns='posted', inplace=True)
# --------------------
# Crear una instancia del codificador
label_encoder = LabelEncoder()

# Cargar los datos
df_ml = pd.read_csv(r'C:/Users/Mary/Desktop/Pimlops/df_ML.csv')
# Crear una nueva columna llamada genres_encoded, que tiene los generos codificados como int.
df_ml["genres_encoded"] = label_encoder.fit_transform(df_ml["genres"])

# Crear un diccionario de los títulos asociados a cada item_id
titles_by_item_id = {}
for i in range(len(df_ml)):
    titles_by_item_id[df_ml.loc[i, "item_id"]] = df_ml.loc[i, "app_name"]

# Crear el modelo K-Nearest Neighbors
k = 5
model = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
model.fit(df_ml[['genres_encoded']], df_ml['app_name'])

# Guardar el modelo
with open('modelo.pkl', 'wb') as f:
    pickle.dump(model, f)

# Guardar el diccionario
with open('titles_by_item_id.pkl', 'wb') as f:
    pickle.dump(titles_by_item_id, f)
#---------------------------------------

#http://127.0.0.1:8000/ 

@app.get("/")
def prueba():
  return 'Hola, soy Nicolás Pontis Ledda, formo parte de la cohorte DATAFT18 de Henry y este es el apartado de presentación de mi PI01 - MLOPs'
#-------------------------------------------
@app.get('/playtime_genre')
def PlayTimeGenre(genero):
    genero = genero.lower()
    genero = genero.capitalize()
    df_f1_2['release_date'] = pd.to_datetime(df_f1_2['release_date'], errors='coerce')
    max_horas_anio = None
    max_horas = 0
    horas_por_anio = {}
    
    for index, row in df_f1_2.iterrows():
        if genero in row['genres']:
            # Obtener el año de la fecha de lanzamiento
            year = row['release_date'].year
            
            # Sumar las horas jugadas
            horas_jugadas = row['playtime_forever']
            
            if year not in horas_por_anio:
                horas_por_anio[year] = 0
                
            horas_por_anio[year] += horas_jugadas
            
            if horas_por_anio[year] > max_horas:
                max_horas = horas_por_anio[year]
                max_horas_anio = year
    res = {
    "Año con más horas": max_horas_anio,"Total de horas sumadas": max_horas
    }
            
    return res
#--------------------------------------------
@app.get('/user_for_genre')
def UserForGenre(genero):

    df_f1_2["release_date"] = pd.to_datetime(df_f1_2["release_date"])
    df_f1_2["release_year"] = df_f1_2["release_date"].dt.year

    # Filtrar el dataframe por género.
    df_filtrado = df_f1_2[df_f1_2["genres"].str.contains(genero)]

    # Calcular la acumulación de horas jugadas por usuario.
    df_acumulado = df_filtrado.groupby("user_id")["playtime_forever"].sum()

    # Obtener el usuario con más horas jugadas.
    usuario_mas_horas = df_acumulado.idxmax()

    # Calcular la acumulación de horas jugadas por año.
    df_acumulado_por_ano_1 = df_filtrado.groupby(["release_year"])["playtime_forever"].sum().to_frame()
    df_1 = df_acumulado_por_ano_1.add_suffix("_Sum").reset_index()

    # Convertir el dataframe a una lista de diccionarios.
    df_1 = df_1.rename(columns={"release_year": "Año", "playtime_forever_Sum": "Horas"})
    lista_acumulado_por_ano = df_1.to_dict(orient="records")

    # Devolver el resultado.
    return {
        "Usuario con más horas jugadas para Género X": usuario_mas_horas,
        "Horas jugadas": lista_acumulado_por_ano
    }
#-----------------------------------------------
@app.get('/users_recommend')
def UsersRecommend(year: int):
    # Verificar si el año es igual a -1 y mostrar un mensaje personalizado
    if year == -1:
        return "El año ingresado es -1, lo cual no es válido."

    # Verificar que el año sea un número entero
    if not isinstance(year, int):
        return "El año debe ser un número entero."

    # Verificar que el año ingresado esté en la columna 'posted_year'
    if year not in df_f3_4_5['posted_year'].unique():
        return "El año no se encuentra en la columna 'posted_year'."

    # Filtrar el dataset para obtener solo las filas correspondientes al año dado
    juegos_del_año = df_f3_4_5[df_f3_4_5['posted_year'] == year]

    # Calcular la cantidad de recomendaciones para cada juego
    recomendaciones_por_juego = juegos_del_año.groupby('app_name')['recommend'].sum().reset_index()

    # Ordenar los juegos por la cantidad de recomendaciones en orden descendente
    juegos_ordenados = recomendaciones_por_juego.sort_values(by='recommend', ascending=False)

    # Tomar los tres primeros lugares
    primer_puesto = juegos_ordenados.iloc[0]['app_name']
    segundo_puesto = juegos_ordenados.iloc[1]['app_name']
    tercer_puesto = juegos_ordenados.iloc[2]['app_name']

    # Crear el diccionario con los tres primeros lugares
    top_tres = {
        "Puesto 1": primer_puesto,
        "Puesto 2": segundo_puesto,
        "Puesto 3": tercer_puesto
    }

    return top_tres
#--------------------------------------------
@app.get('/users_worst_developer')
def UsersWorstDeveloper(año: int):
    # Verificar si el año es igual a -1 y mostrar un mensaje personalizado
    if año == -1:
        return "El año ingresado es -1, lo cual no es válido."

    # Verificar que el año sea un número entero
    if not isinstance(año, int):
        return "El año debe ser un número entero."

    # Verificar que el año ingresado esté en la columna 'posted_year'
    if año not in df_f3_4_5['posted_year'].unique():
        return "El año no se encuentra en la columna 'posted_year'."

    # Filtrar el dataset para obtener solo las filas correspondientes al año dado
    juegos_del_año = df_f3_4_5[df_f3_4_5['posted_year'] == año]

    # Calcular la cantidad de recomendaciones para cada developer
    recomendaciones_por_juego = juegos_del_año.groupby('developer')['recommend'].sum().reset_index()

    # Ordenar los juegos por la cantidad de recomendaciones en orden descendente
    devs_ordenados = recomendaciones_por_juego.sort_values(by='recommend', ascending=True)

    # Tomar los tres primeros lugares
    ultimo_puesto = devs_ordenados.iloc[0]['developer']
    penultimo_puesto = devs_ordenados.iloc[1]['developer']
    antepenultimo_puesto = devs_ordenados.iloc[2]['developer']

    # Crear el diccionario con los tres primeros lugares
    ultimos_tres = {
        "Puesto 1": ultimo_puesto,
        "Puesto 2": penultimo_puesto,
        "Puesto 3": antepenultimo_puesto
    }

    return ultimos_tres
#-----------------------------------------
@app.get('/sentiment_analysis')
def sentiment_analysis(devs:str):

    # Filtrar el DataFrame por la desarrolladora ingresada
    df_devs = df_f3_4_5[df_f3_4_5['developer'] == devs]

    # Verificar si se encontraron registros para el año
    if df_devs.empty:
        return {"Mensaje": "No se encontraron registros para la desarrolladora especificada."}

    # Contar la cantidad de registros para cada categoría de análisis de sentimiento
    sentiment_analysis_column = df_devs['sentiment_analysis']
    sentiment = sentiment_analysis_column.value_counts().to_dict()

    # Crear una lista con los resultados, tuve que colocar en formato de str todo para que me lo tomara como válido 
    resultado= list(["Negative= "+ str(sentiment.get(0, 0)), "Neutral= "+ str(sentiment.get(1, 0)), "Positive= "+ str(sentiment.get(2, 0))])

    #Crear el diccionario final
    result = {
        devs:resultado
    }

    return result
#-----------------------------------------
#Función 6, recomendación de juegos segun item_id indicado
@app.get('/get_recomendations')
def get_recommendations(item_id: int):

    # Cargar el modelo 
    with open("modelo.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar el diccionario de títulos
    with open("titles_by_item_id.pkl", "rb") as f:
        titles_by_item_id = pickle.load(f)

    # Buscar el género codificado del juego proporcionado por el usuario
    input_game = df_ml[df_ml["item_id"] == item_id]["genres_encoded"].values[0]

    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])

    # Obtener los títulos de los juegos similares
    similar_games = [titles_by_item_id[df_ml.loc[i, "item_id"]] for i in indices[0]]

    # Devolver un diccionario de los títulos
    return {"similar_games": similar_games}