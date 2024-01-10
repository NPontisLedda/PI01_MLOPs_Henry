
# Proyecto individual 1 - Henry
# Machine Learning Operations (MLOps)

![MLOps](https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png)

# Introducción

El proyecto actual se basa en datos de ficción y no es un proyecto que busca resultados de un análisis, sino mostrar algunas habilidades que he adquirido a lo largo del bootcamp.

Se me confió la tarea de desarrollar una API utilizando el marco **FastAPI** para mostrar un sistema de recomendación y análisis de bases de datos de juegos. El resultado solicitado fue un **Producto Mínimo Viable (MVP)** que contiene cinco puntos finales de función y un último para un sistema de recomendación de aprendizaje automático.

![PI1_MLOps_Mapa1](https://raw.githubusercontent.com/pjr95/PI_ML_OPS/main/src/DiagramaConceptualDelFlujoDeProcesos.png)


# Descripción y diccionario del conjunto de datos
Para descargar los archivos originales, ya que tienen mucho peso, se pueden encontrar en el siguiente enlace. [Datasets originales](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)



_**ETL**_:
Para conocer más sobre el desarrollo del proceso ETL, existe el siguiente enlace
[Notebook ETL](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/PI_MLOPs_ETL_EDA.ipynb)

_**Nombres de los datasets**_:
- australian_user_reviews
- australian_users_items
- output_steam_games

_**Desanidado**_:
1. Algunas columnas están anidadas, es decir, tienen un diccionario o una lista como valores en cada fila, las desanidamos para poder realizar algunas de las consultas API.

_**Eliminar columnas no utilizadas**_:

2. Se eliminan las columnan que no se utilizarán:
   - De output_steam_games: publisher, url, tags, price, specs, early_access.
   - De australian_user_reviews: user_url, funny, last_edited, helpful.
   - De australian_users_items: user_url, playtime_2weeks, steam_id, items_count.

_**Control de valores nulos**_:

3. Se eliminan valores nulos:
   - De output_steam_games: genres, release_date.
   - De australian_user_reviews: item_id.
   - De australian_user_items: user_id

_**Cambio del tipo de datos**_:

4. Las fechas se cambian a datetime para luego extraer el año:
   - De australian_user_reviews: la columna posted .
   - De output_steam_games: la columna release_date.

_**Se quitan datos sin valor**_:

5. Los datos que no tienen ningún valor:
   - De australian_user_items: la columna playtime_forever.

_**Fusión de conjuntos de datos**_:

6. Combiné los datasets para las funciones 1 y 2 en un archivo .csv [Archivo para funciones 1 y 2](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/df_f1_2.csv), y para las funciones 3, 4 y 5 en otro archivo .csv [Archivo para funciones 3,4 y 5](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/df_f3_4_5.csv).

_**Análisis de sentimiento**_:

7. En el conjunto de datos australian_user_reviews, hay reseñas de juegos realizadas por diferentes usuarios. Creación de la columna 'sentiment_analysis' aplicando análisis de sentimiento de PNL con la siguiente escala: toma el valor '0' si es negativo, '1' si es neutral y '2' si es positivo. Esta nueva columna reemplaza la columna australian_user_reviews.review para facilitar el trabajo de los modelos de aprendizaje automático y el análisis de datos. Si este análisis no es posible por falta de una reseña escrita, toma el valor de 1.


# _Funciones_
- _**Para obtener más información sobre el desarrollo de las diferentes funciones y una explicación más detallada de cada una, haga clic en el siguiente enlace**_
[Notebook de funciones](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/FastAPI/fastapi-env/main.py)

Desarrollo API: Se propone disponibilizar los datos de la empresa usando el framework FastAPI . Las consultas que proponemos son las siguientes:

Cada aplicación tiene un decorador (@app.get('/')) con el nombre de la aplicación para poder reconocer su función.

Las consultas son las siguientes:

1. **PlayTimeGenre(género:str)**:
Debe devolver año con más horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

2. **UserForGenre(género:str)**:
Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas : 23}]}

3. **UsersRecommend(año:int)**:
Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

4. **def UsersWorstDeveloper(año:int)**:
Devuelve el top 3 de desarrolladores con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

5. **sentiment_analysis(empresa desarrolladora:str)**:
Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.
Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

# _**EDA (Análisis exploratorio de datos)**_
Los conjuntos de datos tenían algunos aspectos que corregir relacionados con variables numéricas. La columna playtime_forever tenía algunos valores atípicos con cantidades irreales de horas jugadas para algunos usuarios; las cantidades se corrigieron.

# _**Aprendizaje automático(Machine Learning)**_

El modelo establece una relación artículo-artículo. Esto significa que dado un item_id, en función de qué tan similar sea al resto, se recomendarán artículos similares. Aquí, la entrada es un juego y la salida es una lista de juegos recomendados.

El método de aprendizaje automático utilizado es K-Neighbours. No es el mejor método para abordar los conjuntos de datos y parte de este proyecto se centra en eso. Debido a que el proyecto debe implementarse en Render, la memoria RAM disponible es limitada y lo importante aquí era comprender la diferencia entre los diferentes modelos de Machine Learning. Anteriormente, probé árboles de decisión y procesamiento de lenguaje natural utilizando similitud de coseno.

El sistema de recomendación item-item se planteó originalmente así:

6. **get_recommendations(item_id)**: 
Ingresando el id de producto(item_id), deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.



# _**Implementación de API**_
La implementación de nuestra FastAPI se realiza utilizando Render un entorno virtual.

Haga clic para acceder a mi aplicación FastAPI: [API Deployment](https://proyecto1-gkkk.onrender.com/docs#/)

Para consumir la API, utilice los 6 endpoints diferentes para obtener información y realizar consultas sobre estadísticas de juegos.



# Requisitos
- Python
- Scikit-Learn
- Pandas
- NumPy
- FastAPI
- nltk
- [Render](https://render.com/)

# _Autor_
- Nicolás Pontis Ledda
- Mail: nicopontis54@gmail.com
- Linkedin: [Linkedin](https://www.linkedin.com/in/nicol%C3%A1s-pontis-ledda-8a8083197/)
