import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# regla de Sturges para determinar el número de bins en un histograma
def Sturges_rule(n):
    return int(np.ceil(1 + np.log2(n)))
# funcion de espacios para visualización
def espacios(n):
    print("\n"*n)

def mean_non_zero(series):
    return series[series > 0].mean()
# Función para convertir un mes en texto a un número
def mes_a_numero(mes):
    meses = {
        'Enero': 1,
        'Febrero': 2,
        'Marzo': 3,
        'Abril': 4,
        'Mayo': 5,
        'Junio': 6,
        'Julio': 7,
        'Agosto': 8,
        'Septiembre': 9,
        'Octubre': 10,
        'Noviembre': 11,
        'Diciembre': 12,
    }
    return meses[mes]

# Conectar a la base de datos SQLite
conn = sqlite3.connect('sensores.db')

# Leer los datos de la tabla en un DataFrame de pandas
dosificaciones = pd.read_sql_query("SELECT * FROM datos_sensores", conn)
calidad=pd.read_sql_query("SELECT * FROM datos_calidad", conn)
calidad["mes"] = calidad["mes"].apply(mes_a_numero)
calidad.rename(columns={"Conductividad": "Conductividad_tratada"}, inplace=True)

# Cerrar la conexión

# los valores que eran originalmente 0, se convierten en NA para que no lo considere en los cálculos
dosificaciones.replace(0, pd.NA, inplace=True)

# Convertir las columnas 'hora', 'dia', 'mes' y 'año' a un objeto datetime y crear una nueva columna 'datetime'
calidad['FechaHora'] = pd.to_datetime(calidad[['año', 'mes', 'dia', 'hora']].rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day', 'hora': 'hour'}))
calidad.set_index('FechaHora', inplace=True)

dosificaciones['FechaHora'] = pd.to_datetime(dosificaciones[['año', 'mes', 'dia', 'hora']].rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day', 'hora': 'hour'}))
dosificaciones.set_index('FechaHora', inplace=True)

#Extraemos los datos de calidad a partir del 2018
calidad_filtrada = calidad[calidad.index > '2018-01-01']
# Multiplicar la columna de sulfato (columna 5) por 1.33, esto para pasar de ml/L a mg/L
dosificaciones.iloc[:, 5] = dosificaciones.iloc[:, 5] * 1.33

# Creamos un nuevo Dataset sin las columnas de año, mes, día y hora

df_selected = dosificaciones.iloc[:, 4:]

#Las siguientes lineas de codigo tienen la intención de convertir todas las columnas a valores numericos
columnnas = df_selected.columns
for columna in columnnas:
    df_selected[columna]=pd.to_numeric(df_selected[columna], errors='coerce')
del columnnas
columnas = calidad_filtrada.columns
for columna in columnas:
    calidad_filtrada[columna]=pd.to_numeric(calidad_filtrada[columna], errors='coerce')
del columnas


# print(df_selected.info())
# espacios(2)



calidad=calidad.iloc[:, 4:]
#  Unir los DataFrames con una unión externa
Datos_unificados = pd.merge(calidad, df_selected, how='outer', left_index=True, right_index=True)
Datos_unificados.sort_index(inplace=True)


# Eliminar las filas con valores NA en la columna 'Caudal'
Datos_unificados=Datos_unificados[Datos_unificados["Caudal"].notna()]

columnas_a_eliminar = ['Cloruros_cruda',"Turbiedad_filtrada"]
for columna in columnas_a_eliminar:
    Datos_unificados.drop(columna, axis=1, inplace=True)
Datos_unificados.to_sql('datos_unificados', conn, if_exists='replace')
conn.close()