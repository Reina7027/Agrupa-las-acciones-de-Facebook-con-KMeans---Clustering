'''
    Description:
    Group Facebook actions with KMeans
    Author: YolaM
    Version: 1.0
'''

###1.-Cargar librerias
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

###2.-Cargar la base de datos
path='C:/Users/usuario/Desktop/TAREAS-5°Ciclo-n°complej/Python/EVA2PY/' # your path
df=pd.read_csv(path+'FB.csv')

###3.-Preprocesado de los datos
#Eliminamos los filas con valores NaN
df = df.dropna()
df = df.reset_index(drop=True)
#reseteamos el indice que enumera las filas para que se reajuste 
#cuando eliminamos las filas afin de evitar problemas de dimensionalidad al 
#extraer la columna que luego vamos a agregar

#Eliminamos la columna de fecha que no utilizaremos en el algoritmo 
dates = df['Date'] # Guardamos la columna Date.
df = df.drop('Date', 1) # Borramos la columna del dataframe.

# Analizando outliers
#####################

# Grafico de cajas para ver outliers/atipicos
pd.DataFrame(df['Close']).boxplot()
pd.DataFrame(df['Volume']).boxplot()
# Hitograma 
plt.hist(df['Volume'],bins=50,alpha=0.5,color='green',
         orientation='vertical',histtype='step')

# Anlaizando percentiles
np.percentile(df['Volume'],range(101))
cota_vol=np.percentile(df['Volume'],[96,97,98,99,100])

# Asignacion cotas superiores
df.loc[(df['Volume'])>=cota_vol[0],['Volume']]=cota_vol[0]
df.loc[(df['Volume'])>=cota_vol[0],['Volume']]=cota_vol[0]

# Validacion de cambios
np.percentile(df['Volume'],range(101))
plt.hist(df['Volume'],bins=50,alpha=0.5,color='green',
         orientation='vertical',histtype='step')


###3.- Normalizacion de los datos
min_max_scaler = preprocessing.MinMaxScaler() 
X_std = min_max_scaler.fit_transform(df)

X_std = pd.DataFrame(X_std) # Hay que convertir a DF el resultado.
X_std = X_std.rename(columns = {0: 'Close', 1: 'Volume'})


###4.-Representacion grafica de los datos
x = X_std['Close'].values
y = X_std['Volume'].values
plt.xlabel('Close price')
plt.ylabel('Volume')
plt.title('Facebook stocks CLOSE vs. VOLUME')
plt.plot(x,y,'o',markersize=1)


###5.- Aplicacion  de KMeans
nc = range(1, 15) # El número de iteraciones que queremos hacer.
kmeans = [KMeans(n_clusters=i) for i in nc]
score = [kmeans[i].fit(X_std).score(X_std) for i in range(len(kmeans))]
score
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Suma de los errores cuadráticos')
plt.plot(nc,score)


kmeans = KMeans(n_clusters=4).fit(X_std)
centroids = kmeans.cluster_centers_
print(centroids)


labels = kmeans.predict(X_std)
df['label'] = labels
df.insert(0, 'Date', dates)

#Representación grafica de los clusteres de KMeans
colores=['red','blue','yellow','fuchsia']
asignar=[]
for row in labels:
     asignar.append(colores[row])
plt.scatter(x, y, c=asignar, s=1)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=70) # Marco centroides.
plt.xlabel('Close price')
plt.ylabel('Volume')
plt.title('Facebook stocks k-means clustering')
plt.show()


#DETERMINAR A QUE CLUSTER PERTENECEN NUEVOS DATOS DE ENTRADA
close= 187.89
volume= 8047400


#Introducimos estos nuevos datos como un dataframe de una única fila:
nuevo_dato = pd.DataFrame([[close,volume]]) # Nueva muestra
nuevo_dato = nuevo_dato.rename(columns = {0: 'Close', 1: 'Volume'})


#añadimos a nuestro dataframe de inicio 
#y lo guardamos con el nombre df_n para no sobrescribir el original:
df_n = df.append(nuevo_dato)


#eliminamos la columna Date y Label
df_n = df_n.drop('Date', 1)
df_n = df_n.drop('label', 1)
df_n = df_n.reset_index(drop=True)


#Normalizamos el DataFrame y obtenemos los nuevos datos normalizados
min_max_scaler = preprocessing.MinMaxScaler() 
X_std = min_max_scaler.fit_transform(df_n)
X_std = pd.DataFrame(X_std) # Hay que convertir a DF el resultado.
X_std = X_std.rename(columns = {0: 'Close', 1: 'Volume'})


#X_new es el array con los nuevos datos normalizados.
close_n = X_std['Close'][1862]
volume_n = X_std['Volume'][1862]
import numpy as np
X_new = np.array([[close_n, volume_n]]) # Nueva muestra


#introducimos el array X_new en k-means:
new_labels = kmeans.predict(X_new)
print(new_labels)


#Representacion grafica de la nueva muestra
colores=['red','blue','yellow','fuchsia']
asignar=[]
for row in labels:
     asignar.append(colores[row])

fig, ax = plt.subplots()
x_n = close_n
y_n = volume_n
 
plt.plot(x_n,y_n, '*', color = 'lime', markersize = 20)
plt.scatter(x, y, c=asignar, s=1)
plt.xlabel('Close price')
plt.ylabel('Volume')
plt.title('Facebook stocks k-means clustering')
plt.show()







