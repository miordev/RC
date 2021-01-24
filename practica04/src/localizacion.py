#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional 
# Grado en Ingeniería Informática (Cuarto)
# Práctica 5:
#     Simulación de robots móviles holonómicos y no holonómicos.

import sys
import math
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ******************************************************************************
# Declaración de funciones

# Distancia entre dos puntos (admite poses)
def distancia(a, b):
  return np.linalg.norm(np.subtract(a[:2], b[:2]))


# Diferencia angular entre una pose y un punto objetivo 'p'
def angulo_rel(pose, p):
  w = math.atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
  
  while (w > math.pi):
    w -= 2 * math.pi
  while (w < - math.pi):
    w += 2 * math.pi
  return w


def mostrar(objetivos, ideal, trayectoria):
  # Mostrar objetivos y trayectoria:
  plt.ion() # modo interactivo
  # Fijar los bordes del gráfico
  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()
  bordes = [min(trayT[0]+objT[0]+ideT[0]),max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),max(trayT[1]+objT[1]+ideT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')
  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*math.cos(p[2])],[p[1],p[1]+r*math.sin(p[2])],'-r')
    #plt.plot(p[0],p[1],'or')
  objT   = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  plt.show()
  input()
  plt.clf()


# Buscar la localización más probable del robot, a partir de su sistema
# sensorial, dentro de una región cuadrada de centro "centro" y lado "2 * radio".
def localizacion(landmarks, real, ideal, center, radio, show = False):
      
  LOCATION_FACTOR = 0.05

  list_size = int(round((2 * radio) / LOCATION_FACTOR))
  imagen = [[0 for x in range(list_size)] for y in range(list_size)]

  real_measurements = real.sense(landmarks)

  min_weight = float('inf')
  best_x = 0
  best_y = 0

  y = center[1] - radio
  for i in range(list_size):
    
    x = center[0] - radio
    for j in range(list_size):

      ideal.set(x, y, ideal.orientation)
      weight = ideal.measurement_prob(real_measurements, landmarks)
      imagen[i][j] = weight

      if (weight < min_weight):
        min_weight = weight
        best_x = x
        best_y = y

      x += LOCATION_FACTOR
    y += LOCATION_FACTOR

  ideal.set(best_x, best_y, ideal.orientation)

  if show:
    # Modo interactivo
    plt.ion()

    left = center[0] - radio
    right = center[0] + radio
    down = center[1] - radio
    up = center[1] + radio

    plt.xlim(left, right)
    plt.ylim(down, up)
    imagen.reverse()
    plt.imshow(imagen, extent = [left, right, down, up])

    landmarks_list = np.array(landmarks).T.tolist()    
    plt.plot(landmarks_list[0], landmarks_list[1], 'or', ms = 10)
    plt.plot(ideal.x, ideal.y, 'D', c = '#ff00ff', ms = 10, mew = 2)
    plt.plot(real.x, real.y, 'D', c = '#00ff00', ms = 10, mew = 2)

    plt.show()
    input()
    plt.clf()

# ******************************************************************************

# ------- Definición del robot --------
P_INICIAL = [0.0, 4.0, 0.0]     # Pose inicial (posición y orientación)
V_LINEAL  = 0.7                 # Velocidad lineal    (m/s)
V_ANGULAR = 140.0               # Velocidad angular   (m/s)
FPS       = 10.0                # Resolución temporal (fps)

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = 0.2

# ------ Definición de trayectorias ------
trayectorias = [
  [[1,3]],
  [[0,2], [4,2]],
  [[2,4], [4,0], [0,0]],
  [[2,4], [2,0], [0,2], [4,2]],
  [[2 + 2 * math.sin(0.8 * math.pi * i), 2 + 2 * math.cos(0.8 * math.pi * i)] for i in range(5)]
]

# ------- Definición de los puntos objetivo -------
if len(sys.argv) < 2 or int(sys.argv[1]) < 0 or int(sys.argv[1]) >= len(trayectorias):
  sys.exit(sys.argv[0]+" <Índice entre 0 y " + str(len(trayectorias)-1) + ">")
objetivos = trayectorias[int(sys.argv[1])]


# ------ Definición de constantes --------
EPSILON = 0.1                           # Umbral de distancia
V = V_LINEAL / FPS                      # Metros por fotograma
W = V_ANGULAR * math.pi / (180 * FPS)   # Radianes por fotograma
MAX_WEIGHT = 0.3


ideal = robot()
ideal.set_noise(0.0, 0.0, 0.1)    # Ruido: Lineal / Radial / Sensado
ideal.set(*P_INICIAL)             # Usa el operador 'splat'

real = robot()
real.set_noise(0.01, 0.01, 0.1)   # Ruido: Lineal / Radial / Sensado
real.set(*P_INICIAL)

tray_ideal  = [ideal.pose()]    # Trayectoria percibida
tray_real   = [real.pose()]     # Trayectoria seguida

tiempo  = 0.0
espacio = 0.0

random_pos = [0, 0]
localizacion(objetivos, real, ideal, random_pos, 5, True)

random.seed(datetime.now())
for punto in objetivos:
  while distancia(tray_ideal[-1], punto) > EPSILON and len(tray_ideal) <= 1000:
    pose = ideal.pose()

    w = angulo_rel(pose, punto)
    if (w > W):
      w =  W
    if (w < - W):
      w = - W

    v = distancia(pose, punto)
    if (v > V):
      v = V
    if (v < 0):
      v = 0

    if (HOLONOMICO):
      if (GIROPARADO and abs(w) > 0.01):
        v = 0

      ideal.move(w, v)
      real.move(w, v)
      
    else:
      ideal.move_triciclo(w, v, LONGITUD)
      real.move_triciclo(w, v, LONGITUD)
    
    tray_ideal.append(ideal.pose())
    tray_real.append(real.pose())
    
    # Revisar si las medidas son similares entre el robot real e ideal 
    real_measurements = real.sense(objetivos)
    weight = ideal.measurement_prob(real_measurements, objetivos)
    
    # La distancia entre el robot real e ideal ha superado el límite 
    if (weight > MAX_WEIGHT):
      localizacion(objetivos, real, ideal, ideal.pose(), 1, False)

    espacio += v
    tiempo  += 1

if len(tray_ideal) > 1000:
  print ("<!> Trayectoria muy larga - puede que no se haya alcanzado la posición final.")

print ("Recorrido: " + str(round(espacio, 3)) + "m / " + str(tiempo / FPS) + "s")
final_distance = str(round(distancia(tray_real[-1], objetivos[-1]), 3))
print ("Distancia real al objetivo: " + final_distance + "m")

# Representación gráfica
mostrar(objetivos, tray_ideal, tray_real)   

