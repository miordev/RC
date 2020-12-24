#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - Curso 2020/2021
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj):
  # Muestra el robot graficamente
  plt.figure(1)
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.show()
  input()
  plt.clf()


# Calcula la matriz T
# Los ángulos de entrada deben estar en grados
def matriz_T(d, th, a, al):
  return  [ 
            [math.cos(th), - math.sin(th) * math.cos(al),   math.sin(th) * math.sin(al), a * math.cos(th)],
            [math.sin(th),   math.cos(th) * math.cos(al), - math.sin(al) * math.cos(th), a * math.sin(th)],
            [           0,                  math.sin(al),                  math.cos(al),                d],
            [           0,                             0,                             0,                1]
          ]


#Sea 'th' el vector de thetas
#Sea 'a'  el vector de longitudes
def cin_dir(th, a):
  T = np.identity(4)
  o = [[0, 0]]
  for i in range(len(th)):
    T = np.dot(T, matriz_T(0, th[i], a[i], 0))
    tmp = np.dot(T, [0,0,0,1])
    o.append([tmp[0], tmp[1]])
  return o


def obtener_angulo(punto_a, punto_b, punto_c):
  vector_ac = np.subtract(punto_a, punto_c)
  vector_bc = np.subtract(punto_b, punto_c)

  alfa_1 = math.atan2(vector_ac[1], vector_ac[0])
  alfa_2 = math.atan2(vector_bc[1], vector_bc[0])

  beta = alfa_1 - alfa_2
  return beta


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# Valores arbitrarios de las articulares para la cinemática directa inicial
# Los gardos están en radianes
th = [0.0, 0.0, 0.0]
a  = [5.0, 5.0, 5.0]

# Variable para representación gráfica
L  = sum(a)
EPSILON = 0.01

plt.ion() # modo interactivo

# Introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo = [float(i) for i in sys.argv[1:]]

# Reserva una estructura en memoria, para las posiciones de las articulaciones
O = list(range(len(th) + 1)) 

# Calcula la posicion inicial
O[0] = cin_dir(th, a) 
print("- Posicion inicial:")
muestra_origenes(O[0])

dist = float("inf")
prev = 0.0
iteracion = 1

while (dist > EPSILON and abs(prev - dist) > EPSILON/100.0):
  prev = dist  

  final_index = len(th)

  # Para cada combinación de articulaciones
  for i in range(len(th)):
    indice_articulacion_actual = len(th) - 1 - i

    punto_final = O[i][-1]
    efector = O[i][indice_articulacion_actual]

    variacion_angulo = obtener_angulo(objetivo, punto_final, efector)
    th[indice_articulacion_actual] += variacion_angulo
    
    O[i+1] = cin_dir(th, a)


  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo)
  print("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if dist <= EPSILON:
  print("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist,5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(th)):
  print("  L" + str(i+1) + "     = " + str(round(a[i],3)))
