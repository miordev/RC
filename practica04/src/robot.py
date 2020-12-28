#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional 
# Grado en Ingeniería Informática (Cuarto)
# Clase robot

import math
import random
import numpy as np
import copy


class robot:

  # Inicializacion de pose y parámetros de ruido
  def __init__(self):
    self.x             = 0.0
    self.y             = 0.0
    self.orientation   = 0.0
    self.forward_noise = 0.0
    self.turn_noise    = 0.0
    self.sense_noise   = 0.0
    self.weight        = 1.0
    self.old_weight    = 1.0
    self.size          = 1.0


  # Constructor de copia
  def copy(self):
    return copy.deepcopy(self)


  # Modificar la pose
  def set(self, new_x, new_y, new_orientation):
    self.x = float(new_x)
    self.y = float(new_y)
    self.orientation = float(new_orientation)

    while (self.orientation >  math.pi):
      self.orientation -= 2 * math.pi
    while (self.orientation < - math.pi):
      self.orientation += 2 * math.pi


  # Modificar los parámetros de ruido
  def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
    self.forward_noise = float(new_f_noise)
    self.turn_noise    = float(new_t_noise)
    self.sense_noise   = float(new_s_noise)


  # Obtener pose actual
  def pose(self):
    return [self.x, self.y, self.orientation]


  # Calcular la distancia a una de las balizas
  def sense1(self, landmark, noise):
    return np.linalg.norm(np.subtract([self.x,self.y],landmark)) \
                                        + random.gauss(0.,noise)

  
  # Calcular las distancias a cada una de las balizas
  def sense(self, landmarks):
    d = [self.sense1(l,self.sense_noise) for l in landmarks]
    d.append(self.orientation + random.gauss(0.,self.sense_noise))
    return d


  # Modificar pose del robot (holonómico)
  def move(self, turn, forward):
    self.orientation += float(turn) + random.gauss(0., self.turn_noise)
    while self.orientation >  math.pi: self.orientation -= 2 * math.pi
    while self.orientation < -math.pi: self.orientation += 2 * math.pi
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.x += math.cos(self.orientation) * dist
    self.y += math.sin(self.orientation) * dist

  
  # Modificar pose del robot (Ackermann)
  def move_triciclo(self, turn, forward, largo):
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.orientation += dist * math.tan(float(turn)) / largo\
              + random.gauss(0.0, self.turn_noise)
    while self.orientation >  math.pi: self.orientation -= 2 * math.pi
    while self.orientation < -math.pi: self.orientation += 2 * math.pi
    self.x += math.cos(self.orientation) * dist
    self.y += math.sin(self.orientation) * dist


  # Calcular la probabilidad de 'x' para una distribución normal
  # de media 'mu' y desviación típica 'sigma'
  def Gaussian(self, mu, sigma, x):
    if sigma:
      return math.exp(-(((mu-x)/sigma)**2)/2)/(sigma * math.sqrt(2 * math.pi))
    else:
      return 0

  
  # Calcular la probabilidad de una medida.
  def measurement_prob(self, measurements, landmarks):
    self.weight = 0.
    for i in range(len(measurements)-1):
      self.weight += abs(self.sense1(landmarks[i],0) -measurements[i])
    diff = self.orientation - measurements[-1]
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < - math.pi: diff += 2 * math.pi
    self.weight = self.weight + abs(diff) 
    return self.weight


  # Representación de la clase robot
  def __repr__(self):
    return '[x=%.6s y=%.6s orient=%.6s]' % \
            (str(self.x), str(self.y), str(self.orientation))

