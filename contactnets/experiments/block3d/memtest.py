from contactnets.experiments.block3d import train
import pandas as pd
from operator import itemgetter
from pympler import tracker

import objgraph

import random

import ctypes

import os
import psutil

import gc

import pdb

process = psutil.Process(os.getpid())
print(process.memory_info().rss)

objgraph.show_growth()
objgraph.get_new_ids()
print('-------------------')

train.do_train(epochs=0, batch=1, patience=0, resume=False)
gc.collect()

new_ids = objgraph.get_new_ids()
objgraph.show_growth()

process = psutil.Process(os.getpid())
print(process.memory_info().rss)

pdb.set_trace()

objgraph.show_chain(objgraph.find_backref_chain(random.choice(list(new_ids['dict'])), objgraph.is_proper_module), filename='chain.png')
objgraph.show_chain(objgraph.find_backref_chain(ctypes.cast(list(new_ids['dict'])[5], ctypes.py_object).value, objgraph.is_proper_module), filename='chain.png')
