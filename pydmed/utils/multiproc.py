import numpy as np
import matplotlib.pyplot as plt
import os, sys
import psutil
from pathlib import Path
import re
import time
import random
import multiprocessing as mp
from abc import ABC, abstractmethod
import openslide
import torch
import torchvision
import torchvision.models as models
from multiprocessing import Process, Queue


def poplast_from_queue(queue):
    '''
    Pops the last element of a `multiprocessing.Queue`.
    '''
    size_queue = queue.qsize()
    if(size_queue == 0):
        return None
    elem = None
    for count in range(size_queue):
        try:
            elem = queue.get_nowait()
        except:
            pass
    return elem


def set_nicemax():
    '''
    Sets the priority of the process to the highest value. 
    '''
    maxcount = 1000
    N_old = os.nice(0)
    count = 0
    while(True):
        count += 1
        N_new = os.nice(N_old+1000)
        if(N_new == N_old):
            return
        if(count > maxcount):
            return



def terminaterecursively(pid):
    print("=================================================================================")
    parent = psutil.Process(pid)#TODO:copyright, https://www.reddit.com/r/learnpython/comments/7vwyez/how_to_kill_child_processes_when_using/
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except:
            pass
            #print(" killed subprocess {}".format(child))
        #if including_parent:
    try:
        parent.kill()
    except:
        pass
