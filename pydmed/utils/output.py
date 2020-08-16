
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import multiprocessing as mp
import csv
import time



class StreamWriter(mp.Process):
    def __init__(self, list_patients=None, rootpath=None, fname_tosave=None, 
                 waiting_time_before_flush = 3):
        '''
        StreamWriter works in two modes:
            1) one file is created for the whole dataset. In this case, 
                only `fname_tosave` is used and the argument `rootpath` must be None.
            2) one file is created for each `Patient` in the directory `rootpath`.
               In this case, `fname_tosave` must be None.
        Inputs:
            - waiting_time_before_flush: before flushing the contents, it should 
                sleep a few seconds. Default is 3 seconds.
        '''
        super(StreamWriter, self).__init__()
        if(isinstance(rootpath, str) and isinstance(fname_tosave, str)):
            if((rootpath!=None) and (fname_tosave!=None)):
                exception_msg = "One of the arguments `rootpath` and `fname_tosave`"+\
                                " must be set to None. For details, please refer to"+\
                                " `StreamWriter` documentation"
                raise Exception(exception_msg)
        if(isinstance(fname_tosave, str)):
            if(fname_tosave != None):
                self.op_mode = 1
        if(isinstance(rootpath, str)):
            if(rootpath != None):
                self.op_mode = 2
        if(hasattr(self, "op_mode") == False):
            exception_msg = "Exactly one of the arguments `rootpath` or `fname_tosave`"+\
                    " must be set to a string."+\
                    " For details, please refer to"+\
                     " `StreamWriter` documentation"
            raise Exception(exception_msg)
        if(self.op_mode == 1):
            if(fname_tosave.endswith(".csv") == False):
                raise Exception("The argument `fname_tosave` must end with .csv."+\
                                "Because only .csv format is supported.")
        if(self.op_mode == 2):
            if(len(list(os.listdir(rootpath))) > 0):
                print(list(os.listdir(rootpath)))
                raise Exception("The folder {} \n is not empty.".format(rootpath)+\
                        " Delete its files before continuing.")
        #grab privates ================
        self.list_patients = list_patients
        self.rootpath = rootpath
        self.fname_tosave = fname_tosave
        self.waiting_time_before_flush = waiting_time_before_flush
        #make/open csv file(s) =======================
        if(self.op_mode == 1):
            self.list_files = [open(fname_tosave, mode='a+')]
            self.list_writers = [csv.writer(f, delimiter=',',\
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL
                                    ) for f in self.list_files]
        elif(self.op_mode == 2):
            self.list_files = [open(os.path.join(rootpath,\
                                    "patient_{}.csv".format(patient.int_uniqueid))
                                   , mode='a+') for patient in list_patients]
            self.list_writers = [csv.writer(f, delimiter=',',\
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL
                                    ) for f in self.list_files]
        #make mp stuff ========
        self.queue_towrite = mp.Queue() #there is one queue in both operating modes.
        self.queue_signal_end = mp.Queue() #this queue not empty means "close"
        self.flag_closecalled = False #once close is called, writing would be disabled.
    
    def flush_and_close(self):
        self.flag_closecalled = True
        time.sleep(self.waiting_time_before_flush)
        self.queue_signal_end.put_nowait("stop")
    
    
    def run(self):
        while True:
            if(self.queue_signal_end.qsize()>0):
                #execute flush_and_close ==========
                self.flag_closecalled = True
                self._wrt_onclose()
                for f in self.list_files:
                    f.flush()
                    f.close()
                break
            else:
                #patrol the queue ==========
                self._wrt_patrol()
    
    
    def write(self, patient, str_towrite):
        '''
        Writes to file (s).
        Inputs.
            - patient: an instance of `Patient`. This argument is ignored
                    when operating in mode 1.
            - str_towrite: the string to be written to file.
        '''
        if(self.flag_closecalled == False):
            self.queue_towrite.put_nowait({"patient": patient, "str_towrite":str_towrite})
        else:
            print("`StreamWriter` cannot `write` after calling the `close` function.")
    
    def _wrt_patrol(self):
        '''
        Pops/writes one element from the queue
        '''
        if(self.queue_towrite.qsize() > 0):
            try:
                poped_elem = self.queue_towrite.get_nowait()
                if(self.op_mode == 1):
                    self.list_files[0].write(poped_elem["str_towrite"])
                elif(self.op_mode == 2):
                    patient, str_towrite = poped_elem["patient"], poped_elem["str_towrite"]
                    assert(patient in self.list_patients)
                    idx_patient = self.list_patients.index(patient)
                    self.list_files[idx_patient].write(str_towrite)
            except:
                pass
            
        
    def _wrt_onclose(self):
        '''
        Pops/writes all elements of the queue.
        '''
        qsize = self.queue_towrite.qsize()
        if(qsize > 0):
            for idx_elem in range(qsize):
                try:
                    poped_elem = self.queue_towrite.get_nowait()
                    if(self.op_mode == 1):
                        self.list_files[0].write(poped_elem["str_towrite"])
                    elif(self.op_mode == 2):
                        patient, str_towrite = poped_elem["patient"], poped_elem["str_towrite"]
                        assert(patient in self.list_patients)
                        idx_patient = self.list_patients.index(patient)
                        self.list_files[idx_patient].write(str_towrite)
                except:
                    pass
        
        
        
