

import numpy as np
import os, sys
import math
from pathlib import Path
import re
import time
import random
import multiprocessing as mp
import openslide
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pydmed.utils.output
from pydmed.utils.output import StreamWriter


class Statistic:
    def __init__(self, stat, source_smallchunk):
         '''
         Inputs:
            - stat: a statistic from a smallchunk.
            - source_smallchunk: the `SmallChunk` from which the stat is collected.
         '''
         #grab privates
         self.stat = stat
         self.source_smallchunk = source_smallchunk
         self.source_smallchunk.data = "None, to avoid memory leak"


class StatCollector(object):
    def __init__(self, lightdl, str_collectortype, flag_visualizestats=False, kwargs_streamwriter=None):
        '''
        TODO:adddoc. str_collectortype can be "accum" or "saveall" or "stream_to_file"
        '''
        #grab initargs
        self.lightdl = lightdl
        self.str_collectortype = str_collectortype
        self.flag_visualizestats = flag_visualizestats
        self.kwargs_streamwriter = kwargs_streamwriter
        #make internals
        self.dict_patient_to_liststats = {patient:[] for patient in self.lightdl.dataset.list_patients}
        self.dict_patient_to_accumstat = {patient:None for patient in self.lightdl.dataset.list_patients}
        self._queue_onfinish_collectedstats = mp.Queue()
        if(self.str_collectortype.startswith("stream_to_file")):
            self.streamwriter = StreamWriter(lightdl.dataset.list_patients, **kwargs_streamwriter)
        
        
    
    def start_collecting(self):
        
        
        #make the following line tunable
        os.system("taskset -a -cp {} {}".format("0,1,2,3,4", os.getpid()))
        self.lightdl.start()
        if(self.str_collectortype.startswith("stream_to_file")):
            self.streamwriter.start()
            print(" statcollector.streamwriter started ")
        
        
        #TODO:make the following lines tunable.
        time.sleep(1*10)
        os.system("taskset -cp {} {}".format("5,6,7", os.getpid()))
        time.sleep(1*10)
        
        time.sleep(2) #TODO:remove
        time_lastcheck = time.time()+2 #TODO:make tunable
        time_lastupdate_visstats = time.time()+5 #TODO:make tunable
        count = 0
        #plot the visstats if needed,
        # ~ if(self.flag_visualizestats == True):
            # ~ pass
            # ~ self.logfile = open(r"lotstat.txt","w+")
            #TODO:handle visualizestats
            # ~ fig = plt.figure()
            # ~ ax = fig.add_subplot(111)
            # ~ x = [u for u in range(len(list(self.dict_patient_to_liststats.keys())))]
            # ~ y = [len(self.dict_patient_to_liststats[pat]) for pat in self.dict_patient_to_liststats.keys()] 
            # ~ line1, = ax.plot(x, y, 'r-')
            # ~ plt.show()
        while True:
            count +=1 
            #collect statistics ============
            retval_dl = self.lightdl.get()
            if((count%10) == 0):
                print(" got {} stats".format(count))
            list_collectedstats = self.get_statistics(retval_dl)
            list_patients = [st.source_smallchunk.patient for st in list_collectedstats]
            self._manage_stats(list_collectedstats, list_patients)
            # ~ #show the visstats if needed ====
            # ~ if((time.time()-time_lastupdate_visstats)>5):
                # ~ time_lastupdate_visstats = time.time()
                # ~ towrite = [len(self.dict_patient_to_liststats[pat]) for pat in self.dict_patient_to_liststats.keys()]
                # ~ str_towrite = ""
                # ~ for u in towrite:
                    # ~ str_towrite = str_towrite + str(u) + " , "
                # ~ str_towrite += "\n"
                # ~ self.logfile.write(str_towrite)
                # ~ print("wrote to file: {}".format(str_towrite))
                # ~ self.logfile.flush()
            #stop collecting if needed ======
            if((time.time()-time_lastcheck) > 5):#TODO:make tunable
                time_lastcheck = time.time()
                if(self.get_flag_finishcollecting() == True):
                    toret_onfinish_collectedstats = {}
                    #colllate all statistics
                    if(self.str_collectortype == "saveall"):
                        for patient in self.lightdl.dataset.list_patients:
                            toret_onfinish_collectedstats[patient] = self.collate_stats_onfinishcollecting(patient, self.dict_patient_to_liststats[patient])
                    elif(self.str_collectortype == "accum"):
                        for patient in self.lightdl.dataset.list_patients:
                            toret_onfinish_collectedstats[patient] = \
                                    self.dict_patient_to_accumstat[patient]
                    elif(self.str_collectortype.startswith("stream_to_file")):
                        self.streamwriter.flush_and_close()
                            
                    self._onfinish_collectedstats = toret_onfinish_collectedstats
                    if(self.str_collectortype.startswith("stream_to_file") == False):
                        pass #self._queue_onfinish_collectedstats.put_nowait(toret_onfinish_collectedstats)
                    time.sleep(3) #TODO:make tunable
                    #stop the lightdl
                    self.lightdl.pause_loading()
                    break
    
    def get_finalstats(self):
        try:
            if(self.str_collectortype.startswith("stream_to_file") == False):
                toret = self._onfinish_collectedstats #self._queue_onfinish_collectedstats.get()
                return toret
            
            # ~ return self._onfinish_collectedstats
        except:
            print("Error in getting the final collected stats. Is the StatCollector finished when you called `StatCollector.get_finalstats`?")

    def _manage_stats(self, list_collectedstats, list_patients):
        for n in range(len(list_patients)):
            patient = list_patients[n]
            if(self.str_collectortype == "saveall"):
                self.dict_patient_to_liststats[patient].append(list_collectedstats[n])
            elif(self.str_collectortype == "accum"):
                self.dict_patient_to_accumstat[patient] = self.accum_statistics(self.dict_patient_to_accumstat[patient],
                                                                                list_collectedstats[n],
                                                                                patient)
            elif(self.str_collectortype.startswith("stream_to_file")):
                self.streamwriter.write(patient, list_collectedstats[n].stat)
                
    
    
    @abstractmethod
    def accum_statistics(self, prev_accum, new_stat, patient):
        '''
        TODO:adddoc
        Outputs.
            - new_accum: TODO:adddoc.
        '''
        pass
    
    @abstractmethod
    def get_statistics(self, returnvalue_of_collatefunction):
        '''
        The abstract method that specifies the stat.collector's behaviour.
        Inputs.
            - TODO:adddoc, retval of collatefunc is by default, `x, list_patients, list_smallchunks`. But if you have overriden that output, ...
        Outputs.
            - list_liststats: a list of the same lenght as list_patients. Each element of the list is an instance of `Statistic`.
        '''
        pass
        
    
    @abstractmethod
    def get_flag_finishcollecting(self): #TODO: make this method patient-wise, so that user would not need to work with self.dict_patient_to_liststats
        pass
        
    @abstractmethod
    def collate_stats_onfinishcollecting(self, patient, list_collectedstats):
        '''
        This function is called when collecting is finished.
        This fucntion should collate and return the collected stats for the input patient and collected stats.
        Inputs.
            - patient: the patient, and instance of of utils.data.Patient.
            - list_collectedstats: the collected statistics, a list of objects as returned by `StatCollector.get_statistics`.
        '''
        pass
    
