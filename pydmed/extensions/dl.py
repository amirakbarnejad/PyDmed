

'''
Extensions related to PyDmed's dataloader.

'''

import math
import numpy as np
import random
import pydmed


class LabelBalancedDL(pydmed.lightdl.LightDL):
    '''
    This dataloader makes sure that the returned smallchunks are have a balanced label
            frequency.
    Inputs.
        - func_getlabel_of_patient: a function that takes in a `Patient` and returns
            the corresponding label. The returned smallchunks are balanced in terms of
            this label.
        - ... other arguments, same as LightDL, 
        https://github.com/amirakbarnejad/PyDmed/blob/8575ea991fe464b6e451d1a3381f9026581153da/pydmed/lightdl.py#L292
    '''
    def __init__(self, func_getlabel_of_patient, *args, **kwargs):
        '''
        Inputs.
        - func_getlabel_of_patient: a function that takes in a `Patient` and returns
            the corresponding label. The returned smallchunks are balanced in terms of
            this label.
        - ... other arguments, same as LightDL, 
        https://github.com/amirakbarnejad/PyDmed/blob/8575ea991fe464b6e451d1a3381f9026581153da/pydmed/lightdl.py#L29
        '''
        super(LabelBalancedDL, self).__init__(*args, **kwargs)
        #grab privates
        self.func_getlabel_of_patient = func_getlabel_of_patient
        #make separate lists for different classes ====
        possible_labels = list(
                  set(
                   [self.func_getlabel_of_patient(patient)\
                    for patient in self.dataset.list_patients]
                  )     
                )
        dict_label_to_listpatients = {label:[] for label in possible_labels}
        for patient in self.dataset.list_patients:
            label_of_patient = self.func_getlabel_of_patient(patient)
            dict_label_to_listpatients[label_of_patient].append(patient)
        self.possible_labels = possible_labels
        self.dict_label_to_listpatients = dict_label_to_listpatients
    
    def initial_schedule(self):
        print("override initsched called.")
        #split numbigchunks to lists of almost equal length ======
        avg_inbin = self.const_global_info["num_bigchunkloaders"]/len(self.possible_labels)
        avg_inbin = math.floor(avg_inbin)
        list_binsize = [avg_inbin for label in self.possible_labels]
        num_toadd = self.const_global_info["num_bigchunkloaders"]-\
                    avg_inbin*len(self.possible_labels)
        for n in range(num_toadd):
            list_binsize[n] += 1
        #randomly sample patients from different classes =====
        toret_list_patients = []
        for idx_bin, size_bin in enumerate(list_binsize):
            label = self.possible_labels[idx_bin]
            toret_list_patients = toret_list_patients +\
                            random.choices(self.dict_label_to_listpatients[label], k=size_bin)
        return toret_list_patients
    
    def schedule(self):
        print("override sched called.")
        #get initial fields ==============================
        list_loadedpatients = self.get_list_loadedpatients()
        list_waitingpatients = self.get_list_waitingpatients()
        schedcount_of_waitingpatients = [self.get_schedcount_of(patient)\
                                         for patient in list_waitingpatients]
        #patient_toremove is selected randomly =======================
        patient_toremove = random.choice(list_loadedpatients)
        #choose the patient to load ================
        minority_label = pydmed.utils.minimath.multiminority(
                [self.func_getlabel_of_patient(patient) for patient in list_loadedpatients]
                        )
        toadd_candidates = self.dict_label_to_listpatients[minority_label]
        weights = 1.0/(1.0+np.array(
                            [self.get_schedcount_of(patient) for patient in toadd_candidates]
                        ))
        weights[weights==1.0] =10000000.0 #if the case is not loaded sofar, give it a high prio
        patient_toload = random.choices(toadd_candidates,\
                                        weights = weights, k=1)[0]
        return patient_toremove, patient_toload


