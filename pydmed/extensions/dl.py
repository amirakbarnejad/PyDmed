

'''
Extensions related to PyDmed's dataloader.

'''

import math
import numpy as np
from abc import ABC, abstractmethod
import random
import time
import openslide
import copy
import torchvision
import pydmed
import pydmed.lightdl
from pydmed import *
from pydmed.lightdl import *



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
        # ~ print("override initsched called.")
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
        # ~ print("override sched called.")
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



class SlidingWindowSmallChunkCollector(pydmed.lightdl.SmallChunkCollector):
    def __init__(self, *args, **kwargs):
        '''
        Inputs: 
            - mode_trmodainortest (in const_global_info): a strings in {"train" and "test"}.
                We need this mode because, e.g., colorjitter is different in training and testing phase.
        '''
        super(SlidingWindowSmallChunkCollector, self).__init__(*args, **kwargs)
        self.mode_trainortest = kwargs["const_global_info"]["mode_trainortest"]
        assert(self.mode_trainortest in ["train", "test"])
        self.flag_unschedme = False
        #grab privates
        self.tfms_onsmallchunkcollection = self.const_global_info["pdmreserved_tfms_onsmallchunkcollection"]
        # ~ \
            # ~ torchvision.transforms.Compose([
            # ~ torchvision.transforms.ToPILImage(),\
            # ~ torchvision.transforms.ToTensor(),\
            # ~ torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                             # ~ std=[0.229, 0.224, 0.225])
        # ~ ])
        
    def slice_by_slidingwindow(self, W, kernel_size, stride):
        '''
        Slices the length `W` by `kernel_size` and `stride`.
        Outputs the number of shifts. 
        '''
        if((stride%(W-kernel_size)) == 0):
            toret = int((W-kernel_size)/stride) + 1
        else:
            toret = math.floor((W-kernel_size)/stride) + 2
        return toret
    
    @abstractmethod     
    def extract_smallchunk(self, call_count, bigchunk, last_message_fromroot):
        '''
        Extract and return a smallchunk. Please note that in this function you have access to 
        self.bigchunk, self.patient, self.const_global_info.
        Inputs:
            - list_bigchunks: the list of extracted bigchunks.
        '''"list_polygons"
        #list of statuses ======
        status_busy = "busy"
        status_idle = "idle"
        status_idlefinished = "idlefinished" #=========
        
        
        try:
            #exit the call if needed.
            if(self.flag_unschedme == True):
                return None
            
            #handle the case where the returned BigChunk is None.
            if(isinstance(bigchunk, str)):
                assert(bigchunk == "None-Bigchunk")
                if(call_count == 0):
                    self.set_status(status_idlefinished)
                    self.set_checkpoint({"idx_bigrow":np.inf})
                    print("&&&&&&&&&&&&& idx_bigrow set to np.inf &&&&&&&&&&")
                return None
            
            #if callcount is zero, increase the checkpoint by 1
            if(call_count == 0):
                checkpoint = self.get_checkpoint()
                if(checkpoint == None):
                    self.set_checkpoint({"idx_bigrow":1})
                else:
                    self.set_checkpoint({"idx_bigrow":checkpoint["idx_bigrow"]+1})
                self.set_status(status_busy)
                self.flag_unschedme = False
           
            #extract fields from const_global_info ====
            intorfunc_opslevel = self.const_global_info["pdmreserved_intorfunc_opslevel"]
            kernel_size = self.const_global_info["pdmreserved_kernel_size"]
            stride = self.const_global_info["pdmreserved_stride"]
            func_patient_to_fnameimage = \
                self.const_global_info["pdmreserved_func_patient_to_fnameimage"]
            if(isinstance(intorfunc_opslevel, int) == True):
                attention_levelidx = intorfunc_opslevel
            else:
                attention_levelidx = intorfunc_opslevel(self.patient)
            
            
            #attention_levelidx = attention_levelidx
            W, H = bigchunk.data.shape[1], bigchunk.data.shape[0]
            #osimage.level_dimensions[self.const_global_info["attention_levelidx"]]
            w, h = H+0, H+0
            x_begin = int(call_count*w)
            x_end = x_begin + w
            num_cols = self.slice_by_slidingwindow(W, kernel_size, stride)
            flag_auxlastcol = False
            if(x_end > W):
                x_end = W
                x_begin = W-w
                flag_auxlastcol = True
            
            WSI_H = bigchunk.dict_info_of_bigchunk["WSI_H"]
            bigchunk_numbigrows = bigchunk.dict_info_of_bigchunk["num_bigrows"]
            bigchunk_idxbigrow = bigchunk.dict_info_of_bigchunk["idx_bigrow"]
            flag_lastbigchunk = (bigchunk_idxbigrow == (bigchunk_numbigrows-1))
            #(bigchunk.dict_info_of_bigchunk["y"]+2*h) > WSI_H
            if(np.random.rand() < 0.2):
                print("bigchunk.dict_info_of_bigchunk['y'] = {}".format(
                                bigchunk.dict_info_of_bigchunk["y"]
                            )
                     )
                print("h = {}".format(h))
                print("WSI_H = {}".format(WSI_H))
                print("flag_lastbigchunk = {}".format(flag_lastbigchunk))
            
            if(call_count > (num_cols-1)):
                #x out of boundary
                if(flag_lastbigchunk == False):
                    self.flag_unschedme = True #next calls will return immediately.
                    self.set_status(status_idle)
                    print("status was set to status_idle")
                    return None
                elif(flag_lastbigchunk == True):
                    self.flag_unschedme = True #next calls will return immediately.
                    self.set_status(status_idlefinished)
                    print("status was set to status_idlefinished")
                    return None
            else:
                #X within boundary ==== 
                np_smallchunk = bigchunk.data[:, x_begin:x_end, :]
                #apply the transformation ===========
                if(self.tfms_onsmallchunkcollection != None):
                    toret = self.tfms_onsmallchunkcollection(np_smallchunk)
                    toret = toret.cpu().detach().numpy() #[3 x 224 x 224]
                    toret = np.transpose(toret, [1,2,0]) #[224 x 224 x 3]
                else:
                    toret = np_smallchunk
                #wrap in SmallChunk
                smallchunk = SmallChunk(data=toret,\
                                        dict_info_of_smallchunk={
                                            "x":x_begin, "y":0,\
                                            "flag_auxlastcol":flag_auxlastcol
                                        },\
                                        dict_info_of_bigchunk = bigchunk.dict_info_of_bigchunk,\
                                        patient=bigchunk.patient
                                )
                return smallchunk
        except Exception as e:
            print("An exception occurred when collecting smallchunk.")
            print(str(e))
            return None
    
    
class SlidingWindowBigChunkLoader(pydmed.lightdl.BigChunkLoader):
    def slice_by_slidingwindow(self, W, kernel_size, stride):
        '''
        Slices the length `W` by `kernel_size` and `stride`.
        Outputs the number of shifts. 
        '''
        if((stride%(W-kernel_size)) == 0):
            toret = int((W-kernel_size)/stride) + 1
        else:
            toret = math.floor((W-kernel_size)/stride) + 2
        return toret
    
    @abstractmethod
    def extract_bigchunk(self, last_message_fromroot):
        '''
        Extract and return a bigchunk. 
        Please note that in this function you have access to
        self.patient and self.const_global_info.
        '''
        try:
            #get `idx_bigrow` to be extracted =====
            checkpoint = self.get_checkpoint()
            if(checkpoint == None):
                idx_bigrow = 0
            else:
                idx_bigrow = checkpoint["idx_bigrow"]
                
            #extract fields from const_global_info ====
            intorfunc_opslevel = self.const_global_info["pdmreserved_intorfunc_opslevel"]
            kernel_size = self.const_global_info["pdmreserved_kernel_size"]
            stride = self.const_global_info["pdmreserved_stride"]
            func_patient_to_fnameimage = self.const_global_info["pdmreserved_func_patient_to_fnameimage"]
            
            #compute some constants ====
            fname_wsi = func_patient_to_fnameimage(self.patient) #os.path.join(wsi.rootdir, wsi.relativedir)
            osimage = openslide.OpenSlide(fname_wsi)
            if(isinstance(intorfunc_opslevel, int) == True):
                attention_levelidx = intorfunc_opslevel
            else:
                attention_levelidx = intorfunc_opslevel(self.patient)
            w, h = kernel_size, kernel_size #in the taget level
            W, H = osimage.level_dimensions[attention_levelidx] #size in the target level
            num_bigrows = self.slice_by_slidingwindow(H, kernel_size, stride)
            
            #extract the target row ====
            y_begin = int(stride*idx_bigrow) #size in the target level
            y_begin_at_level0 = int(y_begin*osimage.level_downsamples[attention_levelidx])
            if(y_begin_at_level0 < 0):
                y_begin_at_level0 = 0
            if((num_bigrows-1) < idx_bigrow):
                return "None-Bigchunk" #it happens when a done case is loaded by schedule.
            y_end = y_begin + h
            flag_from_auxbigrow = False
            if(y_end > H):
                #refine the range for the last bigrow
                y_end = H-1
                y_begin = H-kernel_size-1
                y_begin_at_level0 = int(y_begin*osimage.level_downsamples[attention_levelidx])
                flag_from_auxbigrow = True
            pil_bigchunk = osimage.read_region(
                                [0, y_begin_at_level0],
                                attention_levelidx,
                                [W,h]
                              )
            np_bigchunk = np.array(pil_bigchunk)[:,:,0:3]
            self.patient.dict_records["precomputed_opsimage"] =  "none"
            patient_without_foregroundmask = copy.deepcopy(self.patient)
            for k in patient_without_foregroundmask.dict_records.keys():
                if(k.startswith("precomputed")):
                    patient_without_foregroundmask.dict_records[k] = None
            bigchunk = BigChunk(data=np_bigchunk,\
                                dict_info_of_bigchunk={
                                    "W":W, "H":H, "x":0, "y":y_begin,
                                    "WSI_W":W, "WSI_H":H,
                                    "flag_from_auxbigrow":flag_from_auxbigrow,
                                    "num_bigrows":num_bigrows,
                                    "idx_bigrow":idx_bigrow
                                },\
                                patient=patient_without_foregroundmask
                         )
            return bigchunk
        except Exception as exception:
            print("extractbigchunk failed for patient {}.".format(self.patient))
            print(exception)
            return "None-Bigchunk"
        
class SlidingWindowDL(pydmed.lightdl.LightDL):
    def __init__(
        self, intorfunc_opslevel, kernel_size,
        func_patient_to_fnameimage, stride, mininterval_loadnewbigchunk, 
        tfms_onsmallchunkcollection, 
        *args, **kwargs):
        '''
        Inputs.
            - intorfunc_opslevel: it can be either an integer, or a function. 
                    This argument specifies the level of the openslide image
                    from which the patches are extracted.
                    If it is an integer, e.g., 0, the DL will return from level 0.
                    If it is a function, it has to take in a patient and return
                    the intended level based on the input patient.
            - kernel_size: an integer, the width of the sliding windon.
            - stride: stride of the sliding window, an integer.
            - func_patient_to_fnameimage: a function. 
                This function has to take in a `Patient` and return the aboslute path
                of the image (or WSI).
            - mininterval_loadnewbigchunk: a floating point number, minimum time (in seconds)
                between loading two bigchunks.
                This number depends on how big each `BigChunk` is as well as system specs.
            - tfms_onsmallchunkcollection: a callable object, the transformations to be applied to each SmallChunk (i.e. each tile).
            
                
        '''
        super(SlidingWindowDL, self).__init__(*args, **kwargs)
        #grab privates ====
        kwargs["type_bigchunkloader"] = SlidingWindowBigChunkLoader
        kwargs["type_smallchunkcollector"] = SlidingWindowSmallChunkCollector
        self.time_lasteffective_sched = None
        self.list_itwaslastbigchunk = []
        self._dict_patient_to_lastschedtime = {
                    patient:None for patient in self.dataset.list_patients
            } #to avoid unscheduling right after scheduling.
        #place the input arguments within `const_global_info`
        self.const_global_info["pdmreserved_intorfunc_opslevel"] = intorfunc_opslevel
        self.const_global_info["pdmreserved_kernel_size"] = kernel_size
        self.const_global_info["pdmreserved_stride"] = stride
        self.const_global_info["pdmreserved_func_patient_to_fnameimage"] = func_patient_to_fnameimage
        self.const_global_info["pdmreserved_mininterval_loadnewbigchunk"] = mininterval_loadnewbigchunk
        self.const_global_info["pdmreserved_tfms_onsmallchunkcollection"] = tfms_onsmallchunkcollection
    def initial_schedule(self):
        #Default is to choose randomly from dataset.
        toret =  random.choices(
                    self.dataset.list_patients,\
                    k=self.const_global_info["num_bigchunkloaders"]
                )
        for patient in toret:
            self._dict_patient_to_lastschedtime[patient] = time.time()
        return toret
    
    def schedule(self):
        '''
        This function is called when schedulling a new patient, i.e., loading a new BigChunk.
        This function has to return:
            - patient_toremove: the patient to remove, an instance of `utils.data.Patient`.
            - patient_toload: the patient to load, an instance of `utils.data.Patient`.
        In this function, you have access to the following fields:
            - self.dict_patient_to_schedcount: given a patient, returns the number of times the patients has been schedulled in dl, a dictionary.
            - self.list_loadedpatients:
            - self.list_waitingpatients:
            - TODO: add more fields here to provide more flexibility. For instance, total time that the patient have been loaded on DL.
        '''
        print("sched called. <><><><<><><><><><><><><><><><><><><><><><><><><><><><>")
        #list of statuses ======
        status_busy = "busy"
        status_idle = "idle"
        status_idlefinished = "idlefinished" #=========
        try:
            #see if any BigChunkLoader process is still running ====
            list_activesubprocs = list(self.active_subprocesses)
            for subproc in list_activesubprocs:
                if(subproc.get_flag_bigchunkloader_terminated() == False):
                    print("returned None, None (case BigChunkLoader still running).")
                    return None, None
            
            #see if all patients are done ====
            print("set of it was last bigchunk = {}\n".format(set(self.list_itwaslastbigchunk)))
            if(set(self.list_itwaslastbigchunk) == set(self.dataset.list_patients)):
                print("dl is done ========================")
#                 self._queue_imdone.put_nowait("imdone")
                #print(" returned None, None (case imdone)")                
                return pydmed.lightdl.PYDMEDRESERVED_HALTDL, None

            #see if current time is not too close to last effective schedule ======
            if(self.time_lasteffective_sched == None):
                self.time_lasteffective_sched = time.time()
            mininterval_loadnewbigchunk = \
                self.const_global_info["pdmreserved_mininterval_loadnewbigchunk"]
            if((time.time()-self.time_lasteffective_sched) < mininterval_loadnewbigchunk):
                print("     but returned None, None (case 1)")
                return None, None #let the loaded subprocesses continue working 
            
            #get initial fields ==============================
            list_loadedpatients = self.get_list_loadedpatients()
            list_activesubprocs = list(self.active_subprocesses)
            list_statuses = [subproc.get_status()\
                             for subproc in list_activesubprocs]
            print("<><><><><><><><><><><><><><><><><><><><> list of statuses = {}".\
                    format(list_statuses))
            #update list_itwaslastbigchunk
            for idx_status, status in enumerate(list_statuses):
                if(status == status_idlefinished):
                    patient_itwaslastbigchunk = list_activesubprocs[idx_status].patient
                    if((patient_itwaslastbigchunk in self.list_itwaslastbigchunk) == False):
                        self.list_itwaslastbigchunk.append(patient_itwaslastbigchunk)
            
            #find a candiate to unschedule (with priority to idle+unfinished) ======
            flag_exists_idlebutnotfinished = False
            for idx_status, status in enumerate(list_statuses):
                if(status == status_idle):
                    flag_exists_idlebutnotfinished = True
            flag_foundcandidate_unsched = False
            for idx_status, status in enumerate(list_statuses):
                if(flag_exists_idlebutnotfinished == True):
                    criteria_unsched = (status == status_idle)
                else:
                    criteria_unsched = (status==status_idlefinished)
                
                if(criteria_unsched == True):
                    patient_unschedcandidate = list_activesubprocs[idx_status].patient
                    if((time.time()-\
                           self._dict_patient_to_lastschedtime[patient_unschedcandidate])>1.0):
                        flag_foundcandidate_unsched = True
                        idx_subproc_toremove = idx_status
                        break
                   
            if(flag_foundcandidate_unsched == False):
                #if no unsched candidates were found, return None, None
                print("     but returned None, None (case 2) "+\
                     "flag_foundcandidate_unsched = False")
                return None, None #let the loaded subprocesses continue working
            elif(flag_foundcandidate_unsched == True):
                #find a candidate to schedule =====
                list_waitingpatients = self.get_list_waitingpatients()
                for patient in self.list_itwaslastbigchunk:
                    if(patient in list_waitingpatients):
                        list_waitingpatients.remove(patient)
                        print("{} removed because it was last bigchunk.".format(patient))
                patient_toremove = list_loadedpatients[idx_subproc_toremove]
                waitingpatients_schedcount = [self.get_schedcount_of(patient)\
                                              for patient in list_waitingpatients]
                if(len(list_waitingpatients)>0):
                    patient_toload = random.choice(list_waitingpatients)
                else:
                    #load a new patient, which indeed won't return any big/small chunk.
                    list_waitingpatients = self.get_list_waitingpatients()
                    patient_toload = random.choice(list_waitingpatients)
                    
                #check if the sched/unsched is useful (i.e. not swapping two finished patients)
                flag_usefule_sched = not(
                            (patient_toload in self.list_itwaslastbigchunk) and\
                            (patient_toremove in self.list_itwaslastbigchunk)
                          )
                if(flag_usefule_sched == True):
                    print("<><><><> sched returned a new pair")
                    self.time_lasteffective_sched = time.time()
                    self._dict_patient_to_lastschedtime[patient_toload] = time.time()
                    return patient_toremove, patient_toload
                else:
                    print("    but returned None, None. The sched is flagged as unuseful.")
                    return None, None
        except Exception as e:
            print("exception in schedule.")
            print(str(e))
            print("     but returned None, None (case 3)")
            return None, None



