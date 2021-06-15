
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


def Tensor3DtoPdmcsvrow(np_input, smalchunk_input):
    '''
    Converts a Tensor of shape [C x H x W] to pdmcsv format.
    Inputs.
        - np_input: a numpy array of shape [CxHxW].
        - smallchunk_input: an instnace of SmallChunk,
                            the smallchunk that that the tensor corresponds to.
    '''
    chw = list(np_input.shape)
    str_toret = str([
        smalchunk_input.dict_info_of_bigchunk["y"],\
        smalchunk_input.dict_info_of_smallchunk["x"],\
        smalchunk_input.dict_info_of_bigchunk["H"],\
        smalchunk_input.dict_info_of_bigchunk["W"],\
        smalchunk_input.dict_info_of_smallchunk["patch_levelidx"],\
        smalchunk_input.dict_info_of_smallchunk["kernel_size"],
        smalchunk_input.dict_info_of_bigchunk["downsample_of_patchlevel"],
        chw[0], chw[1], chw[2]
       ])[1:-1]+ "," +\
       str(np_input.flatten().tolist())[1:-1] + "\n"
    return str_toret
    


def pdmcsvtoarray(fname_pdmcsv, func_WSIxyWHval_to_rasterpoints, scale_upsampleraster=1.0):
    '''
    Converts a pdmcsv file to an array.
    Inputs.
        - fname_pdmcsv: a string, the path-filename to the pdmcsv file.
        - outputsize: a float, the scale of output. The default value is 1.0 meaning the output
            array is not scaled.
        - func_WSIxyval_to_rasterpoints: a function.
            - Inputs.
                x: a number, as in one line of the pdm.csv file.
                y: a number, as in one line of the pdm.csv file.
                W: an integer.
                H: an integer.
                val: list of values.
            -Outputs.
                - list_x_onraster:
                - list_y_onraster:
                - list_val_onraster:
    '''
    #read the file line-by-line =====
    file_pdmcsv = open(fname_pdmcsv, 'r')
    count_line = 0
    dict_raster = {}
    while True:
        count_line += 1
        line = file_pdmcsv.readline() 
        
        if not line: 
                break
              
        list_numbers = line.split(",")
        for idx, u in enumerate(list_numbers):
            if(isinstance(list_numbers[idx], str)):
                if("None" in list_numbers[idx]):
                    list_numbers[idx] = np.nan
        list_numbers = [float(u) for u in list_numbers]
        
        if(count_line == 1):
            H, W = list_numbers[2], list_numbers[3]
            H, W = int(H), int(W)
        
        #order: y,x,H,W,....  
        y, x = list_numbers[0], list_numbers[1]
        patch_levelidx = list_numbers[4]
        kernel_size = list_numbers[5]
        downsample_of_patchlevel = list_numbers[6]
        c = int(list_numbers[7])
        h = int(list_numbers[8])
        w = int(list_numbers[9])
        val = list_numbers[10:] #np.mean(np.array([list_numbers[4:]]))
        
        #convert the points to raster space using the function
        list_x_onraster, list_y_onraster, val = func_WSIxyWHval_to_rasterpoints(
                                            x, y, W, H,
                                            patch_levelidx,
                                            kernel_size,
                                            downsample_of_patchlevel,
                                            c, h, w, val
                                        )
        #np_x_onraster, np_y_onraster = np.array(list_x_onraster), np.array(list_y_onraster)
        for idx_rasterpoint in range(len(list_x_onraster)):
            dict_raster["({},{})".format(
                     math.floor(list_x_onraster[idx_rasterpoint]),
                     math.floor(list_y_onraster[idx_rasterpoint])
                    )
                ] = val[idx_rasterpoint]
    
    #convert dict_raster to np.ndarray =====
    list_allrasterx, list_allrastery = [], []
    for u in dict_raster.keys():
        x, y = u[1:-1].split(',')
        x, y = float(x), float(y) 
        if(scale_upsampleraster > 1.0):
            x, y = scale_upsampleraster*x, scale_upsampleraster*y
        x, y = math.floor(x), math.floor(y)
        list_allrasterx.append(x); list_allrastery.append(y)
    list_allrasterx = list(set(list_allrasterx))
    list_allrastery = list(set(list_allrastery))
    list_allrasterx.sort(); list_allrastery.sort()
    max_x, max_y = np.max(list_allrasterx), np.max(list_allrastery)
    output_raster = np.zeros((len(list_allrastery), len(list_allrasterx), c))
    num_totalloops = len(list(dict_raster.keys()))
    count = 0
    for u in dict_raster.keys():
        count += 1
        if((count%10000) == 0):
            print("    >>>>>> Interpolation in progress: point {} out of {}. Please wait .... .".format(count, num_totalloops), end="\r")
        x,y = u[1:-1].split(',')
        x, y = float(x), float(y)
        if(scale_upsampleraster > 1.0):
            x, y = scale_upsampleraster*x, scale_upsampleraster*y
        x, y = math.floor(x), math.floor(y)
        output_raster[list_allrastery.index(y), list_allrasterx.index(x),:] = dict_raster[u]
    #fill-in the zeros if scale_upsample>1.0
    if(scale_upsampleraster > 1.0):
        list_output_scaled = []
        for count_c in range(c):
            f = interp2d(
                np.array(list_allrasterx),
                np.array(list_allrastery),
                output_raster[:,:,count_c], kind='cubic'
            )
            output_raster_scaled_forchannel = f(
                   np.array([j for j in range(max_x)]),
                   np.array([i for i in range(max_y)])
                 )
            list_output_scaled.append(output_raster_scaled_forchannel)
        return np.stack(list_output_scaled, 2)
    return output_raster


class DefaultWSIxyWHvaltoRasterPoints:
    def __init__(self):
        pass
        
    def func_WSIxyWHval_to_rasterpoints(
                self, x, y, W, H,
                patch_levelidx, kernel_size,
                downsample_of_patchlevel,
                c, h, w, val):
        assert(isinstance(val, list))
        assert((c*h*w)== len(val))
        np_val = np.reshape(val, [c,h,w])
        
        size_blockonraster = h #np.sqrt(len(val))
        scale_wsi_to_raster = kernel_size/size_blockonraster
        x_onraster = (x+0.0)/scale_wsi_to_raster
        y_onraster = (y+0.0)/scale_wsi_to_raster
        #make list_x_onraster and list_y_onraster ======
        np_x_onraster = np.array([[j for j in range(int(size_blockonraster))]\
                             for i in range(int(size_blockonraster))]).flatten()+x_onraster
        np_y_onraster = np.array([[i for j in range(int(size_blockonraster))]\
                             for i in range(int(size_blockonraster))]).flatten()+y_onraster
        list_x_onraster = np_x_onraster.tolist()
        list_y_onraster = np_y_onraster.tolist()
        toret_val = []
        for i in range(h):
            for j in range(w):
                toret_val.append(np_val[:,i,j])
        return list_x_onraster, list_y_onraster, toret_val



class SlidingWindowSmallChunkCollector(pydmed.lightdl.SmallChunkCollector):
    def __init__(self, *args, **kwargs):
        '''
        Inputs: 
            - mode_trmodainortest (in const_global_info): a strings in {"train" and "test"}.
                We need this mode because, e.g., colorjitter is different in training and testing phase.
        '''
        super(SlidingWindowSmallChunkCollector, self).__init__(*args, **kwargs)
        if("mode_trainortest" in kwargs["const_global_info"].keys()):
            self.mode_trainortest = kwargs["const_global_info"]["mode_trainortest"]
        else:
            self.mode_trainortest = "test"
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
            vertbar_overlaptheprevpatch = "None"
            if(x_end > W):
                prev_x_end = x_end-stride
                x_end = W
                x_begin = W-w
                vertbar_overlaptheprevpatch = kernel_size-(x_end-prev_x_end)
                flag_auxlastcol = True
            
            WSI_H = bigchunk.dict_info_of_bigchunk["WSI_H"]
            bigchunk_numbigrows = bigchunk.dict_info_of_bigchunk["num_bigrows"]
            bigchunk_idxbigrow = bigchunk.dict_info_of_bigchunk["idx_bigrow"]
            flag_lastbigchunk = (bigchunk_idxbigrow == (bigchunk_numbigrows-1))
            #(bigchunk.dict_info_of_bigchunk["y"]+2*h) > WSI_H
            if(np.random.rand() < 0.2):
                print("Please wait. SlidingWindowDL is still working ..... ")
                pass
            
            if(call_count > (num_cols-1)):
                #x out of boundary
                if(flag_lastbigchunk == False):
                    self.flag_unschedme = True #next calls will return immediately.
                    self.set_status(status_idle)
                    return None
                elif(flag_lastbigchunk == True):
                    self.flag_unschedme = True #next calls will return immediately.
                    self.set_status(status_idlefinished)
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
                                            "patch_levelidx":attention_levelidx,
                                            "kernel_size":kernel_size,
                                            "flag_auxlastcol":flag_auxlastcol,
                                            "vertbar_overlaptheprevpatch":vertbar_overlaptheprevpatch
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
            downsample_of_patchlevel = osimage.level_downsamples[attention_levelidx] 
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
            horizbar_overlaptheprevpatch = "None"
            if(y_end > H):
                #refine the range for the last bigrow
                old_y_end, old_y_begin = y_end+0.0, y_begin+0.0
                prev_y_end, prev_y_begin = old_y_end-stride+0.0, old_y_begin-stride+0.0 
                y_end = H-1
                y_begin = H-kernel_size-1
                y_begin_at_level0 = int(y_begin*osimage.level_downsamples[attention_levelidx])
                flag_from_auxbigrow = True
                horizbar_overlaptheprevpatch = kernel_size - (y_end-prev_y_end)
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
                                    "downsample_of_patchlevel":downsample_of_patchlevel,
                                    "num_bigrows":num_bigrows,
                                    "idx_bigrow":idx_bigrow,
                                    "flag_from_auxbigrow":flag_from_auxbigrow,
                                    "horizbar_overlaptheprevpatch": horizbar_overlaptheprevpatch
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
        #list of statuses ======
        status_busy = "busy"
        status_idle = "idle"
        status_idlefinished = "idlefinished" #=========
        try:
            #see if any BigChunkLoader process is still running ====
            list_activesubprocs = list(self.active_subprocesses)
            for subproc in list_activesubprocs:
                if(subproc.get_flag_bigchunkloader_terminated() == False):
                    return None, None
            
            #see if all patients are done ====
            #TODO:print? print("set of it was last bigchunk = {}\n".format(set(self.list_itwaslastbigchunk)))
            if(set(self.list_itwaslastbigchunk) == set(self.dataset.list_patients)):
                print("\n =================== dl's job is done! ========================")
#                 self._queue_imdone.put_nowait("imdone")
                #print(" returned None, None (case imdone)")                
                return pydmed.lightdl.PYDMEDRESERVED_HALTDL, None

            #see if current time is not too close to last effective schedule ======
            if(self.time_lasteffective_sched == None):
                self.time_lasteffective_sched = time.time()
            mininterval_loadnewbigchunk = \
                self.const_global_info["pdmreserved_mininterval_loadnewbigchunk"]
            if((time.time()-self.time_lasteffective_sched) < mininterval_loadnewbigchunk):
                return None, None #let the loaded subprocesses continue working 
            
            #get initial fields ==============================
            list_loadedpatients = self.get_list_loadedpatients()
            list_activesubprocs = list(self.active_subprocesses)
            list_statuses = [subproc.get_status()\
                             for subproc in list_activesubprocs]
            
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
                return None, None #let the loaded subprocesses continue working
            elif(flag_foundcandidate_unsched == True):
                #find a candidate to schedule =====
                list_waitingpatients = self.get_list_waitingpatients()
                for patient in self.list_itwaslastbigchunk:
                    if(patient in list_waitingpatients):
                        list_waitingpatients.remove(patient)
                        
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
                    self.time_lasteffective_sched = time.time()
                    self._dict_patient_to_lastschedtime[patient_toload] = time.time()
                    return patient_toremove, patient_toload
                else:
                    return None, None
        except Exception as e:
            print("exception in schedule.")
            print(str(e))
            print("     but returned None, None (case 3)")
            return None, None
