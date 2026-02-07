import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.KNNGraph import KNNGraph
from datasets.KNNGraph2 import KNNGraph2
from datasets.AuxFunction import FFT
import pickle
import torch
from datasets.Generator import visualize_graph
import networkx as nx
#-------------------------------------------------------------

#Elevator
label = [1,2,3,4,5,6]
datasetname=[XXX]
normal_name=[]
dataname1 = []  
dataname2 = []  
dataname3 = []
dataname4 = []
dataname5 = []
dataname6 = []
axis=["_time"]
#generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data_root1 = os.path.join(root, datasetname[3])  
    data_root2 = os.path.join(root, datasetname[0])  
    data_root3 = os.path.join(root, datasetname[1]) 
    data_root4 = os.path.join(root, datasetname[2])    
    data_root5 = os.path.join(root, datasetname[4])   
    data_root6 = os.path.join(root, datasetname[5])   
    data_root7 = os.path.join(root, datasetname[6])  

    data=[]
    node_data=[]
    node_graph=[]
    num=0
    graph_num=0
    p=os.path.join(data_root1,normal_name[1])
    f=loadmat(p)['t']
    widnow=len(f)/sample_length
    while (num+1)<widnow:
        for folder_name in datasetname:
            if folder_name=="normal":
                l=0
                data_root = os.path.join(root, folder_name)
                file_name_list=[]
            elif folder_name=="XX":
                l=1
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            elif folder_name=="XX":
                l=2
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            elif folder_name=="XX":
                l=3
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            elif folder_name=="XX":
                l=4
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            elif folder_name=="XX":
                l=5
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            else: #right up roller
                l=6
                data_root = os.path.join(root, folder_name)
                file_name_list=[]  
            length_of_list = len(file_name_list)
            for file_name in file_name_list:
                file_path=os.path.join(data_root,file_name)
                nd=data_load2(sample_length,file_path,file_name,num=num,label=l,InputType=InputType,task=task)
                node_data+=nd
  
            graphset2=KNNGraph2(9,node_data,label=l,task=task,mat_len=length_of_list,graph_num=graph_num)
            graph_num += 1
            node_graph+=graphset2
        num+=1

    for i in tqdm(range(len(normal_name))):
        path1=os.path.join(data_root1,normal_name[i])
        data0=data_load(sample_length, path1, normal_name[i], label=0,InputType=InputType,task=task)
        data+=data0
    # path1=os.path.join(data_root1,normal_name[0])
    # data=data_load(sample_length,path1,axisname=normal_name[0],label=0,InputType=InputType,task=task)

    for i in tqdm(range(len(dataname1))):
        path2=os.path.join(data_root2,dataname1[i])
        data1=data_load(sample_length, path2, dataname1[i], label=1,InputType=InputType,task=task)
        data+=data1
    for i in tqdm(range(len(dataname2))):
        path3=os.path.join(data_root3,dataname2[i])
        data2=data_load(sample_length, path3, dataname2[i], label=2,InputType=InputType,task=task)
        data+=data2
    for i in tqdm(range(len(dataname3))):
        path4=os.path.join(data_root4,dataname3[i])
        data3=data_load(sample_length, path4, dataname3[i], label=3,InputType=InputType,task=task)
        data+=data3
    for i in tqdm(range(len(dataname4))):
        path5=os.path.join(data_root5,dataname4[i])
        data4=data_load(sample_length, path5, dataname4[i], label=4,InputType=InputType,task=task)
        data+=data4
    for i in tqdm(range(len(dataname5))):
        path6=os.path.join(data_root6,dataname5[i])
        data5=data_load(sample_length, path6, dataname5[i], label=5,InputType=InputType,task=task)
        data+=data5
    for i in tqdm(range(len(dataname6))):
        path7=os.path.join(data_root7,dataname6[i])
        data6=data_load(sample_length, path7, dataname6[i], label=6,InputType=InputType,task=task)
        data+=data6

    data+=node_graph
    return data


def data_load(signal_size, filename,axisname,label,InputType,task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)['t'][X:X]

    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1,)
    data=[]
    graph_list = []
    start,end=0,signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        if InputType == "TD":
            x = fl[start:end]
        elif InputType == "FD":
            x = fl[start:end]
            x = FFT(x)
        else:
            print("The InputType is wrong!!")

        data.append(x)

        start += signal_size
        end += signal_size
    graphset = KNNGraph(10,data,label,task)
    return graphset

def data_load2(signal_size, filename,axisname,num,label,InputType,task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)['t'][X:X]

    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1,)
    data=[]
    graph_list = []
    start,end=signal_size*num,signal_size*(num+1)
    if end<=len(fl):
        if InputType == "TD":
            x = fl[start:end]
        elif InputType == "FD":
            x = fl[start:end]
            x = FFT(x)
        else:
            print("The InputType is wrong!!")

        data.append(x)
    return data

class ELEKnn(object):
    num_classes = X

    def __init__(self, sample_length, data_dir,InputType,task):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            with open(os.path.join(self.data_dir, "ELEKnn.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            return train_dataset, val_dataset

