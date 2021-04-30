from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras

import numpy as np
import pickle
import sys
import math
# from matplotlib import pyplot as plt
# #tf.enable_eager_execution()
# from mpl_toolkits import mplot3d

# ref:[data,name]

inp=[]
out=[]
xx=[]
name=[]

data_file='./data_file_new/foil_uiuc.pkl'	
                
with open(data_file, 'rb') as infile:
    result = pickle.load(infile,encoding='bytes')
    print (result[-1:])    
            
    inp.extend(result[0])
    out.extend(result[1])
    xx.extend(result[2])
    name.extend(result[3])

inp=np.asarray(inp)
out=np.asarray(out)
out=out/0.25
name=np.asarray(name)

inp=np.reshape(inp,(len(inp),216,216,1))  

model_cnn=tf.keras.models.load_model('./selected_model/P8_C5F7/model_cnn/model_cnn_2000_0.000011_0.000076.hdf5')

model_get_para=Model(model_cnn.layers[0].input,model_cnn.layers[15].output)

#model_get_shape=Model(model_cnn.layers[16].input,model_cnn.layers[19].output)

cnn_out=model_get_para.predict(inp)

mm=[]
for i in range(8):
    mm.append([cnn_out[:,i].max(),cnn_out[:,i].min()])
mm=np.asarray(mm)
    
mm_scale=[]    
for i in range(8):    
    mm_scale.append(max(abs(mm[i])))
mm_scale=np.asarray(mm_scale)

cnn_out_scaled=cnn_out.copy()
for i in range(8): 
    cnn_out_scaled[:,i]=cnn_out_scaled[:,i]/mm_scale[i]

    
info='[cnn_out_scaled,cnn_out,name,mm_scale, xx,info: using pickled uiuc 1433 foil data tf2 ]'    
data2=[cnn_out_scaled,cnn_out,name,mm_scale,xx,info]
with open('cnn_gen_para_8_tanh_tf24_v1.pkl', 'wb') as outfile:
    pickle.dump(data2, outfile, pickle.HIGHEST_PROTOCOL)