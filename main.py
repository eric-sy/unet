from model import *
from data import *
from itertools import tee
import warnings
from numpy.distutils.tests.test_fcompiler_nagfor import TestNagFCompilerVersions
warnings.simplefilter(action='ignore', category=FutureWarning)

#train_base = 'data/membrane/train'
#test_base = 'data/membrane/test'

train_base = 'C:\\_ERIC\\_datasets\\discs\\train'
model_path = 'C:\\_ERIC\\_datasets\\discs\\unet_discs.hdf5'

def train():
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(2,train_base,'image_2','label_2',data_gen_args, target_size=(320,320))
     
    model = unet(input_size = (320,320,1))
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint])

def eval(test_img,test_predict):
    model = load_model(model_path)
    test_imgs_name, test_imgs_path = testFileName(test_img,'.jpg')
    batch_size = 10
    if len(test_imgs_name) > 100:
        batch_size = 100
    if len(test_imgs_name) > 1000:
        batch_size = 1000    
    for i in range(0, len(test_imgs_name)-batch_size+1, batch_size):
        testGene = testGeneratorMine(test_imgs_path[i:i+batch_size],'.jpg')
        results = model.predict_generator(testGene,batch_size,verbose=1)
        saveResult(test_predict,results,test_imgs_name[i:i+batch_size])
    testGene = testGeneratorMine(test_imgs_path[len(test_imgs_name)-batch_size:],'.jpg')
    results = model.predict_generator(testGene,batch_size,verbose=1)
    saveResult(test_predict,results,test_imgs_name[len(test_imgs_name)-batch_size:])
    

if __name__ == "__main__":
    train()
#     eval(test_img = 'C:\\_ERIC\\_datasets\\dataset_v10.2\\left', test_predict = 'C:\\_ERIC\\_datasets\\dataset_v12')
#     eval(test_img = 'C:\\_ERIC\\_datasets\\dataset_v10.2\\right', test_predict = 'C:\\_ERIC\\_datasets\\dataset_v12')