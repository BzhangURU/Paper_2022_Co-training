import os
import numpy as np
from PIL import Image

def load_images_from_folder(main_folder, subfolder, target_H_main_folder, target_E_main_folder,sub_folder_index):
    for filename in os.listdir(main_folder+'/'+subfolder):
        file_in_H_exist=os.path.exists(os.path.join(target_H_main_folder,subfolder,filename))
        file_in_E_exist=os.path.exists(os.path.join(target_E_main_folder,subfolder,filename))
        subfolder_H_exist = os.path.exists(os.path.join(target_H_main_folder,subfolder))
        if not subfolder_H_exist:
            os.makedirs(os.path.join(target_H_main_folder,subfolder))
        subfolder_E_exist = os.path.exists(os.path.join(target_E_main_folder,subfolder))
        if not subfolder_E_exist:
            os.makedirs(os.path.join(target_E_main_folder,subfolder))
        if file_in_H_exist and file_in_E_exist:
            print(filename+' already exists......')
        if filename.find('.png')>=0 and (file_in_H_exist==False or file_in_E_exist==False):
            print('processing subfolder '+str(sub_folder_index)+' '+filename)
            image = np.asarray(Image.open(os.path.join(main_folder,subfolder,filename)))#(height, width, channel)
            #image = np.transpose(image, (2, 0, 1))#change to (channel, height, width)
            image = image.astype('float32')
            image_ones=np.ones(image.shape,dtype=np.float32)
            image = np.maximum(image,image_ones)#change all pixel intensity =0 to =1
            width=int(image.shape[2])
            height= int(image.shape[1])
            
            RGB_absorption=np.log10(255.0*(image_ones/image))
            #If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            gray_image_H=np.dot(RGB_absorption,np.array([1.838, 0.0341, -0.76]))
            gray_image_H=np.clip(gray_image_H,0.0,1.0)
            gray_image_H=255.0*gray_image_H
            gray_image_H = gray_image_H.astype('int')

            gray_image_E=np.dot(RGB_absorption,np.array([-1.373, 0.772, 1.215]))
            gray_image_E=np.clip(gray_image_E,0.0,1.0)
            gray_image_E=255.0*gray_image_E
            gray_image_E = gray_image_E.astype('int')

            img_H = Image.fromarray(np.uint8(gray_image_H))
            img_H.save(os.path.join(target_H_main_folder,subfolder,filename))

            img_E = Image.fromarray(np.uint8(gray_image_E))
            img_E.save(os.path.join(target_E_main_folder,subfolder,filename))

if __name__ == '__main__':
    #GP_list = ['benign', 'cancer']
    GP_list = ['TCGA-2A-A8VV-01Z-00-DX1/benign', 'TCGA-2A-A8VV-01Z-00-DX1/cancer',
    'TCGA-EJ-7123-01Z-00-DX1/benign', 'TCGA-EJ-7123-01Z-00-DX1/cancer',
    'TCGA-EJ-7314-01Z-00-DX1/benign', 'TCGA-EJ-7314-01Z-00-DX1/cancer',
    'TCGA-EJ-7315-01Z-00-DX1/benign', 'TCGA-EJ-7315-01Z-00-DX1/cancer',
    'TCGA-EJ-7328-01Z-00-DX1/benign', 'TCGA-EJ-7328-01Z-00-DX1/cancer',
    'TCGA-EJ-7784-01Z-00-DX1/benign', 'TCGA-EJ-7784-01Z-00-DX1/cancer',
    'TCGA-EJ-7793-01Z-00-DX1/benign', 'TCGA-EJ-7793-01Z-00-DX1/cancer',
    'TCGA-G9-6329-01Z-00-DX1/benign', 'TCGA-G9-6329-01Z-00-DX1/cancer',
    'TCGA-G9-6333-01Z-00-DX1/benign', 'TCGA-G9-6333-01Z-00-DX1/cancer',
    'TCGA-G9-6363-01Z-00-DX1/benign', 'TCGA-G9-6363-01Z-00-DX1/cancer',
    'TCGA-G9-6364-01Z-00-DX1/benign', 'TCGA-G9-6364-01Z-00-DX1/cancer',
    'TCGA-G9-6384-01Z-00-DX1/benign', 'TCGA-G9-6384-01Z-00-DX1/cancer',
    'TCGA-G9-6385-01Z-00-DX1/benign', 'TCGA-G9-6385-01Z-00-DX1/cancer',
    'TCGA-G9-6494-01Z-00-DX1/benign', 'TCGA-G9-6494-01Z-00-DX1/cancer',
    'TCGA-G9-7522-01Z-00-DX1/benign', 'TCGA-G9-7522-01Z-00-DX1/cancer',
    'TCGA-HC-8216-01Z-00-DX1/benign', 'TCGA-HC-8216-01Z-00-DX1/cancer',
    'TCGA-HC-A8D1-01Z-00-DX1/benign', 'TCGA-HC-A8D1-01Z-00-DX1/cancer',
    'TCGA-J4-8200-01Z-00-DX1/benign', 'TCGA-J4-8200-01Z-00-DX1/cancer']

    # for i in range(len(GP_list)):
    #     load_images_from_folder('./dataset/Training_set_256x256',GP_list[i],\
    #         './dataset/Training_set_256x256_H_my_code',\
    #         './dataset/Training_set_256x256_E_my_code', i)

    for i in range(len(GP_list)):
        load_images_from_folder('./dataset/Test_set_256x256',GP_list[i],\
            './dataset/Test_set_256x256_H_my_code',\
            './dataset/Test_set_256x256_E_my_code')

