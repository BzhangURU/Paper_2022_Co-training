# Paper_2022_Co-training

Public available code for paper "Stain Based Contrastive Co-training for Histopathological Image Analysis" accepted by MICCAI 2022 Workshop

Author list: Bodong Zhang, Beatrice Knudsen, Deepika Sirohi, Alessandro Ferrero, Tolga Tasdizen


Paper_Co-training_Prostate_Partly_Labeled_Group8.py:

Read prostate cancer dataset for training, validation and test. 

Currently the training set is from University of Utah and only available through material transfer agreement, please contact us(bodong.zhang [at] utah [dot] edu) for details.

The validation and test sets are publicly available, you can set program_mode='only_test' for testing.

To train start from scratch, set program_mode='normal_training'

To train by resuming best model ever got in previous training, set program_mode='resume_best_training'

To resume training from latest trained model, set program_mode='resume_latest_training'

To only test already trained model on validation and test set, set program_mode='only_test'

To run program, simply implement "python Paper_Co-training_Prostate_Partly_Labeled_Group8.py" in terminal




convert_RGB_to_H_or_E_prostate:
Our code to get Hematoxylin/Eosin channel images from original H&E images. 

We also have kidney cancer dataset from University of Utah that is available through material transfer agreement, please contact us(bodong.zhang [at] utah [dot] edu) for details.