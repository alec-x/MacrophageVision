# Train Macrophage phenotype detection CNN
# Alec Xu
# Multi-scale Design Lab
# University of British Columbia
# 2020

from argparse import ArgumentParser as arg_parser
from model import MacrophageCNNModel
import numpy as np
import pickle
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 16
        
def main(raw_args=None):
    parser = arg_parser(description="Train macrophage phenotyping CNN. \
        \n Used with dataset generated from createdataset.py")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for pickle containing dataframe.")
    args = parser.parse_args(raw_args)
    print("Training macrophage CNN with settings:") 

    configs = [[50, 5], [45, 5], [40, 5], [40,10], [35,10], [30,10], [30,20]]
    
    for i in range(1):#len(configs)):
        train_datagen = ImageDataGenerator(
            rotation_range=360,
            fill_mode="nearest",
            samplewise_std_normalization=True,
            samplewise_center=True,
            dtype=np.float32
        )

        test_datagen = ImageDataGenerator(
            samplewise_std_normalization=True,
            samplewise_center=True,
            dtype=np.float32,
        )

        print("\nRetrieving and augmenting training data")
        
        print("\nRetrieving and augmenting training data")
        train_generator = train_datagen.flow_from_directory(
            "C:\\Users\\Alec\\Documents\\Source\\Repos\\MDLMacrophageVision\\data\\training_set_phenotypes_manually_filtered\\training_mito",
            target_size=(96, 96),
            class_mode='binary',
            batch_size=BATCH_SIZE)

        print("\nRetrieving testing data")
        validation_generator = test_datagen.flow_from_directory(
            "C:\\Users\\Alec\\Documents\\Source\\Repos\\MDLMacrophageVision\\data\\training_set_phenotypes_manually_filtered\\validation_mito",
            target_size=(96, 96),
            class_mode='binary',
            batch_size=BATCH_SIZE)
            
        model = MacrophageCNNModel()
        model.compile(optimizer = 'adam', 
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics = ['accuracy'])

        history = model.fit_generator(
                train_generator,
                steps_per_epoch=2000,
                epochs=20,
                validation_data=validation_generator,
                validation_steps=400)
               
        hist_df = pd.DataFrame(history.history) 
        hist_csv_file = 'history' + str(configs[i][0]) + '_' + str(configs[i][1]) + '.csv'
        
        hist_df.to_csv(open(hist_csv_file, mode='w'))
                
if __name__=="__main__":
    main()