# Train Macrophage phenotype detection CNN
# Alec Xu
# Multi-scale Design Lab
# University of British Columbia
# 2020

from argparse import ArgumentParser as arg_parser
import functional_model
import numpy as np
import pickle
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 1
        
def main(raw_args=None):
    parser = arg_parser(description="Train macrophage phenotyping CNN. \
        \n Used with dataset generated from createdataset.py")
    parser.add_argument("path", action="store", type=str, \
                        help="Source path for pickle containing dataframe.")
    args = parser.parse_args(raw_args)
    print("Training macrophage CNN with settings:") 
    
    data = pickle.load(open(args.path + "\\measured.pickle", "rb"))
    data = data.sample(frac=1).reset_index(drop=True)
    configs = [[50, 5], [45, 5], [40, 5], [40,10], [35,10], [30,10], [30,10]]
    
    for i in range(len(configs)):
        cd206_pos = data[data["CD206_diff"] >= configs[i][0]]
        cd206_pos["label"] = "1"
        
        cd206_neg = data[data["CD206_diff"] <= configs[i][1]]
        cd206_neg["label"] = "0"
        num_neg = len(cd206_neg)
        num_pos = len(cd206_pos)
        print("# negative samples: " + str(num_neg))
        print("# positive samples: " + str(num_pos))
        num_total = 0
        if num_neg > num_pos:
            num_total = 2*num_pos
            print("Randomly sampling down negatives... Samples used: " + str(num_total))
            cd206_neg = cd206_neg.sample(n=num_pos)
        else:
            num_total = 2*num_neg
            print("Randomly sampling down positives... Samples used: " + str(num_total))
            cd206_pos = cd206_pos.sample(n=num_neg)
        
        num_training = int(len(cd206_neg) * 0.8)
        num_validation = int(len(cd206_neg) * 0.2)
        
        cd206_pos_training = cd206_pos.head(num_training)
        cd206_pos_validation = cd206_pos.tail(num_validation)
        cd206_neg_training = cd206_neg.head(num_training)
        cd206_neg_validation = cd206_neg.tail(num_validation)
        
        cd206_training = cd206_pos_training.append(cd206_neg_training)
        cd206_validation = cd206_pos_validation.append(cd206_neg_validation)

        train_datagen = ImageDataGenerator(
            #rotation_range=360,
            #fill_mode="nearest",
            horizontal_flip=True,
            vertical_flip=True,
            samplewise_center=True,
            samplewise_std_normalization=True,
            dtype=np.float32
        )
        
        test_datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            dtype=np.float32,
        )

        print("\nRetrieving and augmenting training data")
        train_generator_mito = train_datagen.flow_from_dataframe(
                dataframe=cd206_training,
                directory=args.path,
                x_col="Mito",
                y_col="label",
                target_size=(96, 96),
                class_mode='binary',
                batch_size=BATCH_SIZE)
        
        train_generator_bf = train_datagen.flow_from_dataframe(
                dataframe=cd206_training,
                directory=args.path,
                x_col="BF",
                y_col="label",
                target_size=(96, 96),
                class_mode='binary',
                batch_size=BATCH_SIZE)
        
        print("\nRetrieving testing data")
        validation_generator_mito = test_datagen.flow_from_dataframe(
                dataframe=cd206_validation,
                directory=args.path,
                x_col="Mito",
                y_col="label",
                target_size=(96, 96),
                class_mode='binary',
                batch_size=BATCH_SIZE)
                
        validation_generator_bf = test_datagen.flow_from_dataframe(
                dataframe=cd206_validation,
                directory=args.path,
                x_col="BF",
                y_col="label",
                target_size=(96, 96),
                class_mode='binary',
                batch_size=BATCH_SIZE)                
                
        data_list = []
        data_list_2 = []
        batch_index = 0
        while batch_index <= train_generator_bf.batch_index:
            data = train_generator_bf.next()
            data_2 = train_generator_mito.next()
            data_list.append(data[0])
            data_list_2.append(data_2[0])
            batch_index = batch_index + 1
        
        from matplotlib import pyplot as plt
        print(data_list[0].shape)
        rearranged = np.transpose(data_list[0], axes=[0, 3, 1, 2])
        rearranged_2 = np.transpose(data_list_2[0], axes=[0, 3, 1, 2])
        #plt.figure()
        f, axarr = plt.subplots(3,2) 
        axarr[0][0].imshow(rearranged[0][0])
        axarr[1][0].imshow(rearranged[0][1])
        axarr[2][0].imshow(rearranged[0][2])
        axarr[0][1].imshow(rearranged_2[0][0])
        axarr[1][1].imshow(rearranged_2[0][1])
        axarr[2][1].imshow(rearranged_2[0][2])        
        plt.show()
        """
        
        model = functional_model.get_model()
        model.compile(optimizer = 'adam', 
            loss=tf.keras.losses.mean_squared_error,
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
        """
if __name__=="__main__":
    main()