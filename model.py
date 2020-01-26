import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet import MobileNet #batch_norm layers throw exceptions with the default tensorflow version 1.3 -> fix by running 'pip install --upgrade tensorflow-gpu==1.4'
from keras.models import Model
from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Dropout, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

def image_generator(samples, batch_size=32):
    while True:
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
                       
            images = []
            angles = []
            for sample in batch_samples:
                filename = '/IMG/' + sample[0].split('/')[-1]
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = sample[1]
                images.append(image)
                angles.append(angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
        
def resize(image):
    import tensorflow as tf #has to be imported here to avoid bug: https://github.com/keras-team/keras/issues/5298
    return tf.image.resize_images(image, (128, 128))
        
def create_model(input_shape=(160, 320, 3)):
    mobilenet = MobileNet(weights='imagenet', alpha=1.0, include_top=False, input_shape=(128, 128, 3))   
    inp = Input(shape=input_shape)
    cropped_inp = Cropping2D(cropping=((54, 32), (0, 0)))(inp) #crop landscape and hood -> shape (74, 320, 3)
    resized_inp = Lambda(resize)(cropped_inp)
    normalized_inp = Lambda(lambda image: image/127.5 - 1.)(resized_inp)
    base_model = mobilenet(normalized_inp)
    x = GlobalAveragePooling2D()(base_model)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    pred = Dense(1, activation=None)(x)
    model = Model(inputs=inp, outputs=pred)
    print(model.summary())
    return model

def lr_camera_augmentation(data_df):
    samples = []
    for row in data_df.values:
        samples.append([row[0], float(row[3])]) # center image with label steering angle
        samples.append([row[1], float(row[3]+0.1)]) # left image with label steering angle + 0.1
        samples.append([row[2], float(row[3])-0.1]) # right image with label steering angle - 0.1
    return np.array(samples)

def main():
    data_df = pd.read_csv('/driving_log.csv')
    data_df.columns = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
    samples = lr_camera_augmentation(data_df)
    
    batch_size = 64
    epochs = 20
    
    train_samples, val_samples = train_test_split(samples, test_size=0.2)
    
    train_generator = image_generator(train_samples, batch_size=batch_size)
    val_generator = image_generator(val_samples, batch_size=batch_size)
    
    checkpoint = ModelCheckpoint(filepath='./models/weights.{epoch:02d}.h5', 
                                 verbose=1, 
                                 save_best_only=False,
                                 monitor='val_loss')
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=6,
                              verbose=1)
    
    model = create_model()
    model.compile(optimizer='Adam', loss='mse')
    model.fit_generator(train_generator,
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                        validation_data=val_generator,
                        validation_steps=np.ceil(len(val_samples)/batch_size),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpoint, earlystop])
    model.save('./models/last_model.h5')

if __name__=='__main__':
    main()