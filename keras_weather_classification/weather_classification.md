주제 : 날씨 분류

### class
* cloudy
* rain
* shine
* sunrise

### image sample
![](https://github.com/Deplim/ML_practice/blob/main/keras_weather_classification/image/image_sample.png?raw=true)

### 목차

- **neural network structure**
- **Source Code**
- **Test loss, Test accuracy**
- **Training 마지막 5개 epoch 및 test loss/accuracy 출력 화면**
- **학습 곡선 그래프**
- **참여 소감**

---

# 1. **neural network structure**

```
Model: "sequential_24"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_292 (Conv2D)          (None, 128, 128, 64)      1792      
_________________________________________________________________
conv2d_293 (Conv2D)          (None, 128, 128, 16)      9232      
_________________________________________________________________
max_pooling2d_112 (MaxPoolin (None, 64, 64, 16)        0         
_________________________________________________________________
conv2d_294 (Conv2D)          (None, 64, 64, 32)        4640      
_________________________________________________________________
conv2d_295 (Conv2D)          (None, 64, 64, 32)        9248      
_________________________________________________________________
max_pooling2d_113 (MaxPoolin (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_296 (Conv2D)          (None, 32, 32, 64)        18496     
_________________________________________________________________
conv2d_297 (Conv2D)          (None, 32, 32, 64)        36928     
_________________________________________________________________
conv2d_298 (Conv2D)          (None, 32, 32, 64)        36928     
_________________________________________________________________
max_pooling2d_114 (MaxPoolin (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_299 (Conv2D)          (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_300 (Conv2D)          (None, 16, 16, 128)       147584    
_________________________________________________________________
conv2d_301 (Conv2D)          (None, 16, 16, 128)       147584    
_________________________________________________________________
max_pooling2d_115 (MaxPoolin (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_302 (Conv2D)          (None, 8, 8, 128)         147584    
_________________________________________________________________
conv2d_303 (Conv2D)          (None, 8, 8, 128)         147584    
_________________________________________________________________
conv2d_304 (Conv2D)          (None, 8, 8, 128)         147584    
_________________________________________________________________
max_pooling2d_116 (MaxPoolin (None, 4, 4, 128)         0         
_________________________________________________________________
flatten_22 (Flatten)         (None, 2048)              0         
_________________________________________________________________
dense_66 (Dense)             (None, 512)               1049088   
_________________________________________________________________
dropout_38 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_67 (Dense)             (None, 128)               65664     
_________________________________________________________________
dropout_39 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_68 (Dense)             (None, 4)                 516       
=================================================================
Total params: 2,044,308
Trainable params: 2,044,308
Non-trainable params: 0
_________________________________________________________________
```

# 2. Source Code

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

data_load_seed = 123
```

```python
from tensorflow.keras import preprocessing

data_dir = "./dataset"
batch_size = 32
img_height = 128
img_width = 128

train_ds = preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  seed=data_load_seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
class_num = len(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

def get_model4(img_height, img_width, class_num, opt):
    model = Sequential()
    model.add(Conv2D(input_shape=(img_height,img_width,3),filters=64,kernel_size=(3,3),
                     padding="same", activation="relu"))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.005)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=512,activation="relu"))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units=128,activation="relu"))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units=class_num, activation="softmax"))

    model.compile(optimizer=opt, 
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    
    return model
```

```python
def flip_image(image, label):
  image = tf.image.flip_left_right(image)
  return image, label

def crop_image(image, label): 
  image = tf.image.random_crop(image, size=[32, img_height-18,img_width-18, 3])
  image = tf.image.resize(image, [img_height, img_width])
  return image, label

def flip_and_crop(image, label):
  image = tf.image.flip_left_right(image)
  image = tf.image.central_crop(image, central_fraction=0.9)
  image = tf.image.resize(image, [img_height, img_width])
  return image, label

import copy
a_train_ds_3 = copy.copy(train_ds)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
temp_ds = train_ds.map(crop_image, num_parallel_calls=AUTOTUNE)
temp_ds = temp_ds.take(24)
a_train_ds_3 = a_train_ds_3.concatenate(temp_ds)

plt.figure(figsize=(10, 10))
for images, labels in temp_ds.take(1):
  print(images[0].shape)
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
temp_ds = train_ds.map(flip_and_crop, num_parallel_calls=AUTOTUNE)
a_train_ds_3 = a_train_ds_3.concatenate(temp_ds)

plt.figure(figsize=(10, 10))
for images, labels in temp_ds.take(1):
  print(images[0].shape)
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
temp_ds = train_ds.map(flip_image, num_parallel_calls=AUTOTUNE)
a_train_ds_3 = a_train_ds_3.concatenate(temp_ds)

plt.figure(figsize=(10, 10))
for images, labels in temp_ds.take(1):
  print(images[0].shape)
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

a_train_ds_3.batch(32).shuffle(123)
print("\n data length after agumentation : ", len(a_train_ds_3))
```

```python
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

n_train_ds = a_train_ds_3.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(n_train_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

n_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(n_val_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
```

```python
opt = Adam(lr=0.0001, decay=0.0002)
model2 = get_model4(img_height, img_width, class_num, opt)
model2.summary()
```

```python
'''
  data 에 imbalance 가 있어서 "가장 많은 클래스 데이터 개수 / 각 클래스의 데이터 개수"
  를 CrossEntropyLoss의 가중치로 줌 
'''
disp = model2.fit(
  n_train_ds,
  validation_data=n_val_ds,
  epochs=300,
  class_weight = {0: 356/300, 1: 356/215, 2 : 356/253, 3: 356/356}
)
```

```python
# summarize history for accuracy 
plt.plot(disp.history['accuracy']) 
plt.plot(disp.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
```

```python
score = model.evaluate(n_val_ds, verbose=0)
print("Test loss : ", score[0])
print("Test accuaracy : ", score[1])
```

# 3. **Test loss, Test accuracy**

![Untitled](https://github.com/Deplim/ML_practice/blob/main/keras_weather_classification/image/score.png?raw=true)

# 4. **Training 마지막 5개 epoch 및 test loss/accuracy 출력 화면**

```
Epoch 296/300
99/99 [==============================] - 14s 137ms/step - loss: 0.1510 - accuracy: 0.9992
 - val_loss: 0.6430 - val_accuracy: 0.9199
Epoch 297/300
99/99 [==============================] - 14s 138ms/step - loss: 0.1515 - accuracy: 0.9987
 - val_loss: 0.7024 - val_accuracy: 0.9139
Epoch 298/300
99/99 [==============================] - 14s 137ms/step - loss: 0.1546 - accuracy: 0.9975
 - val_loss: 0.6904 - val_accuracy: 0.9139
Epoch 299/300
99/99 [==============================] - 14s 137ms/step - loss: 0.1480 - accuracy: 1.0000
 - val_loss: 0.7202 - val_accuracy: 0.9139
Epoch 300/300
99/99 [==============================] - 14s 138ms/step - loss: 0.1537 - accuracy: 0.9973
 - val_loss: 0.6675 - val_accuracy: 0.9318
```

# 5. **학습 곡선 그래프**

![Untitled](https://github.com/Deplim/ML_practice/blob/main/keras_weather_classification/image/history.png?raw=true)

# 6. 참여 소감

처음에는 vgg16모델을 기준으로 어느 정도 loss 가 줄어드는지 확인한 후 모델 최적화(파라미터 수를 줄이는) 와 정규화를 통해서 최대한 가벼우면서 과적합이 없는 모델을 얻으려고 하였습니다. 기본적으로 bias 를 충분히 줄일 수 있는 모델이어야 했기에 epoch 50 이 넘어가기 전에 train data set 의 accuaracy 가 1에 도달할 수 있어야 한다는 기준을 놓고 cnn layer 의 filter 수를 줄여나갔습니다. 그 이후에는 data augumentation 과 l2 regularization 과 dropout 을 이용한 정규화를 진행했습니다. 

이번 문제에서는 이미지에서 날씨 class 를 분류하는 모델을 만들어야 했는데, 날씨를 분류하는 것에서는 색감과 밝기 등이 중요한 역할을 한다고 생각하여 augumentation 과정에서는 밝기와 색감 조정 보다는 좌우반전, crop 위주로 사용하였습니다. 마지막으로 loss 와 accuracy 가 안정적으로 수렴하게 하기 위하여 augumentation 이후의 데이터에 shuffle 을 추가하고 learning rate decay 를 적용하였으며, 실제로 이전 trial 들보다 안정적으로 수렴하는 모습을 볼 수 있었습니다.

기본적으로 validation data set 의 성능이 잘 나오려면 train data set 의 데이터가 충분해야 하는데(해당 문제의 실제 데이터 분포를 설명할 수 있을 만큼) 만약 이 문제에서 성능 향상을 위한 작업을 더 진행한다면, 추가적인 데이터 수집이 불가능하다는 가정 하에는 어떻게 가장 적합한 augumentation 을 할 수 있는 지에 집중할 것 같습니다.

풀이를 하면서 어려웠던 점은 같은 작업을 하는데 사용할 수 있는 툴이 많아서 오히려 어떤 것을 사용해야 할지 고민하는데 생각보다 시간이 많이 들어갔다는 부분이었습니다. keras 로 프로젝트를 할 때에 똑같은 데이터를 로드하고 전처리 하더라도 사용할 수 있는 모듈이나 방법이 다양하게 있었습니다. (ex. 데이터를 tensorflow 의 dataset 객체를 활용하여 다룰지, numpy 상태에서 처리할지) 이번에는 tensorflow 의 documentation 에 추천되어있는 방법 위주로 사용하였지만, 나중에 다시 keras 를 사용하는 프로젝트를 한다면 사전에 최대한 다양한 코드를 보고 자신의 프로젝트에 맞는 방법을 선택할 수 있어야겠다는 생각을 했습니다.
