import bilsm_crf_model
import os
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,TensorBoard
import tensorflow as tf

#设置训练的轮数和每次训练的批次大小
EPOCHS = 100
batch_size = 128
#设置GPU
import keras.backend.tensorflow_backend as KTF
#配置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 使用编号为1，2号的GPU
config = tf.ConfigProto()

config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

#两轮不提高就降低学习率
reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=3, mode='auto',min_lr = 1e-10)

#添加checkpoint，earlystop
file_name = "model_{epoch:03d}.weights"
checkpoint = ModelCheckpoint(filepath="./checkpoint/"+file_name, verbose=1, save_best_only=False,save_weights_only=True)
#六轮不提高就终止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
csv_logger = CSVLogger('./checkpoint/word_training.log')

#设置tensorboard
log_filepath = "./checkpoint/train_log"
tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)  

model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
#加载以前训练的模型
# model.load_weights("./model/best_one.weights")

# 训练模型
model.fit(train_x, train_y,batch_size=batch_size,shuffle = True,
          epochs=EPOCHS, validation_data=[test_x, test_y],
          callbacks=[csv_logger,early_stopping,checkpoint,reduce_lr,tb_cb])

