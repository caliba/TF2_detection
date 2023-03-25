from __future__ import print_function

import grpc
import pandas as pd
import tensorflow as tf
import argparse

from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import time
import numpy as np
import os
from timeit import default_timer as timer
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

gpu = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpu[0:], 'GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.experimental.set_memory_growth(gpu[1], True)


parser = argparse.ArgumentParser(
    description='TF Serving Test',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--server_address', default='localhost:8500', help='Tenforflow Model Server Address')
parser.add_argument('--image', default='../imgs/test2.jpg', help='Path to the image')
parser.add_argument('--model_name', default='mnist', help='model name')
args = parser.parse_args()

Cycle_times = 1# 每一个batch有效测算几次
BATCH_SIZE  = 2 # batchsize大小
TOLERANCE = 10 # 抛弃前多少个测试结果
MAX_MESSAGE_LENGTH = 1024*1024*1024

def load_img2tensor(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) 
    input_tensor = tf.convert_to_tensor(image,dtype='float32') # tf.float32
    input_tensor = input_tensor[tf.newaxis, ...]
    # image = np.array(Image.open(image_path))
    # input_tensor = tf.convert_to_tensor(image,dtype=tf.float32)
    # input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor


def load_imgbatch(img_path,batchsize):
    batch = []
    for i in range(batchsize):
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image,dtype='float32')
        batch.append(image)

    return tf.convert_to_tensor(batch,dtype='float32') # float32


def model_warmup(img_path):
    input_tensor = load_img2tensor(img_path)
    return input_tensor


def openchannel():
    channel = grpc.insecure_channel(args.server_address,
                                    options=[
                                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                    ]
    )

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    return stub
    


def inference(model_name, batch):
    stub = openchannel()

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    # model warmup
    input_tensor = model_warmup(args.image)
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(input_tensor))
    send_time = timer()
    result = stub.Predict(request, 60.0)  
    response_time = timer()
    print('%.3f' % ((response_time - send_time)*1000))

    batch_size = np.arange(1,BATCH_SIZE+1)
    for batch in batch_size:
        sum = 0.0
        batches = load_imgbatch(args.image,batch)
        request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(batches))
        # inference  probs
        for i in range(Cycle_times+TOLERANCE):
            if i >=TOLERANCE:
                send_time = time.time()
                result = stub.Predict(request, 60.0)  # 30 secs timeout
                response_time = timer()
                sum = sum + (time.time() - send_time)*1000
                print(tf.make_ndarray(result.outputs['probs']).shape)
        dur = sum/(Cycle_times)
        thrput = batch*1000
        thrput = thrput / dur
        print('batch size = {}, avg inference time is {:.3f} ms,throughput is {:.1f} req/s'.format(batch,dur,thrput))
   


def main():
    inference(args.model_name,2)



if __name__ == '__main__':
    main()
