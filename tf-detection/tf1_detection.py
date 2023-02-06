import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pynvml

pynvml.nvmlInit()  # 初始化
gpu_device_count = pynvml.nvmlDeviceGetCount()
for gpu_index in range(gpu_device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    gpu_name = str(pynvml.nvmlDeviceGetName(handle))
    print("GPU{} :{}".format(gpu_index+1,gpu_name))

pynvml.nvmlShutdown()  # 关闭管理工具

import numpy as np
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpu[0:], 'GPU')
# print(gpu)
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.experimental.set_memory_growth(gpu[1], True)
import pathlib
import time
from PIL import Image
from object_detection.utils import ops as util_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as util_vis
util_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

Cycle_times = 6 # 每一个batch有效测算几次
BATCH_SIZE  = 10 # batchsize大小
TOLERANCE = 3 # 抛弃前多少个测试结果
img_path = 'cat.jpg'

def load_model(model_name):
     # 根据url下载
    # url = 'http://download.tensorflow.org/models/object_detection'+model_name+'.tar.gz'
    # model_dir = tf.keras.utils.get_file(fname=model_name, origin=url, untar=True)
    # model_dir = pathlib.Path(model_dir)/"saved_model"

    # /root/.keras/datasets/model_name/saved_model 
    # 从路径指定加载模型
    model_dir = '/root/.keras/datasets/'+model_name+'/saved_model' 
    # print(model_dir)
    return tf.saved_model.load(str(model_dir)).signatures['serving_default']


def load_img2tensor(image_path):
    image = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor


def load_img2batch(image_paths):
    batch = []
    for image in image_paths:
        batch.append(np.array(Image.open(image)))
    return tf.convert_to_tensor(batch, dtype='uint8')


def main():
    model_name = [
        'ssd_mobilenet_v1_coco_2018_01_28',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        'ssd_inception_v2_coco_2018_01_28',
        'faster_rcnn_inception_v2_coco_2018_01_28',
    ]

    img = img_path

    for j in range(len(model_name)):
        # 模型加载的时间
        print()
        print()
        print('model name : {}'.format(model_name[j]))
        print()
        t_load = time.time()
        model_fn = load_model(model_name[j])
        print('load model time is {:.3f}'.format(1000*(time.time()-t_load)))
        # 模型第一次推理的启动时间
        t_infer_st = time.time()
        model_fn(load_img2tensor(img))
        print('First inference time for model is {:.3f}'.format(1000*(time.time()-t_infer_st)))

        batch_size = np.arange(1,BATCH_SIZE+1)

        for i in batch_size:
            print('inference: batch size = {}'.format(i))
            sum = 0.0
            batch = load_img2batch([img] * i)
            for j in range(Cycle_times+TOLERANCE):
                _start = time.time()
                model_fn(batch)
                if j>=TOLERANCE:
                    sum = sum + 1000*(time.time()-_start)
                
                    # print('  infer-{} t={:.3f}ms'.format(j-TOLERANCE+1, 1000*(time.time()-_start)))
            # 计算平均处理时间
            dur = sum/(Cycle_times)
            # 计算吞吐量
            thrput = i*1000
            thrput = thrput / dur
            print('batch size = {}, avg inference time is {:.3f} ms,throughput is {:.1f} req/s'.format(i,dur,thrput))

if __name__ == '__main__':

    # models from tf1 model zoo
    """
    Model name                  Speed(ms)  COCO mAP
    ssd_mobilenet_v1_fpn_coco   56          32
    ssd_resnet_50_fpn_coco      76          35
    faster_rcnn_resnet101_coco  106         32
    faster_rcnn_nas             1833        43
    """
    #'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    # model_name = [
    #     'ssd_mobilenet_v1_coco_2018_01_28',
    #     'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    #     'faster_rcnn_resnet101_coco_2018_01_28',
    #     'faster_rcnn_nas_coco_2018_01_28',
    # ]
    main()

