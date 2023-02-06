# TensorFlow2 Model Zoo


##  Help
[export-trained-model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)
## Step
### Prepare Datasets
#### 使用labelImg进行数据标注
在conda环境下 `pip install labelImg`
使用`labelImg` 命令打开软件，进行数据标注
![image.png](https://cdn.nlark.com/yuque/0/2023/png/33921914/1675652295042-3b4f6c6e-de89-4434-938b-b72941396d6d.png#averageHue=%23cccccc&clientId=u45988019-2c6c-4&from=paste&height=375&id=u9e4955b9&name=image.png&originHeight=1194&originWidth=1886&originalType=binary&ratio=1&rotation=0&showTitle=false&size=133443&status=done&style=none&taskId=ucc7752fb-d085-44b4-89ab-4864f5d8710&title=&width=592)
打开文件夹后，进行数据标注。

- **或者使用别人已经标注好的数据集**

标注好的数据应为`.xml`文件格式，并划分为训练集和测试集。
#### Create Label Map
```
item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
```
表示模型能够用来识别的品种类别，文件后缀应为`.pbtxt`
#### Convert tfrecord
[script_tf_convert](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py)
使用脚本，通过`label_map.pbtxt`将保存的`.xml`转为`.record`格式
> # Create train data:
> python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
> 
> # Create test data:
> python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

### Prepare Model

#### Download pre-trained model
[TF2_detection_ModelZoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
修改`pipeline.config`中的属性：
> num_classes = `**1**`
> fine_tune_checkpoint = `**Pre-tarined-model checkpoint path**`
> label_map_path = `**path label_map.pbtxt**`
> input_path = `**path .record**`
> 


#### Train model
根据测试集训练模型
`python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config`
> model_dir : 模型路径
> pipeline_config_path : 使用的pipeline路径

最终会生成该模型的checkpoint信息

#### Evaluate model（Option）
`python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn`
> model_dir : 模型路径
> pipeline_config_path : 使用的pipeline路径
> checkpoint_dir : train过程中生成的checkpoint路径


#### Export Trained Model
`python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model`

> input_type : 输入种类
> pipeline_config_path :  pipeline路径
> trained_checkpoint_dir ： checkpoint 路径
> output_directory: 输出路径


输出结果包含以下几部分：
> checkpoint 
> saved_model
> pipeline.config


### Batching Script
#### Load Model
```python
model_name = [
        'mobilenet1',
        'ssd_resnet50',
        'ssd_resnet101',
    ]
def load_model(model_name):
    model_dir = '/workspace/Tensorflow/workspace/training_demo/exported-models/'+model_name+'/saved_model'
    return tf.saved_model.load(str(model_dir)).signatures['serving_default']
```
#### Load Image

- Load Singal Image
```python
def load_img2tensor(image_path):
    image = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor
```

- Load Batch
```python
def load_img2batch(image_paths):
    batch = []
    for image in image_paths:
        batch.append(np.array(Image.open(image)))
    return tf.convert_to_tensor(batch, dtype='uint8')
```

#### Visualization

- 导入标签
```python
# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
```

- 可视化
```python
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections


# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


image_np_with_detections = image_np.copy()


viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False)


plt.figure()
plt.imshow(image_np_with_detections)
print('Done')
plt.savefig("./test.jpg")
```

## Tips

- 需要重新训练模型，训练结果与数据集大小，训练轮数相关
- 不能使用pre-trained-model中的checkpoint直接使用

## Code
[code](https://github.com/caliba/TF2_detection)
