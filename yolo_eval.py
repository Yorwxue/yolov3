import os,sys
sys.path.append('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/acerCar')
sys.path.append('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/acerCar/yolov3')
from scipy import misc
import json
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from yolov3.utils.utils import get_yolo_boxes, get_yolo_boxes_by_tf, crop_boxes, freeze_session


import configure as config_obj


def image_reader(img_path):
    input_image = misc.imread(img_path)
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]
    return input_image


if __name__ == "__main__":
    config = config_obj.Config(root_path=os.path.abspath(os.path.join(os.getcwd(), "..")))

    yolo_weights_path = config.weights_path
    yolo_model = load_model(yolo_weights_path)

    yolo_config_path = config.config_path
    with open(yolo_config_path, 'r') as config_buffer:
        yolo_config = json.load(config_buffer)

    filepath = "../testing/demo_car_org.png"
    filename = filepath.split('.')[-2].split('/')[-1]
    input_image = image_reader(filepath)

    input_image_list = list()
    input_image_list.append(input_image)

    yolo_boxes = get_yolo_boxes(yolo_model, input_image_list, 416, 416, yolo_config['model']['anchors'], 0.5, 0.45)

    car_imgs = crop_boxes(input_image, yolo_boxes[0])

    for car_idx, each_cat_img in enumerate(car_imgs):
        misc.imsave("../testing/%s_result_%d.%s" % (filename, car_idx, filepath.split('.')[-1]), each_cat_img)
        # misc.imshow(each_cat_img)

    # exit()

    # freeze
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in yolo_model.outputs],
                                  clear_devices=False)

    freeze_model_path = "../keras_freeze_model"
    freeze_model_name = "yolo_freeze_model.pb"
    tf.train.write_graph(frozen_graph, freeze_model_path, freeze_model_name, as_text=False)

    print("freeze model ok")

    graph = tf.Graph()
    graph_def = graph.as_graph_def()

    with tf.gfile.GFile(os.path.join(freeze_model_path, freeze_model_name), "rb") as fr:
        graph_def.ParseFromString(fr.read())

        with graph.as_default():
            tf.import_graph_def(graph_def, name="")

            yolo_input_placeholder = graph.get_tensor_by_name("input_1:0")
            yolo_output_list = [graph.get_tensor_by_name("conv_81/BiasAdd:0"),
                                graph.get_tensor_by_name("conv_93/BiasAdd:0"),
                                graph.get_tensor_by_name("conv_105/BiasAdd:0")]

            with tf.Session(graph=graph) as sess:
                yolo_boxes = get_yolo_boxes_by_tf(sess, yolo_input_placeholder, yolo_output_list,
                                                  [input_image], 416, 416, yolo_config['model']['anchors'],
                                                  0.5, 0.45)
