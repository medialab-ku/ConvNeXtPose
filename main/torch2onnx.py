import torch.nn as nn
import torch.nn.init as init
import torch
import argparse
import numpy as np
import onnxruntime
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import imageio.v2 as imageio

import argparse
from config import cfg
from base import Tester

import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_name', type=str, default='Convnext.onnx', dest='model_name')
    parser.add_argument('--jointnum', type=int, default=18)
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    name = args.model_name
    epoch = int(args.test_epoch)
    print(name, epoch)
    # cudnn.fastest = True
    # cudnn.benchmark = True
    # cudnn.deterministic = False
    # cudnn.enabled = True
    # tester = Tester()
    # tester.joint_num = args.jointnum
    # tester._make_model(epoch)

    # batch_size = 1
    # x = np.random.rand(batch_size, 3, 256, 256)
    # input_2D = torch.from_numpy(x.astype('float32')).cuda()
    # # print(input_2D.shape)
    # torch_out = tester.model(input_2D)
    # # print("output3D:    ", torch_out.shape)

    # # Export the model
    # torch.onnx.export(tester.model,               # model being run
    #                 input_2D,                         # model input (or a tuple for multiple inputs)
    #                 name,   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=12,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['output']) # the model's output names

    # ort_session = onnxruntime.InferenceSession(name)
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_2D)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # # print(ort_outs[1], ort_outs[2], ort_outs[3])
    # print(ort_outs[0].shape)
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    onnx_model = onnx.load(name)

    TF_PATH = "/tfmodel_out" # where the representation of tensorflow model will be stored

    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(TF_PATH)

    TFLITE_PATH = "baseline.tflite"

    PB_PATH = "saved_model.pb"

    # make a converter object from the saved tensorflow file
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)


    # def representative_dataset():

    #     dataset_size = 3

    #     for i in range(dataset_size):
    #         print(i)
    #         data = imageio.imread("../sample_images/" + "0000" + str(i) + ".png")
    #         data = np.resize(data, [1, 3, 256, 256])
    #         yield [data.astype(np.float32)]


    # converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8


    tf_lite_model = converter.convert()
    # Save the model.
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)

if __name__ == "__main__":
    main()