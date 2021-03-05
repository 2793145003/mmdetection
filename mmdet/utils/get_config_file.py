import os

def get_config_file_from_params(params):
    file_path = "/output/mmdetection/configs/{}".format(params["args"]["model"])
    if params["args"]["model"] == "faster_rcnn":
        file_name = "faster_rcnn_{}".format(params["resnet_depth"])
        if params["use_caffe"]:
            if params["resnet_depth"] == "r50":
                file_name = "{}_caffe".format(file_name)
            if params["resnet_depth"] == "r101" and params["lr_schd"] == '1x':
                file_name = "{}_caffe".format(file_name)
        if params["resnet_depth"] == "r50" and params["use_caffe"]:
            file_name = "{}_{}".format(file_name, params["backbone_end"])
        else:
            file_name = "{}_fpn".format(file_name)
        if params["resnet_depth"] == "r50" and not params["use_caffe"]:
            if not params["loss"] == "l1":
                file_name = "{}_{}".format(file_name, params["loss"])
        if params["resnet_depth"] == "r50" and params["use_caffe"]:
            file_name = "{}_1x".format(file_name)
        elif params["resnet_depth"] == "r50" and not params["use_caffe"] and params["loss"] == "l1":
            file_name = "{}_1x".format(file_name)
        else:
            file_name = "{}_{}".format(file_name, params["lr_schd"])
        file_name = "{}_coco.py".format(file_name)
        file_path = os.path.join(file_path, file_name)   
    return file_path