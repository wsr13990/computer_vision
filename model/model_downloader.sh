model_name=$1
precision="FP16"
python "D:/Program Files (x86)/IntelSWTools/openvino_2020.1.033/deployment_tools/tools/model_downloader/downloader.py" --name $model_name -o "D:/BELAJAR/C++/facial_recognition/model/"

#python "D:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/downloader.py" --name person-vehicle-bike-detection-crossroad-0078 --precisions FP16 -o "D:/BELAJAR/C++/facial_recognition/model/"