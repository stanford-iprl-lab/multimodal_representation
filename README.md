# Multimodal Representation 

## requirements
`pip install -r requirements.txt`

## get dataset

`./script/download_data.sh`

## run training

python mini_main.py --config configs/training_default.yaml


ROBOT DATASET
----
action                   Dataset {50, 4}
contact                  Dataset {50, 50}
depth_data               Dataset {50, 128, 128, 1}
ee_forces_continuous     Dataset {50, 50, 6}
ee_ori                   Dataset {50, 4}
ee_pos                   Dataset {50, 3}
ee_vel                   Dataset {50, 3}
ee_vel_ori               Dataset {50, 3}
ee_yaw                   Dataset {50, 4}
ee_yaw_delta             Dataset {50, 4}
image                    Dataset {50, 128, 128, 3}
joint_pos                Dataset {50, 7}
joint_vel                Dataset {50, 7}
optical_flow             Dataset {50, 128, 128, 2}
proprio                  Dataset {50, 8}

