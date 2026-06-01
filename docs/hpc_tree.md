这是远程服务器的le-wm的目录结构，代码文件与当前目录一致。除了少数大文件，像权重，数据集，输出日志等未同步到本地。
我让你写的代码都要基于远程服务器的目录来做

le-wm$ tree
.
├── assets
│   └── lewm.gif
├── checkpoints
│   └── lewm
│       ├── weights_epoch_10.pt
│       ├── weights_epoch_11.pt
│       ├── weights_epoch_12.pt
│       ├── weights_epoch_13.pt
│       ├── weights_epoch_14.pt
│       ├── weights_epoch_15.pt
│       ├── weights_epoch_16.pt
│       ├── weights_epoch_17.pt
│       ├── weights_epoch_18.pt
│       ├── weights_epoch_19.pt
│       ├── weights_epoch_1.pt
│       ├── weights_epoch_20.pt
│       ├── weights_epoch_21.pt
│       ├── weights_epoch_22.pt
│       ├── weights_epoch_23.pt
│       ├── weights_epoch_24.pt
│       ├── weights_epoch_25.pt
│       ├── weights_epoch_26.pt
│       ├── weights_epoch_27.pt
│       ├── weights_epoch_28.pt
│       ├── weights_epoch_29.pt
│       ├── weights_epoch_2.pt
│       ├── weights_epoch_30.pt
│       ├── weights_epoch_31.pt
│       ├── weights_epoch_32.pt
│       ├── weights_epoch_33.pt
│       ├── weights_epoch_34.pt
│       ├── weights_epoch_35.pt
│       ├── weights_epoch_36.pt
│       ├── weights_epoch_37.pt
│       ├── weights_epoch_38.pt
│       ├── weights_epoch_39.pt
│       ├── weights_epoch_3.pt
│       ├── weights_epoch_40.pt
│       ├── weights_epoch_41.pt
│       ├── weights_epoch_42.pt
│       ├── weights_epoch_4.pt
│       ├── weights_epoch_5.pt
│       ├── weights_epoch_6.pt
│       ├── weights_epoch_7.pt
│       ├── weights_epoch_8.pt
│       └── weights_epoch_9.pt
├── conda_env
│   ├── 12-12-23
│   │   └── train.log
│   ├── environment.json
│   └── requirements_frozen.txt
├── config
│   ├── eval
│   │   ├── cube.yaml
│   │   ├── launcher
│   │   │   └── local.yaml
│   │   ├── pusht.yaml
│   │   ├── reacher.yaml
│   │   ├── solver
│   │   │   ├── adam.yaml
│   │   │   └── cem.yaml
│   │   └── tworoom.yaml
│   └── train
│       ├── data
│       │   ├── dmc.yaml
│       │   ├── ogb.yaml
│       │   ├── pusht.yaml
│       │   └── tworoom.yaml
│       ├── launcher
│       │   └── local.yaml
│       ├── lewm.yaml
│       └── model
│           └── lewm.yaml
├── data
│   ├── checkpoints
│   │   └── config.yaml
│   ├── hf_pusht
│   │   ├── config.json
│   │   └── weights.pt
│   ├── hf_tworoom
│   │   ├── config.json
│   │   └── weights.pt
│   ├── pusht
│   ├── pusht_expert_train.h5.zst
│   ├── tworoom
│   │   ├── lewm_object.ckpt
│   │   ├── rollout_0.mp4
│   │   ├── rollout_1.mp4
│   │   └── tworoom_results.txt
│   ├── tworoom.h5
│   └── tworoom.tar.zst
├── docs
│   ├── 复现步骤.md
│   └── 复现计划.md
├── eval.py
├── jepa.py
├── lewm
│   └── d2az0cg8
│       └── checkpoints
│           └── epoch=41-step=53928.ckpt
├── LICENSE
├── module.py
├── __pycache__
│   ├── jepa.cpython-310.pyc
│   ├── module.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── README.md
├── slurm-183915.out
├── test
│   ├── convert_pretrained.py
│   └── validate_data.py
├── test_dataset_field.py
├── train.py
├── train_tworoom.sh
└── utils.py

25 directories, 90 files