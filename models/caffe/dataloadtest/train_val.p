name: "DataLoadTest"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 211
    mean_file: "$DATA_DIR/aux/imagenet_mean.binaryproto"
  }
  data_param {
    source: "$DATA_DIR/dbs/ilsvrc12_train_lmdb"
    batch_size: 128
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 211
    mean_file: "$DATA_DIR/aux/imagenet_mean.binaryproto"
  }
  data_param {
    source: "$DATA_DIR/dbs/ilsvrc12_val_lmdb"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "data"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 1
    stride: 70
  }
}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "pool1"
  top: "fc8"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc8"
  bottom: "label"
  top: "loss"
}
