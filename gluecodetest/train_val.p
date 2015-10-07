name: "AlexNet"
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  top: "label"
  include {
	phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 211
  }
  data_param {
    batch_size: 256
    num_labels: 1000
  }
}
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  top: "label"
  include {
	phase: TEST
  }
  transform_param {
    mirror: true
    crop_size: 211
  }
  data_param {
    batch_size: 16
    num_labels: 1000
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
