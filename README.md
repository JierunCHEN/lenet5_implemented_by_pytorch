# LeNet-5

This implements a slightly modified LeNet-5 [LeCun et al., 1998a] and achieves an accuracy of ~99% on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Usage

Start the `visdom` server for visualization

```
$ python -m visdom.server
```

Start the training procedure on CPU

```
$ python run_lenet5_cpu.py
```

Start the training procedure on GPU

```
$ python run_lenet5_gpu.py
```

See epoch train loss live graph at [`http://localhost:8097`](http://localhost:8097).

## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
