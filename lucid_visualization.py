import tensorflow as tf
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts
import lucid.optvis.render as render


save_dir = 'experiments/googlenet/pb/'
saved_frozen_googlenet_path = "{}/epoch-0_frozen.pb".format(save_dir)


def _populate_inception_bottlenecks(scope):
    """Add Inception bottlenecks and their pre-Relu versions to the graph."""
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        if op.name.startswith(scope+'/') and 'Concat' in op.type:
            name = op.name.split('/')[1]
            pre_relus = []
            for tower in op.inputs[:-1]:
                if tower.op.type == 'Relu':
                    tower = tower.op.inputs[0]
                pre_relus.append(tower)
            concat_name = scope + '/' + name + '_pre_relu'
            _ = tf.concat(pre_relus, -1, name=concat_name)


class InceptionV1(Model):
    """InceptionV1 (or 'GoogLeNet')
    This is a (re?)implementation of InceptionV1
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
    The weights were trained at Google and released in an early TensorFlow
    tutorial. It is possible the parameters are the original weights
    (trained in TensorFlow's predecessor), but we haven't been able to
    confirm this.
    As far as we can tell, it is exactly the same as the model described in
    the original paper, where as the slim and caffe implementations have
    minor implementation differences (such as eliding the heads).
    """
    model_path = saved_frozen_googlenet_path
    labels_path = 'gs://modelzoo/labels/ImageNet_alternate.txt'
    synsets_path = 'gs://modelzoo/labels/ImageNet_alternate_synsets.txt'
    dataset = 'ImageNet'
    image_shape = [224, 224, 3]
#     image_value_range = (-117, 255-117)
    input_name = 'input'

    def post_import(self, scope):
        _populate_inception_bottlenecks(scope)


InceptionV1.layers = _layers_from_list_of_dicts(InceptionV1(), [
  {'tags': ['conv'], 'name': 'conv1', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv3', 'depth': 192},
  {'tags': ['conv'], 'name': 'inception3a', 'depth': 256},
  {'tags': ['conv'], 'name': 'inception3b', 'depth': 480},
  {'tags': ['conv'], 'name': 'inception4a', 'depth': 508},
  {'tags': ['conv'], 'name': 'inception4b', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception4c', 'depth': 512},
  {'tags': ['conv'], 'name': 'inception4d', 'depth': 528},
  {'tags': ['conv'], 'name': 'inception4e', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception5a', 'depth': 832},
  {'tags': ['conv'], 'name': 'inception5b', 'depth': 1024},
  {'tags': ['conv'], 'name': 'head0_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn0', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax0', 'depth': 1008},
  {'tags': ['conv'], 'name': 'head1_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn1', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax1', 'depth': 1008},
  {'tags': ['dense'], 'name': 'softmax2', 'depth': 1008},
])


if __name__ == "__main__":
    inception_model = InceptionV1()
    inception_model.load_graphdef()

    _ = render.render_vis(inception_model, "inception4a_pre_relu:476")
