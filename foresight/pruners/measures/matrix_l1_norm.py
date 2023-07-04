from . import measure
from ..p_utils import get_layer_metric_array

@measure('matrix_l1_norm', copy_net=False, mode='param')
def get_matrix_l1_norm_array(net, inputs, targets, mode, split_data=1, loss_fn=None):
    def get_matrix_l1_norm(layer):
        weight = layer.weight.abs()
        if len(weight.shape)==4:
            sum_w = weight.sum(dim=(1, 2, 3))
        elif len(weight.shape)==2:
            sum_w = weight.sum(dim=1)
        else:
            raise NotImplementedError(f'layer {l} has weight shape {weight.shape}')
        return sum_w.max()
        
    return get_layer_metric_array(net, get_matrix_l1_norm, mode=mode)