from . import measure
from ..p_utils import get_layer_metric_array

@measure('matrix_inf_norm', copy_net=False, mode='param')
def get_matrix_inf_norm_array(net, inputs, targets, mode, split_data=1, loss_fn=None):
    def get_matrix_inf_norm(layer):
        weight = layer.weight.abs()
        if len(weight.shape)==4:
            sum_w = weight.sum(dim=(0, 2, 3))
        elif len(weight.shape)==2:
            sum_w = weight.sum(dim=0)
        else:
            raise NotImplementedError(f'layer {l} has weight shape {weight.shape}')
        inf_norm = sum_w.max()
        if layer.bias is not None:
            bias = layer.bias.abs()
            inf_norm = max(inf_norm, bias.sum())
        return inf_norm
    return get_layer_metric_array(net, get_matrix_inf_norm, mode=mode)