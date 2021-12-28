# reference https://github.com/researchmm/soho/blob/d98b2ba52ffda2ba857aa4bc0d4e9239efcfd806/SOHO/models/necks/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new, 1.0 - decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())


def sum_inplace(sum_data, new):
    sum_data.data.add_(new)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def laplace_smoothing_dim(x, n_categories, dim=1, eps=1e-5):
    return (x + eps) / (x.sum(dim=dim, keepdim=True) + n_categories * eps)


class VectorQuantizer(nn.Module):
    """
    Inputs:
    - num_tokens : number of embeddings
    - token_dim : dimension of embedding
    - dist : distribution training or not
    """
    def __init__(self,
                 num_tokens,
                 token_dim,
                 decay=0.1,
                 max_decay=0.99,
                 eps=1e-5,
                 dist=False):
        super(VectorQuantizer, self).__init__()
        self.dist = dist
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        embed = torch.randn(num_tokens, token_dim)
        self.register_buffer('embed', embed)
        nn.init.normal_(self.embed)
        self.register_buffer('cluster_size', torch.zeros(num_tokens))
        self.register_buffer('cluster_sum', torch.zeros(num_tokens))
        self.register_buffer('embed_avg', torch.zeros(num_tokens, token_dim))

        self.decay = decay
        self.eps = eps
        self.curr_decay = self.decay
        self.max_decay = max_decay

        print("Performing VectorQuantizer to cluster %d tokens ...." %(num_tokens))

    def set_decay_updates(self, num_update):
        self.curr_decay = min(self.decay * num_update, self.max_decay)

    def forward(self, inputs_flatten):

        distances = (torch.sum(inputs_flatten**2, dim=1, keepdim=True) +
                     torch.sum(self.embed.data**2, dim=1) -
                     2 * torch.matmul(inputs_flatten, self.embed.data.t()))
        """
                encoding_indices: Tensor containing the discrete encoding indices, ie
                which element of the quantized space each input element was mapped to.
        """

        # print('distances:\t', distances.size())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print('encoding_indices:\t', encoding_indices.size())

        encodings = torch.zeros(encoding_indices.shape[0],
                                self.num_tokens,
                                dtype=torch.float,
                                device=inputs_flatten.device)
        # print('encodings:\t', encodings.size())

        encodings.scatter_(1, encoding_indices, 1)
        # print('encodings:\t', encodings.size())

        if self.training:

            tmp_sum = torch.sum(encodings, dim=0, keepdim=True)
            if self.dist:
                encoding_sum = torch.sum(concat_all_gather(tmp_sum), dim=0)
            else:
                encoding_sum = torch.sum(tmp_sum, dim=0)

            sum_inplace(self.cluster_sum, encoding_sum)
            ema_tensor_inplace(self.cluster_size, encoding_sum,
                               self.curr_decay)
            embed_sum_tmp = torch.matmul(encodings.t(), inputs_flatten)
            if self.dist:
                embed_sum = torch.sum(concat_all_gather(
                    embed_sum_tmp.unsqueeze(dim=0)),
                                      dim=0)
            else:
                embed_sum = torch.sum(embed_sum_tmp.unsqueeze(dim=0), dim=0)

            ema_tensor_inplace(self.embed_avg, embed_sum, self.curr_decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.num_tokens,
                self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

            if self.dist:
                world_size = dist.get_world_size()
                dist.all_reduce(embed_normalized.div_(world_size))
            self.embed.data.copy_(embed_normalized)

        quantize = torch.matmul(encodings, self.embed)
        # print('encodings:\t', encodings)
        #quantize = inputs_flatten
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten

        return quantize, encoding_indices
