import torch


def multi_head_single_quantize(inp: torch.Tensor, codebook: torch.Tensor):
    # inp: batch x head x seqlen x d_hidden
    # codebook: head x codebook_size x d_hidden
    
    bsz, head, seqlen, hsz = inp.shape
    
    product = torch.einsum('zhqx,hcx->hzqc', inp, codebook)
    argmax = product.argmax(-1)  # head x batch x seqlen
    
    quantized_inp = codebook.gather(1, argmax.flatten(1).unsqueeze(-1).expand(-1, -1, hsz))
    quantized_inp = quantized_inp.view(head, bsz, seqlen, hsz).transpose(0, 1)
    
    argmax = argmax.transpose(0, 1)
    onehot = torch.nn.functional.one_hot(argmax, codebook.shape[1])
    return quantized_inp, argmax, onehot