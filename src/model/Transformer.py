from turtle import forward
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.module import _forward_unimplemented

"""
Transformer for utilizing DMRS to estimate the channel information.

Input SIZE:
    [batch, time, freq(real/imag)]: [*, 2, 2*48]

Output SIZE:
    [batch, time, freq(real/imag)]: [*, 14, 2*96]
"""

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class GELU(nn.Module):
    """
    Activation Function.
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/np.pi))*(x+0.044715*torch.pow(x,3))))


class MultiHead(nn.Module):
    """
    This is the multi-head attention block aiming to obtain the self-correlation.

    Args(init):
        h: the number of the heads
        modelDim: the dimension of the model which is set manually

    Args(forward):
        V: value matrix
        K: key matrix
        Q: query matrix

    Return:
        attention: a probability matrix
    """
    def __init__(self, modelDim, h=4):
        super(MultiHead, self).__init__()
        self.h = h
        headDim = modelDim//h
        self.scale_factor = (headDim)**-.5
        self.head_V = nn.ModuleList([nn.Linear(modelDim, headDim) for _ in range(h)])
        self.head_K = nn.ModuleList([nn.Linear(modelDim, headDim) for _ in range(h)])
        self.head_Q = nn.ModuleList([nn.Linear(modelDim, headDim) for _ in range(h)])
        map(initialize_weight, self.head_V)
        map(initialize_weight, self.head_K)
        map(initialize_weight, self.head_Q)
        self.W_o = nn.Linear(modelDim, modelDim)

    def forward(self, V, K, Q, cache=None):
        if cache is not None and 'encdec_k' in cache:
            V, K = cache['encdec_v'], cache['encdec_k']
        attention = self._multiheadAttention(V, K, Q, self.head_V, self.head_K, self.head_Q,
                                             self.h, self.scale_factor, self.W_o)
        return attention

    def _attention(self, v: torch.tensor, k: torch.tensor, q: torch.tensor, scale_factor: float):
        """
        Calculate the scaled dot-product attention.

        Args:
            v: projected value matrix
            k: projected key matrix
            q: projected query matrix

        Return:
            attention: scaled dot-product attention
        """
        dotProduct = F.softmax(torch.matmul(q, k.transpose(-1,-2))*scale_factor, dim=-1)
        attention = torch.matmul(dotProduct, v)

        return attention

    def _multiheadAttention(self, V, K, Q, head_V, head_K, head_Q, h, scale_factor, W_o):
        """
        Calculate the multi-head self attention.

        Args:
            V: value matrix
            K: key matrix
            Q: query matrix
            head_V: value matrix projector
            head_K: key matrix projector
            head_Q: query matrix projector

        Return:
            multiAttention: the multi-head self attention
        """
        # attList = [self._attention(self.head_V[i](V), self.head_K[i](K), self.head_Q[i](Q), self.scale_factor) 
        #         for i in range(h)]
        # multiAttention = torch.tensor([])
        multiAttention = self._attention(head_V[0](V), head_K[0](K), head_Q[0](Q), scale_factor)
        for i in range(h-1):
            att = self._attention(head_V[i+1](V), head_K[i+1](K), head_Q[i+1](Q), scale_factor)
            multiAttention = torch.cat([multiAttention, att], dim=-1)
        multiAttention = W_o(multiAttention)
        return multiAttention


class FeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout=0.2):
        super(FeedForward, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, filter_size),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_size, hidden_size)
        )
        initialize_weight(self.feedforward[0])
        initialize_weight(self.feedforward[-1])

    def forward(self, x):
        return self.feedforward(x)


class EncoderLayer(nn.Module):
    """
    Performing as the encoder layer, where we get rid of the masking stage because we don't need care about the actual order.

    Args(init):
        hidden_size: the dimension of the input feature. (Here we don't need to rearrange the matrix.)
        filter_size: the size of the linear layer.
        dropout: the dropout rate, for better generalization.

    Args(forward):
        x: the pending data.

    Return:
        x: final result after the res block.
    """
    def __init__(self, hidden_size, filter_size, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-7)
        self.attention = MultiHead(hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-7)
        self.ffn = FeedForward(hidden_size, filter_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        ###################
        # Attention Block #
        ###################
        y = self.attention_norm(x)
        y = self.attention(y,y,y)
        y = self.attention_dropout(y)
        x = x + y

        #####################
        # FeedForward Block #
        #####################
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout=0.2, n_layers=1):
        super(Encoder, self).__init__()
        encoders = [EncoderLayer(hidden_size, filter_size, dropout) for _ in range(n_layers)]
        self.encoders = nn.ModuleList(encoders)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = x
        for encoder in self.encoders:
            y = encoder(y)

        return y


class DecoderLayer(nn.Module):
    """
    Performing as the decoder layer, where we get rid of the masking stage because we don't need care about the actual order.

    Args(init):
        hidden_size: the dimension of the input feature. (Here we don't need to rearrange the matrix.)
        filter_size: the size of the linear layer.
        dropout: the dropout rate, for better generalization.

    Args(forward):
        x: the pending data
        enc_output: the output of the encoder 
        cache: 

    """
    def __init__(self,  hidden_size, filter_size, dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHead(hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHead(hidden_size)
        self.enc_dec_attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForward(hidden_size, filter_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, cache):
        ###############
        # Add & Norm 1#
        ###############
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y)
        y = self.self_attention_dropout(y)
        x = x + y

        ###############
        # Add & Norm 2#
        ###############
        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(enc_output, enc_output, y, cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        ################
        # Feed Forward #
        ################
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout) for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output, layer_cache)
        return self.last_norm(decoder_output)


class Transformer(nn.Module):
    def __init__(self, f_in_size, f_out_size, 
                up_size=(2*96, 14*192),
                n_layers=2,
                hidden_size=256,
                filter_size=512,
                dropout=0.1,
                has_input=True
                ):
        super(Transformer, self).__init__()
        self.f_in_size = f_in_size
        self.f_out_size = f_out_size
        self.up_size = up_size
        self.has_input = has_input

        # Linear layer to upsample the DMRS signals
        self.upsample = nn.Linear(up_size[0], up_size[1])

        # Linear layer for input to encoder
        self.input_layer = nn.Linear(f_in_size, hidden_size)

        self.decoder = Decoder(hidden_size, filter_size, dropout, n_layers)
        if  has_input:
            self.encoder = Encoder(hidden_size, filter_size, dropout, n_layers)

        # Linear layer for input to decoder
        self.target_input_layer = nn.Linear(f_out_size, hidden_size)

        # final linear layer
        self.output_layer = nn.Linear(hidden_size, f_out_size)

    def forward(self, inputs, snr):
        x = inputs.reshape(-1, 1, 2*96)
        inputs_upsampled = self.upsample(x).reshape(-1, 1, 14, int(self.up_size[1]/14))

        enc_input = self.input_layer(inputs_upsampled)
        enc_output = None
        if self.has_input:
            enc_output = self._encode(enc_input)
        return self._decode(inputs_upsampled, enc_output)
        
    
    def _encode(self, inputs):
        return self.encoder(inputs)

    def _decode(self, targets, enc_output, cache=None):
        decoder_input = self.target_input_layer(targets)
        decoder_output = self.decoder(decoder_input, enc_output, cache)
        return self.output_layer(decoder_output)


if __name__ == "__main__":
    layernorm = nn.LayerNorm(96, eps=1e-6)
    n = torch.rand([32,2,96])
    out = layernorm(n)
    print(out.shape)

    Attention = MultiHead(96, 2)
    encoder = Encoder(192, 384)
    transformer = Transformer(96, 192, 3)
    inputs = torch.rand((32, 14, 96))
    targets = torch.rand((32, 14, 192))


    v = torch.rand((32, 14, 96))
    k = torch.rand((32, 14, 96))
     
    q = torch.rand((32, 14, 96))

    # att = Attention(v, k, q)
    # enc_output = encoder(v)
    out = transformer(inputs, targets)
    # att = Attention._attention(v, k, q, scale_factor=1)
    # print(att.shape)
    # print(enc_output.shape)
    print(out.shape)
    
    

