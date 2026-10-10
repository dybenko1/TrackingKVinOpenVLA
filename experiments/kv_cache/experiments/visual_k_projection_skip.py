"""True visual-K projection skipping for selected multimodal reuse prefills."""
from __future__ import annotations
import types
from typing import Any, Dict, Sequence
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from experiments.kv_cache.experiments.static_reuse_experiment import cache_layers

class DuringPrefillVisualKProjectionSkip:
 def __init__(self,model,source,layout,reuse_layers):
  self.model,self.source,self.layout,self.layers=model,source,layout,set(reuse_layers);self.length=int(layout['prefill_length']);self.vs,self.ve=layout['visual'];self.ls,self.le=layout['language'];self.original={};self.diagnostics={'layers':[],'q_rows':[],'v_rows':[],'k_rows':[],'decode_k_rows':[]}
 def __enter__(self):
  assert (self.vs,self.ve)==(1,256) and self.ls==257
  src=cache_layers(self.source)
  for i in self.layers:
   a=self.model.language_model.model.layers[i].self_attn;self.original[i]=a.forward
   def forward(m,hidden_states,position_embeddings,attention_mask,past_key_value=None,cache_position=None,**kwargs):
    if hidden_states.shape[-2]!=self.length or past_key_value is None:
     if hidden_states.shape[-2]==1:self.diagnostics['decode_k_rows'].append({'layer':m.layer_idx,'rows':1})
     return self.original[m.layer_idx](hidden_states,position_embeddings,attention_mask,past_key_value,cache_position,**kwargs)
    B,S,_=hidden_states.shape; shape=(B,S,-1,m.head_dim); idx=torch.cat((torch.zeros(1,dtype=torch.long,device=hidden_states.device),torch.arange(self.ls,S,device=hidden_states.device)))
    q=m.q_proj(hidden_states).view(shape).transpose(1,2);v=m.v_proj(hidden_states).view(shape).transpose(1,2); fresh=m.k_proj(hidden_states.index_select(1,idx)).view(B,idx.numel(),-1,m.head_dim).transpose(1,2)
    self.diagnostics['q_rows'].append({'layer':m.layer_idx,'rows':S});self.diagnostics['v_rows'].append({'layer':m.layer_idx,'rows':S});self.diagnostics['k_rows'].append({'layer':m.layer_idx,'rows':int(idx.numel())})
    cos,sin=position_embeddings;q,_=apply_rotary_pos_emb(q,q,cos,sin); _,fresh=apply_rotary_pos_emb(fresh,fresh,cos.index_select(1,idx),sin.index_select(1,idx))
    prior,_=src[m.layer_idx]; k=torch.cat((fresh[:,:,:1,:],prior[:,:,self.vs:self.ve+1,:],fresh[:,:,1:,:]),dim=2);assert k.shape[-2]==S
    k,v=past_key_value.update(k,v,m.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position});fn=eager_attention_forward if m.config._attn_implementation=='eager' else ALL_ATTENTION_FUNCTIONS[m.config._attn_implementation];o,w=fn(m,q,k,v,attention_mask,dropout=0.0 if not m.training else m.attention_dropout,scaling=m.scaling,**kwargs);self.diagnostics['layers'].append(m.layer_idx);return m.o_proj(o.reshape(B,S,-1).contiguous()),w
   a.forward=types.MethodType(forward,a)
  return self
 def __exit__(self,*_):
  for i,f in self.original.items():self.model.language_model.model.layers[i].self_attn.forward=f
 def validate(self,cache):
  assert sorted(self.diagnostics['layers'])==sorted(self.layers)
  assert all(x['rows']==1+(self.le-self.ls+1) for x in self.diagnostics['k_rows'])
  assert all(x['rows']==self.length for x in self.diagnostics['q_rows']+self.diagnostics['v_rows'])
  for i in self.layers:assert torch.equal(cache_layers(cache)[i][0][:,:,self.vs:self.ve+1,:],cache_layers(self.source)[i][0][:,:,self.vs:self.ve+1,:])
