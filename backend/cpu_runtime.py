"""Experimental CPU port, kept separate from the frozen CUDA release.
Apache-2.0 new wrapper; upstream model/code attribution stays with release.
"""
import json, math, time
from pathlib import Path

class CpuCadence:
    def __init__(self,package,*,profile='fp32',threads=4):
        if profile not in ('fp32','merged_fp32','merged_dynamic_int8'):raise ValueError('Unknown CPU profile')
        import torch
        from safetensors.torch import load_file
        from peft import LoraConfig,get_peft_model
        from huggingface_hub import snapshot_download
        from cadence_runtime import digest
        from cadence_kev.checkpoint import Checkpoint,LoadOptions
        self.path=Path(package);self.config=json.loads((self.path/'cadence_config.json').read_text());cfg=self.config
        assert cfg['format']=='CADENCE_QUALIFIED2357_ADAPTER_HEAD_v1' and cfg['step']==2357
        for name,want in cfg['weights_sha256'].items():assert digest(self.path/name)==want,name
        for name,want in cfg['upstream_code']['files'].items():assert digest(self.path/'cadence_kev'/name)==want,name
        torch.set_num_threads(threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        dep=cfg['released_kev'];upstream=Path(snapshot_download(dep['repo'],revision=dep['revision'],allow_patterns=list(dep['required_files']),cache_dir=str(self.path.parent/'public_cache') if (self.path.parent/'public_cache').exists() else None,local_files_only=True,token=False))
        for name,want in dep['required_files'].items():assert digest(upstream/name)==want,name
        cp=Checkpoint(upstream);assert cp.meta.base==cfg['base_model']['repo'] and cp.meta.base_revision==cfg['base_model']['revision']
        # Reconstruct the released BF16-rounded merged Kev base first, then cast
        # that fixed geometry to FP32 for CPU. This is a new precision profile.
        self.tokenizer,self.model=cp.load('cpu',LoadOptions(dtype=torch.bfloat16,merge=True,attn='sdpa',cuda_graphs=False,temperature=1.0))
        self.model.requires_grad_(False)
        a=cfg['adaptation'];self.model.lm=get_peft_model(self.model.lm,LoraConfig(task_type='FEATURE_EXTRACTION',r=a['rank'],lora_alpha=a['alpha'],lora_dropout=a['dropout'],target_modules=a['target_modules']))
        self.model.head.float();self.model.lm.config.use_cache=False
        saved={**load_file(str(self.path/'adapter.safetensors'),device='cpu'),**load_file(str(self.path/'head.safetensors'),device='cpu')}
        named=dict(self.model.named_parameters());expected={n for n in named if 'lora_' in n or n.startswith('head.')}
        assert set(saved)==expected and len(saved)==376
        with torch.no_grad():
            for name,value in saved.items():
                assert value.dtype==named[name].dtype==torch.float32 and value.shape==named[name].shape and torch.isfinite(value).all()
                named[name].copy_(value);assert torch.equal(named[name],value)
        assert sum(x.numel() for x in self.model.parameters())==cfg['model_parameters']
        self.model.float().eval();self.model.head.temperature=1.;self.model.requires_grad_(False)
        self.profile=profile;self.quantized_linear_count=0
        if profile!='fp32':self.model.lm=self.model.lm.merge_and_unload(safe_merge=True)
        if profile=='merged_dynamic_int8':
            torch.backends.quantized.engine='x86'
            self.model.lm=torch.ao.quantization.quantize_dynamic(self.model.lm,{torch.nn.Linear},dtype=torch.qint8,inplace=True)
            self.quantized_linear_count=sum(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in self.model.lm.modules())
            assert self.quantized_linear_count>0
        self.model.eval();self.model.requires_grad_(False)

    def predict(self,request):
        import torch
        from cadence_kev.api import SystemOneRequest,to_record
        from cadence_runtime import softmax
        if not isinstance(request,dict) or not isinstance(request.get('questions'),dict) or len(request['questions'])!=1:raise ValueError('Provide exactly one typed question')
        clean={'state':request['state'],'questions':{qid:{k:v for k,v in q.items() if k in ('type','instructions','criteria')} for qid,q in request['questions'].items()}}
        rec,meta=to_record(SystemOneRequest.model_validate(clean))
        for q,m in zip(rec['questions'],meta):q.update(src='inference',qtype=m['type'],qid=m['id'],keys=m['keys'])
        ctx=self.config['inference']['context'];began=time.perf_counter()
        enc=self.model.encode(self.tokenizer,rec,max_state=ctx['max_state'],max_branch=ctx['max_branch'],strict=True)
        if len(enc['ids'])>ctx['max_packed']:raise ValueError('Packed input exceeds frozen context')
        with torch.inference_mode():z=self.model.forward(enc)[0].float().detach().cpu().tolist()
        ids=meta[0]['keys'];p=softmax(z);assert len(ids)==len(z)
        out=dict(question_id=meta[0]['id'],kind=meta[0]['type'],native_option_ids=ids,logits=dict(zip(ids,z)),probabilities=dict(zip(ids,p)),temperature=1.,tokens=len(enc['ids']),truncated=False,seconds=time.perf_counter()-began,cpu_profile=self.profile)
        if out['kind']=='noul':out['value']=out['probabilities']['true']
        if out['kind']=='score':out['value']=sum(float(k)*v for k,v in out['probabilities'].items())
        return out
