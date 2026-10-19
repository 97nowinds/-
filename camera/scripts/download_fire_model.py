"""Prepare a pinned checkpoint using restricted deserialization, then export ONNX."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]

ALLOWED_CLASSES={
    'torch.nn.modules.container.Sequential','ultralytics.nn.modules.block.SPPF','ultralytics.nn.modules.conv.Concat',
    'ultralytics.nn.modules.block.Bottleneck','ultralytics.nn.tasks.DetectionModel','torch.nn.modules.container.ModuleList',
    'torch.nn.modules.batchnorm.BatchNorm2d','torch.nn.modules.pooling.MaxPool2d','ultralytics.nn.modules.block.DFL',
    'torch.nn.modules.conv.Conv2d','torch.nn.modules.activation.SiLU','ultralytics.nn.modules.head.Detect',
    'ultralytics.nn.modules.conv.Conv','torch.nn.modules.upsampling.Upsample','torch.nn.modules.linear.Identity',
    'ultralytics.nn.modules.block.C2f'}


def export_checkpoint(source,target,config):
    import importlib
    import torch
    import onnx
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(source))
    if unsafe-ALLOWED_CLASSES:raise ValueError('unexpected checkpoint class; refusing deserialization')
    objects=[getattr(importlib.import_module(s.rsplit('.',1)[0]),s.rsplit('.',1)[1]) for s in sorted(ALLOWED_CLASSES)]
    with torch.serialization.safe_globals(objects):checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    model=(checkpoint.get('ema') or checkpoint['model']).float().eval()
    torch.set_num_threads(config['cpu_threads'])
    for module in model.modules():
        if module.__class__.__name__=='Detect':module.export=True;module.format='onnx';module.dynamic=False
    size=config['export_size']
    torch.onnx.export(model,torch.zeros(1,3,size,size),str(target),input_names=['images'],output_names=['output0'],opset_version=17,dynamo=False)
    proto=onnx.load(str(target));onnx.helper.set_model_props(proto,{'names':repr(model.names),'task':'detect',
        'license':'AGPL-3.0','source_repository':config['repository']})
    onnx.checker.check_model(proto);onnx.save(proto,str(target))


def main():
    config=json.loads((ROOT/'config/fire_model.json').read_text(encoding='utf-8'))
    path=ROOT/config['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==config['sha256']:
        print('Pinned fire model already verified.');return
    with urlopen(config['source_url'],timeout=45) as response:data=response.read(32*1024*1024+1)
    if len(data)>32*1024*1024 or hashlib.sha256(data).hexdigest()!=config['source_sha256']:
        raise ValueError('fire model artifact checksum mismatch')
    checkpoint=path.with_suffix('.checkpoint');checkpoint.write_bytes(data)
    temporary=path.with_name(path.stem+'.export.onnx')
    try:
        export_checkpoint(checkpoint,temporary,config)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=config['sha256']:
            raise ValueError('export differs from pinned artifact; check installed library versions')
        temporary.replace(path)
    finally:
        checkpoint.unlink(missing_ok=True);temporary.unlink(missing_ok=True)
    print('Pinned fire checkpoint safely exported and ONNX verified.')


if __name__=='__main__':main()
