"""Site PPE transfer learning. GPU only; frozen MiniCPM vision encoder, trainable linear heads."""
from pathlib import Path
import hashlib,json,sqlite3,time,os
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6VisionModel
ROOT=Path('/cluster/home/xqyi/lab-safety-host/camera');OUT=ROOT/'runtime/ppe-training-20261104';OUT.mkdir(exist_ok=True)
torch.manual_seed(41);np.random.seed(41);torch.set_num_threads(2)
assert torch.cuda.is_available();device='cuda:0'
model_path=Path(json.loads((ROOT/'config/ppe_observer.json').read_text())['coat_model']);config=AutoConfig.from_pretrained(model_path,local_files_only=True).vision_config;config._attn_implementation='sdpa'
vision=MiniCPMV4_6VisionModel(config)
with safe_open(model_path/'model.safetensors',framework='pt') as f:
 weights={k.removeprefix('model.vision_tower.'):f.get_tensor(k) for k in f.keys() if k.startswith('model.vision_tower.')}
vision.load_state_dict(weights,strict=True);del weights;vision=vision.to(device,dtype=torch.bfloat16).eval();vision.requires_grad_(False)
print('VISION_READY',sum(p.numel() for p in vision.parameters()),flush=True)
c=sqlite3.connect('file:'+str(ROOT/'data/ppe_samples/samples.sqlite3')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;rows=[]
for r in c.execute('SELECT s.*,l.item,l.state,l.visibility,l.annotation_source FROM samples s JOIN labels l ON s.sample_id=l.sample_id'):
 d=dict(r);prov=json.loads(d['provenance_json']);group=prov.get('source_group',prov.get('scenario',''));d['group']=group
 if d['state']=='unknown' or d['visibility']=='not_visible_back_view':continue
 if d['item']=='mask':
  if group in ['transition','person8_approved_mask_reference']:d['split']='train'
  elif group in ['testclothes','questionnaire_06_ultrasonic_cleaner']:d['split']='test'
  else:continue
 else:
  d['split']='train' if group in ['transition','testclothes'] else 'test'
 d['image']=str(ROOT/'data/ppe_samples'/d['head_region_path' if d['item']=='mask' else 'person_crop_path']);rows.append(d)
# Keep all newly recorded centrifuge negatives outside training.
cent=ROOT/'runtime/questionnaire-20261104/centrifuge/captures'
for p in cent.glob('*/request.json'):
 d=json.loads(p.read_text());at=d['captured_at']
 if at[11:23] not in ['05:32:14.858','05:32:21.381','05:33:13.860','05:33:15.864','05:33:18.382']:continue
 t=d['camera']['tracks'][0];x,y,w,h=t['bbox'];im=Image.open(p.parent/'original.jpg').crop((int(x),int(y),int(x+w),int(y+h)));target=OUT/(p.parent.name+'_cent_negative.jpg');im.save(target,quality=95)
 rows.append({'sample_id':'centrifuge:'+p.parent.name,'item':'lab_coat','state':'not_worn','split':'test','group':'centrifuge_heldout','image':str(target),'source_frame_id':d['snapshot_id']})
assert set(r['sample_id'] for r in rows if r['item']=='mask' and r['split']=='train').isdisjoint(r['sample_id'] for r in rows if r['item']=='mask' and r['split']=='test')
def tensor_image(im):
 im=ImageOps.pad(im.convert('RGB'),(224,224),color=(127,127,127),method=Image.Resampling.BICUBIC);arr=np.asarray(im).astype(np.float32)/127.5-1;return torch.from_numpy(arr.transpose(2,0,1).copy())[None].to(device,dtype=torch.bfloat16)
def feature(im):
 with torch.inference_mode():
  z=vision(tensor_image(im),target_sizes=torch.tensor([[16,16]],device=device,dtype=torch.int32),use_vit_merger=False).last_hidden_state.float().mean(dim=1)
  return torch.nn.functional.normalize(z,dim=-1)[0].cpu()
started=time.time();features=[]
for i,r in enumerate(rows):
 im=Image.open(r['image']);augmentations=[im]
 if r['split']=='train':augmentations += [ImageOps.mirror(im),ImageEnhance.Brightness(im).enhance(.85),ImageEnhance.Brightness(im).enhance(1.15)]
 fs=torch.stack([feature(im) for im in augmentations]);features.append(fs)
 if i%10==0:print('FEATURES',i,len(rows),round(time.time()-started,1),flush=True)
torch.save({'rows':rows,'features':features},OUT/'frozen_features.pt');results={};heads={}
for item in ['mask','lab_coat']:
 train=[i for i,r in enumerate(rows) if r['item']==item and r['split']=='train'];test=[i for i,r in enumerate(rows) if r['item']==item and r['split']=='test'];assert train and test
 x=torch.cat([features[i] for i in train]).to(device);y=torch.tensor([int(rows[i]['state']=='worn') for i in train for _ in features[i]],device=device,dtype=torch.float32);head=torch.nn.Linear(x.shape[1],1).to(device);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.1);pos=(len(y)-y.sum())/y.sum()
 losses=[]
 for epoch in range(600):
  opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x).flatten(),y,pos_weight=pos);loss.backward();opt.step()
  if epoch%100==0:losses.append({'epoch':epoch,'loss':float(loss.detach())})
 head.eval();evals={}
 for split,indices in [('train',train),('test',test)]:
  z=torch.stack([features[i][0] for i in indices]).to(device)
  with torch.no_grad():ps=torch.sigmoid(head(z).flatten()).cpu().tolist();similarities=(z@x.T).max(dim=1).values.cpu().tolist()
  preds=[]
  for i,p,sim in zip(indices,ps,similarities):
   expected=rows[i]['state'];binary='worn' if p>=.5 else 'not_worn';state=binary if max(p,1-p)>=.8 and sim>=.65 else 'unknown';preds.append({'sample_id':rows[i]['sample_id'],'group':rows[i]['group'],'truth':expected,'binary':binary,'state':state,'worn_score':p,'nearest_training_similarity':sim})
  n=len(preds);correct=sum(p['state']==p['truth'] for p in preds);unknown=sum(p['state']=='unknown' for p in preds);evals[split]={'n':n,'correct':correct,'wrong':n-correct-unknown,'unknown':unknown,'correct_over_all':correct/n,'coverage':(n-unknown)/n,'binary_correct':sum(p['binary']==p['truth'] for p in preds),'per_class':{s:{'n':sum(p['truth']==s for p in preds),'correct':sum(p['truth']==s and p['state']==s for p in preds)} for s in ['worn','not_worn']},'predictions':preds}
 results[item]={'training_rows':len(train),'augmented_training_rows':len(x),'trainable_parameters':sum(p.numel() for p in head.parameters()),'loss_trace':losses,**evals};heads[item]={'weight':head.weight.detach().cpu(),'bias':head.bias.detach().cpu(),'training_features':x.detach().cpu()}
checkpoint={'schema':'site_ppe_linear_v1','heads':heads,'vision_model':str(model_path),'input_size':224,'feature_pool':'mean_last_hidden_l2','normalization':'rgb/127.5-1','threshold':.8,'minimum_similarity':.65,'seed':41,'vision_frozen':True};torch.save(checkpoint,OUT/'site_ppe_heads.pt')
report={'method':'frozen MiniCPM-V-4.6 vision encoder with two supervised linear classifier heads','device':torch.cuda.get_device_name(),'seed':41,'epochs':600,'test_used_for_tuning':False,'split_basis':'scenario and recording session, not random adjacent frames','training_seconds':round(time.time()-started,3),'checkpoint':str(OUT/'site_ppe_heads.pt'),'results':results,'limitations':['Very small, correlated single-site sample set.','This is classifier-head training, not full MiniCPM language-model fine-tuning.','The held-out set is separate by scenario but not a large independent participant study.','Unknown visual states excluded from binary training and headline accuracy.','No ordinary white non-lab garment negatives are available.']};(OUT/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({item:{s:results[item][s] for s in ['training_rows','trainable_parameters']}|{'test':{k:v for k,v in results[item]['test'].items() if k!='predictions'}} for item in results}),flush=True)
