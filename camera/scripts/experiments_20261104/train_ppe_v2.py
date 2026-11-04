from pathlib import Path
import torch,json,time,collections,sys
import numpy as np
R=Path('/cluster/home/xqyi/lab-safety-host/camera');out=R/'runtime/ppe-training-20261104';torch.set_num_threads(2);torch.manual_seed(41);dev='cuda:0';cache=torch.load(out/'frozen_features.pt',weights_only=False);rows=cache['rows'];features=cache['features'];cent=[i for i,r in enumerate(rows) if r['group']=='centrifuge_heldout'];cent.sort(key=lambda i:rows[i]['sample_id'])
# Adapt only with the first visit at 13:32. Keep the return visit at 13:33 out of training.
for i in cent:
 name=Path(rows[i]['image']).name.split('_cent_negative')[0];request=json.loads((R/'runtime/questionnaire-20261104/centrifuge/captures'/name/'request.json').read_text())
 if request['captured_at']<'2026-11-04T05:33:00':rows[i]['split']='train';rows[i]['group']='centrifuge_first_visit_adaptation'
heads={};results={}
for item in ['mask','lab_coat']:
 train=[i for i,r in enumerate(rows) if r['item']==item and r['split']=='train'];test=[i for i,r in enumerate(rows) if r['item']==item and r['split']=='test'];x=torch.cat([features[i] for i in train]).to(dev);y=torch.tensor([int(rows[i]['state']=='worn') for i in train for _ in features[i]],device=dev,dtype=torch.float32);mean=x.mean(0);scale=x.std(0).clamp(min=.0005);x=(x-mean)/scale;head=torch.nn.Linear(x.shape[1],1).to(dev);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
 for epoch in range(300):
  opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x).flatten(),y,pos_weight=(len(y)-y.sum())/y.sum());loss.backward();opt.step()
 evaluations={}
 for split,indices in [('train',train),('development_holdout',test)]:
  z=(torch.stack([features[i][0] for i in indices]).to(dev)-mean)/scale
  with torch.no_grad():probs=torch.sigmoid(head(z).flatten()).cpu().tolist()
  predictions=[]
  for i,p in zip(indices,probs):
   binary='worn' if p>=.5 else 'not_worn';state=binary if max(p,1-p)>=.8 else 'unknown';predictions.append({'sample_id':rows[i]['sample_id'],'image':rows[i]['image'],'group':rows[i]['group'],'truth':rows[i]['state'],'state':state,'binary':binary,'worn_score':p})
  n=len(predictions);correct=sum(p['state']==p['truth'] for p in predictions);unknown=sum(p['state']=='unknown' for p in predictions);evaluations[split]={'n':n,'correct':correct,'wrong':n-correct-unknown,'unknown':unknown,'correct_over_all':correct/n,'coverage':(n-unknown)/n,'binary_correct':sum(p['binary']==p['truth'] for p in predictions),'per_class':{s:{'n':sum(p['truth']==s for p in predictions),'correct':sum(p['truth']==s and p['state']==s for p in predictions)} for s in ['worn','not_worn']},'predictions':predictions}
 results[item]={'training_rows':len(train),'augmented_training_rows':len(x),'trainable_parameters':1153,'final_loss':float(loss.detach()),**evaluations};heads[item]={'weight':head.weight.detach().cpu(),'bias':head.bias.detach().cpu(),'mean':mean.cpu(),'scale':scale.cpu()}
torch.save({'schema':'site_ppe_linear_v2','heads':heads,'vision_model':str(json.loads((R/'config/ppe_observer.json').read_text())['coat_model']),'input_size':224,'normalization':'rgb/127.5-1_letterbox_gray','feature_pool':'mean_last_hidden_l2','threshold':.8,'vision_frozen':True,'seed':41},out/'site_ppe_heads_v2.pt')
report={'method':'frozen MiniCPM-V-4.6 vision encoder, standardized pooled features, two supervised linear heads','device':torch.cuda.get_device_name(),'seed':41,'epochs':300,'checkpoint':str(out/'site_ppe_heads_v2.pt'),'results':results,'split_basis':'different scenarios and visits; no random adjacent-frame split','validation_status':'development_holdout_not_final_independent_validation','limitations':['Development holdout has been inspected during this iteration.','Two first-visit centrifuge negatives adapted the coat head. The three return-visit negatives share the camera and clothing.','PPE training is limited to classifier heads, not the language model or detector.','Unknown visual samples were excluded from binary training.','No ordinary white non-lab garment negatives are available.']};(out/'training_report_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({item:{'train':v['training_rows'],'holdout':{k:x for k,x in v['development_holdout'].items() if k!='predictions'}} for item,v in results.items()}),flush=True)
# Run the unchanged AIZOO mask detector on exactly the same held-out person inputs.
sys.path.insert(0,str(R));from ppe_detector import MaskDetector
from PIL import Image
import cv2
mask=MaskDetector(R/'runtime/vendor_aizoo/FaceMaskDetection-master');byid={r['sample_id']:r for r in rows if r['item']=='mask'};baseline=[]
for p in results['mask']['development_holdout']['predictions']:
 r=byid[p['sample_id']];s=next(x for x in rows if x['sample_id']==p['sample_id'] and x['item']=='mask');input_path=s['image'];base_path=Path(input_path).with_name(Path(input_path).name.replace('_head_region.jpg','_person.jpg'));image=cv2.imread(str(base_path));pred=mask.observe(image,face_crop=False);baseline.append({'sample_id':s['sample_id'],'truth':s['state'],'state':pred['state']})
report['baseline_mask']={'n':len(baseline),'correct':sum(p['state']==p['truth'] for p in baseline),'unknown':sum(p['state']=='unknown' for p in baseline),'wrong':sum(p['state'] not in ['unknown',p['truth']] for p in baseline),'predictions':baseline};(out/'training_report_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('BASELINE_MASK',json.dumps({k:v for k,v in report['baseline_mask'].items() if k!='predictions'}),flush=True)
