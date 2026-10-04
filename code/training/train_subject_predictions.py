from pathlib import Path
import os, json, random, copy
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('KNEEPAD_DATA', REPO_ROOT / 'data' / 'processed' / 'deep_data'))
OUT = Path(os.environ.get('KNEEPAD_OUT', REPO_ROOT / 'results' / 'strict'))
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

raw=np.load(DATA/'labels.npy'); N=len(raw)
E=np.memmap(DATA/'emg_official.dat',dtype='float32',mode='r',shape=(N,8,5037)); I=np.memmap(DATA/'imu_official.dat',dtype='float32',mode='r',shape=(N,48,593))
tids=np.load(DATA/'trial_ids.npy',allow_pickle=True); subjects=np.load(DATA/'subjects.npy',allow_pickle=True)
SUBSET=json.loads(os.environ['SUBSET']); EX=int(os.environ['EXERCISE']); USE_EMG=int(os.environ['USE_EMG']); CONFIG=os.environ['CONFIG']
EPOCHS=int(os.environ.get('EPOCHS','10')); SEED=int(os.environ.get('SEED','2026')); INNER_FOLDS=int(os.environ.get('INNER_FOLDS','3'))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get('TORCH_THREADS', str(max(1,min(8,os.cpu_count() or 2))))))
mask=(raw//3==EX); idx_all=np.where(mask)[0]; y=((raw[idx_all]%3)!=0).astype(np.int64); tids_sub=tids[idx_all]
tt=pd.DataFrame({'tid':tids_sub,'y':y,'sid':subjects[idx_all]}).drop_duplicates('tid').reset_index(drop=True)
outer_folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(tt.tid,tt.y,groups=tt.sid))
imu_ch=np.array([c for s in SUBSET for c in range((s-1)*6,s*6)],dtype=int)

class DS(Dataset):
 def __init__(self,idx,em,es,im,ins,train=False): self.idx=np.asarray(idx); self.em=em; self.es=es; self.im=im; self.ins=ins; self.train=train
 def __len__(self): return len(self.idx)
 def __getitem__(self,k):
  j=int(self.idx[k]); pos=int(np.where(idx_all==j)[0][0]); i=np.asarray(I[j,imu_ch,::2],np.float32); i=(i-self.im[:,None])/(self.ins[:,None]+1e-5)
  if USE_EMG: e=np.asarray(E[j,:,::20],np.float32); e=(e-self.em[:,None])/(self.es[:,None]+1e-5)
  else: e=np.zeros((1,1),np.float32)
  if self.train:
   i=i+np.random.normal(0,.015,i.shape).astype(np.float32)
   if USE_EMG: e=e+np.random.normal(0,.015,e.shape).astype(np.float32)
  return torch.from_numpy(e),torch.from_numpy(i),torch.tensor(int(y[pos]),dtype=torch.long)rd)ELU(),nn.Conv1d(c2,c2,5,padding=2),nn.BatchNorm1d(c2),nn.GELU(),nn.MaxPool1d(2))
 def forward(self,x): return self.net(x)
class Model(nn.Module):
 def __init__(self):
  super().__init__(); self.use_emg=USE_EMG; self.imu=nn.Sequential(Block(len(imu_ch),48),Block(48,64))
  if USE_EMG: self.emg=nn.Sequential(Block(8,32),Block(32,64)); self.proj=nn.Linear(128,128)
  else: self.proj=nn.Linear(64,128)
  enc=nn.TransformerEncoderLayer(128,8,dim_feedforward=256,dropout=.15,batch_first=True,norm_first=True,activation='gelu'); self.tr=nn.TransformerEncoder(enc,2); self.att=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,1)); self.head=nn.Sequential(nn.LayerNorm(128),nn.Dropout(.25),nn.Linear(128,2))
 def forward(self,e,i):
  b=self.imu(i).transpose(1,2)
  if self.use_emg:
   a=self.emg(e).transpose(1,2); n=min(a.size(1),b.size(1)); z=self.proj(torch.cat([a[:,:n],b[:,:n]],-1))
  else: z=self.proj(b)
  z=self.tr(z); w=torch.softmax(self.att(z).squeeze(-1),-1).unsqueeze(-1); return self.head((z*w).sum(1))
def stats(train):
 im=np.asarray(I[train][:,imu_ch,::2],np.float32); im=im.mean((0,2)),im.std((0,2))
 if USE_EMG:
  em=np.asarray(E[train,:,::20],np.float32); return em.mean((0,2)),em.std((0,2)),im[0],im[1]
 return np.zeros(8),np.ones(8),im[0],im[1]
def loaders(gtr,gva,gte):
 em,es,im,ins=stats(gtr)
 return DataLoader(DS(gtr,em,es,im,ins,True),64,True,num_workers=0),DataLoader(DS(gva,em,es,im,ins,False),128,False,num_workers=0),DataLoader(DS(gte,em,es,im,ins,False),128,False,num_workers=0)
def criterion(train_pos,device):
 cnt=np.bincount(y[train_pos],minlength=2); wt=np.sqrt(cnt.sum()/np.maximum(cnt,1)); wt=wt.mean()
 return nn.CrossEntropyLoss(weight=torch.tensor(wt,dtype=torch.float32,device=device),label_smoothing=.04)
def evaluate(model,loader,device):
 model.eval(); yy=[]; pp=[]; pr=[]
 with torch.no_grad():
  for e,i,t in loader:
   logits=model(e.to(device),i.to(device)); prob=torch.softmax(logits,1).cpu().numpy(); pr.append(prob); pp.extend(prob.argmax(1)); yy.extend(t.numpy())
 yy=np.asarray(yy); pp=np.asarray(pp); pr=np.concatenate(pr); return {'accuracy':accuracy_score(yy,pp),'balanced_accuracy':balanced_accuracy_score(yy,pp),'macro_f1':f1_score(yy,pp,average='macro'),'y_true':yy,'y_pred':pp,'prob_correct':pr[:,0],'prob_wrong':pr[:,1]}
def train_epochs(train_loader,val_loader,train_pos,device,epochs):
 model=Model().to(device); crit=criterion(train_pos,device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=2e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); best_score=-1; best_epoch=epochs; best_state=None; history=[]
 for ep in range(1,epochs+1):
  model.train()
  for e,i,t in train_loader:
   opt.zero_grad(); loss=crit(model(e.to(device),i.to(device)),t.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  sched.step(); met=evaluate(model,val_loader,device); history.append(met)
  if met['balanced_accuracy']>best_score: best_score=met['balanced_accuracy']; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 return best_epoch,best_score,history

def train_fixed(train_loader,train_pos,device,epochs):
 model=Model().to(device); crit=criterion(train_pos,device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=2e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
 for _ in range(epochs):
  model.train()
  for e,i,t in train_loader:
   opt.zero_grad(); loss=crit(model(e.to(device),i.to(device)),t.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  sched.step()
 return model

device='cuda' if torch.cuda.is_available() else 'cpu'; rows=[]; all_inner_rows=[]; config_dir=OUT/CONFIG; config_dir.mkdir(parents=True,exist_ok=True)
for fi,(trg,teg) in enumerate(outer_folds):
 tr_trials=tt.iloc[trg].reset_index(drop=True); te_trials=set(tt.tid.iloc[teg])
 inner=(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+fi) if INNER_FOLDS==1 else StratifiedGroupKFold(INNER_FOLDS,shuffle=True,random_state=SEED+fi))
 epoch_scores=np.zeros(EPOCHS,float); epoch_counts=np.z