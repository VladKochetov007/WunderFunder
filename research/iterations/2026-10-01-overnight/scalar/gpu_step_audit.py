"""Generated-probe only readback audits, outside the timed ordinary update region."""
import torch

def snapshot(model,optimizer):
 params={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
 states={}
 for name,p in model.named_parameters():
  states[name]={k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in optimizer.state.get(p,{}).items()}
 return {'parameters':params,'optimizer':states}

def step_audit(before,model,optimizer,record,completed_updates):
 after=snapshot(model,optimizer)
 skipped=record['scale_after']<record['scale_before']
 finite_loss=bool(torch.isfinite(torch.tensor(record['loss'],dtype=torch.float64)))
 finite_gradient=bool(torch.isfinite(torch.tensor(record['gradient_norm'],dtype=torch.float64)))
 params_finite=all(bool(torch.isfinite(v).all()) for v in after['parameters'].values())
 params_unchanged=all(torch.equal(v,after['parameters'][k]) for k,v in before['parameters'].items())
 states_finite=all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for state in after['optimizer'].values() for v in state.values())
 steps_before=[];steps_after=[];moments_unchanged_or_newzero=True;required_states=True
 for name,state in after['optimizer'].items():
  previous=before['optimizer'][name]
  required_states &= set(state)=={'step','exp_avg','exp_avg_sq'}
  steps_before.append(float(previous.get('step',0)))
  steps_after.append(float(state.get('step',-1)))
  for key in ['exp_avg','exp_avg_sq']:
   if key not in state:moments_unchanged_or_newzero=False
   elif key in previous:moments_unchanged_or_newzero &= torch.equal(state[key],previous[key])
   else:moments_unchanged_or_newzero &= bool(torch.count_nonzero(state[key])==0)
 expected=completed_updates+(not skipped)
 checks={'skipped_by_scaler':skipped,'finite_loss':finite_loss,'finite_gradient_norm':finite_gradient,'all_parameters_finite':params_finite,'all_optimizer_tensors_finite':states_finite,'parameters_unchanged':params_unchanged,'moments_unchanged_or_new_zero':bool(moments_unchanged_or_newzero),'all_optimizer_states_allocated':bool(required_states),'all_step_counters_before_expected':all(v==completed_updates for v in steps_before),'all_step_counters_after_expected':all(v==expected for v in steps_after),'completed_updates_before':completed_updates,'completed_updates_after':int(expected),'step_counter_before_range':[min(steps_before),max(steps_before)],'step_counter_after_range':[min(steps_after),max(steps_after)]}
 checks['PASS']=bool(finite_loss and params_finite and states_finite and required_states and checks['all_step_counters_before_expected'] and checks['all_step_counters_after_expected'] and ((record['step']<3 and params_unchanged and moments_unchanged_or_newzero) if skipped else (finite_gradient and not params_unchanged)))
 return checks,int(expected)

def final_audit(model,ema,optimizer,completed_updates):
 state=snapshot(model,optimizer)
 return {'model_finite':all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'EMA_finite':all(bool(torch.isfinite(v).all()) for v in ema.module.state_dict().values()),'optimizer_finite':all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for s in state['optimizer'].values() for v in s.values()),'all_step_counters_completed_updates':all(float(s['step'])==completed_updates for s in state['optimizer'].values()),'completed_updates':completed_updates}
