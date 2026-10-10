"""Persistent groups of ordinary Gymnasium environments in worker processes."""
import multiprocessing as mp
import traceback
from .environments import make_env


def _worker(connection,config,count):
    envs=[]
    try:
        envs=[make_env(config) for _ in range(count)];observations=[None]*count
        while True:
            command,payload=connection.recv()
            if command=='close':break
            if command=='reset':
                seeds=[payload]*len(envs) if isinstance(payload,int) else payload
                observations=[env.reset(seed=int(seed))[0] for env,seed in zip(envs,seeds)]
                connection.send(observations)
            elif command=='step':
                rewards=[];dones=[]
                for i,(env,action) in enumerate(zip(envs,payload)):
                    if action is None:rewards.append(0.);dones.append(True);continue
                    observations[i],reward,terminated,truncated,_=env.step(action);rewards.append(float(reward));dones.append(bool(terminated or truncated))
                connection.send((observations,rewards,dones))
            else:raise ValueError('Unknown environment command')
    except (EOFError,BrokenPipeError):pass
    except Exception:
        try:connection.send({'error':traceback.format_exc()})
        except (EOFError,BrokenPipeError):pass
    finally:
        for env in envs:env.close()
        connection.close()


class EnvPool:
    def __init__(self,config,count,workers=8):
        if count<1 or workers<1:raise ValueError('Invalid environment pool size')
        self.count=count;self.connections=[];self.processes=[];self.slices=[]
        context=mp.get_context('spawn');workers=min(count,workers)
        try:
            for worker in range(workers):
                first=count*worker//workers;last=count*(worker+1)//workers
                parent,child=context.Pipe();process=context.Process(target=_worker,args=(child,config,last-first),daemon=True);process.start();child.close()
                self.connections.append(parent);self.processes.append(process);self.slices.append(slice(first,last))
        except Exception:self.close();raise
    def receive(self,connection):
        result=connection.recv()
        if isinstance(result,dict) and 'error' in result:raise RuntimeError(result['error'])
        return result
    def reset(self,seed):
        if hasattr(seed,'__len__'):
            if len(seed)!=self.count:raise ValueError('Reset seed batch differs')
            for connection,section in zip(self.connections,self.slices):connection.send(('reset',[int(s) for s in seed[section]]))
        else:
            for connection in self.connections:connection.send(('reset',int(seed)))
        return [obs for connection in self.connections for obs in self.receive(connection)]
    def step(self,actions):
        if len(actions)!=self.count:raise ValueError('Action batch differs')
        for connection,section in zip(self.connections,self.slices):connection.send(('step',actions[section]))
        observations=[];rewards=[];dones=[]
        for connection in self.connections:
            obs,reward,done=self.receive(connection);observations.extend(obs);rewards.extend(reward);dones.extend(done)
        return observations,rewards,dones
    def close(self):
        for connection in self.connections:
            try:connection.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():process.terminate();process.join(timeout=2)
        for connection in self.connections:connection.close()
        self.connections=[];self.processes=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
