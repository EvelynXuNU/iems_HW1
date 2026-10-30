"""Complete on-policy episode batches: CPU environments and one policy device.

Workers send preprocessed features, not RGB frames. They never import torch or
touch CUDA. No autoreset and no replacement of early-finishing episodes.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np


def _environment_worker(connection, slots, game):
    import gymnasium as gym
    if game == "pong":
        import ale_py
        gym.register_envs(ale_py)
    environments, previous = {}, {}

    def features(slot, observation, reset=False):
        if game == "cartpole":
            return observation.astype(np.float32, copy=True)
        crop=observation[35:195:2,::2,0].copy()
        crop[(crop==144)|(crop==109)]=0
        current=(crop!=0).astype(np.int8)
        result=np.zeros_like(current) if reset else current-previous[slot]
        previous[slot]=current
        return result.ravel()

    try:
        for slot in slots:
            env=(gym.make("ALE/Pong-v5",frameskip=4,repeat_action_probability=.25,
                          full_action_space=False,obs_type="rgb") if game=="pong"
                 else gym.make("CartPole-v1"))
            if game=="pong" and env.unwrapped.get_action_meanings()[2:4] != ["RIGHT","LEFT"]:
                raise RuntimeError("Unexpected Pong action mapping")
            environments[slot]=env
        while True:
            command,payload=connection.recv()
            if command=="close": break
            rows=[]
            if command=="reset":
                for slot,seed in payload:
                    observation,_=environments[slot].reset(seed=int(seed))
                    rows.append((slot,features(slot,observation,reset=True),
                                 observation.tolist() if game=="cartpole" else None))
            elif command=="step":
                for slot,action in payload:
                    observation,reward,terminated,truncated,_=environments[slot].step(
                        int(action)+2 if game=="pong" else int(action))
                    rows.append((slot,features(slot,observation),float(reward),bool(terminated),
                                 bool(truncated),observation.tolist() if game=="cartpole" else None))
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True,rows))
    except (EOFError,BrokenPipeError):
        pass
    except Exception:
        try: connection.send((False,traceback.format_exc()))
        except (EOFError,BrokenPipeError): pass
    finally:
        for env in environments.values():env.close()
        connection.close()


class EpisodeBatch:
    def __init__(self, game, slots=16, workers=4):
        if not 1<=workers<=slots:raise ValueError("Require 1 <= workers <= episode slots")
        self.game,self.slots,self.workers=game,slots,workers
        self.connections,self.processes,self.owner=[],[],{}
        context=mp.get_context("spawn")
        for worker in range(workers):
            group=list(range(worker,slots,workers))
            parent,child=context.Pipe()
            process=context.Process(target=_environment_worker,args=(child,group,game),daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
            for slot in group:self.owner[slot]=worker

    def exchange(self,command,payload):
        grouped={}
        for slot,value in payload:grouped.setdefault(self.owner[slot],[]).append((slot,value))
        for worker,items in grouped.items():self.connections[worker].send((command,items))
        rows=[]
        for worker in grouped:
            connection=self.connections[worker]
            if not connection.poll(120):raise TimeoutError("Environment worker did not respond")
            okay,result=connection.recv()
            if not okay:raise RuntimeError(result)
            rows.extend(result)
        return rows

    def collect(self,policy,config,environment_seeds,action_seeds,device="cpu",store_states=True,trace=False):
        import torch
        count=len(environment_seeds)
        if not 1<=count<=self.slots or len(action_seeds)!=count:
            raise ValueError("Provide one action seed per complete episode")
        rngs=[np.random.default_rng(int(s)) for s in action_seeds]
        reset=self.exchange("reset",list(enumerate(environment_seeds)))
        features={slot:state for slot,state,_ in reset}
        observations={slot:obs for slot,_,obs in reset}
        records=[{"states":[],"actions":[],"rewards":[],"trace":[]} for _ in range(count)]
        active=list(range(count))
        with torch.no_grad():
            while active:
                states=np.stack([features[slot] for slot in active])
                logits=policy(torch.as_tensor(states,dtype=torch.float32,device=device))
                probabilities=logits.softmax(-1).cpu().numpy()  # one transfer for the whole batch
                actions={slot:int(rngs[slot].random()>=float(probabilities[j,0])) for j,slot in enumerate(active)}
                traces={}
                for j,slot in enumerate(active):
                    e=records[slot]
                    if store_states:
                        e["states"].append(states[j].copy())
                        e["actions"].append(actions[slot])
                    if trace and slot==0 and len(e["rewards"])<10:
                        traces[slot]={"step":len(e["rewards"]),
                          "observation":observations[slot] if self.game=="cartpole" else
                                        {"nonzero_motion_pixels":int(np.count_nonzero(states[j]))},
                          "p_action_0":float(probabilities[j,0]),"policy_action":actions[slot],
                          "environment_action":actions[slot]+2 if self.game=="pong" else actions[slot]}
                outcomes=self.exchange("step",[(slot,actions[slot]) for slot in active])
                finished=set()
                for slot,state,reward,terminated,truncated,observation in outcomes:
                    e=records[slot]
                    e["rewards"].append(reward)
                    if slot in traces:
                        row=traces[slot]
                        row.update({"reward":reward,"terminated":terminated,"truncated":truncated,
                                    "next_observation":observation})
                        e["trace"].append(row)
                    features[slot],observations[slot]=state,observation
                    if terminated or truncated:
                        e["terminated"],e["truncated"]=terminated,truncated
                        finished.add(slot)
                active=[slot for slot in active if slot not in finished]
        # Return in episode order, including every complete episode, not completion order.
        from q2_policy_gradient import discounted_returns,objective_weights
        for e in records:
            e["rewards"]=np.asarray(e["rewards"],dtype=np.float64)
            e["length"]=len(e["rewards"])
            e["score"]=float(e["rewards"].sum())
            returns=discounted_returns(e["rewards"],config.gamma)
            e["discounted_return"]=float(returns[0])
            if store_states:
                e["states"]=np.asarray(e["states"])
                e["actions"]=np.asarray(e["actions"],dtype=np.int64)
                e["weights"]=objective_weights(e["rewards"],config.gamma,config.reward_scale)
            for row in e["trace"]:
                row["reward_to_go"]=float(returns[row["step"]])
                row["gradient_weight"]=float(config.gamma**row["step"]*returns[row["step"]]/config.reward_scale)
        return records

    def close(self):
        for connection in self.connections:
            try:connection.send(("close",None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(3)
            if process.is_alive():process.terminate();process.join(3)
        for connection in self.connections:connection.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
