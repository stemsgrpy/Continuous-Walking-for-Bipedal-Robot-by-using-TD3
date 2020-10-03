# Continuous-Walking-for-Bipedal-Robot-by-using-TD3

## End-to-end (Input to Output)
- State (Input)  
  - Various information (angle, angular, orientation, sensor)   
```
    config.state_dim = env.observation_space.shape[0]
```

- Action (Output)  
  - **Continuous**   
```
    config.action_dim = env.action_space.shape[0]
```

## Reinforcement Learning TD3
### Train
```
python TD3.py --train --env BipedalWalker-v3

python TD3.py --train --env LunarLanderContinuous-v2
```

### Test
```
python TD3.py --test --env BipedalWalker-v3 --model_path out/BipedalWalker-v3-runx/actor_xxxx.pkl

python TD3.py --test --env LunarLanderContinuous-v2 --model_path out/LunarLanderContinuous-v2-runx/actor_xxxx.pkl
```

### Retrain
```
python TD3.py --retrain --env BipedalWalker-v3 --model_path out/BipedalWalker-v3-runx/checkpoint_policy/checkpoint_fr_xxxxx.tar

python TD3.py --retrain --env LunarLanderContinuous-v2 --model_path out/LunarLanderContinuous-v2-runx/checkpoint_policy/checkpoint_fr_xxxxx.tar
```

## Result

BipedalWalker-v3 (Continuous)    | LunarLanderContinuous-v2 (Continuous)
:-------------------------------:|:-------------------------------:
![](/README/BipedalWalker-v3.gif) |  ![](/README/LunarLanderContinuous-v2.gif)
<p align="center">
  Figure 1: Reinforcement Learning TD3 on Bipedal Robot
</p>

## Reference
[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)  
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)  