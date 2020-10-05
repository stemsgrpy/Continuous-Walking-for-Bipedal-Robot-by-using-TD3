import argparse
import os
import gym
import torch
from torch.optim import Adam
from core.util import get_class_attr_val
from config import Config
from buffer import ReplayBuffer
from model import Actor, Critic
from trainer import Trainer
from tester import Tester
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        # self.buffer = deque(maxlen=self.config.max_buff)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.actor = Actor(self.config.state_dim, self.config.action_dim, self.config.max_action)
        self.actor_target = Actor(self.config.state_dim, self.config.action_dim, self.config.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.learning_rate)
        
        self.critic_1 = Critic(self.config.state_dim, self.config.action_dim)
        self.critic_1_target = Critic(self.config.state_dim, self.config.action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=self.config.learning_rate)
        
        self.critic_2 = Critic(self.config.state_dim, self.config.action_dim)
        self.critic_2_target = Critic(self.config.state_dim, self.config.action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=self.config.learning_rate)

        self.MseLoss = nn.MSELoss()

        if self.config.use_cuda:
            self.cuda()

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()  #.detach()

    def learning(self, fr, t):
        
        for i in range(t):
            state, action_, reward, next_state, done = self.buffer.sample(self.config.batch_size)

            state = torch.tensor(state, dtype=torch.float).to(device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            action = torch.tensor(action_, dtype=torch.float).to(device)
            reward = torch.tensor(reward, dtype=torch.float).reshape((-1,1)).to(device)
            done = torch.tensor(done, dtype=torch.float).reshape((-1,1)).to(device)
            # reward = torch.FloatTensor(reward).reshape((self.config.batch_size,1)).to(device)
            # done = torch.FloatTensor(done).reshape((self.config.batch_size,1)).to(device)

            # Select next action according to target policy:
            noise = torch.tensor(action_, dtype=torch.float).data.normal_(0, self.config.policy_noise).to(device)
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.config.max_action, self.config.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * self.config.gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % self.config.policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (self.config.polyak * target_param.data) + ((1-self.config.polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (self.config.polyak * target_param.data) + ((1-self.config.polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (self.config.polyak * target_param.data) + ((1-self.config.polyak) * param.data))

    def cuda(self):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic_1.to(device)
        self.critic_1_target.to(device)
        self.critic_2.to(device)
        self.critic_2_target.to(device)

    def load_weights(self, model_path):
        policy = torch.load(model_path)
        if 'actor' in policy:
            self.actor.load_state_dict(policy['actor'])
        else:
            self.actor.load_state_dict(policy)

    def save_model(self, output, name=''):
        torch.save(self.actor.state_dict(), '%s/actor_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_policy'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        return fr

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--env', default='BipedalWalker-v3', type=str, help='gym environment')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    args = parser.parse_args()

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.frames = 1000000
    config.use_cuda = True
    config.learning_rate = 0.001
    config.max_buff = 500000
    config.batch_size = 100
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 5000
    config.win_reward = 300
    config.win_break = True

    config.max_episodes = 1000         # max num of episodes
    config.max_timesteps = 2000        # max timesteps in one episode
    config.exploration_noise = 0.1
    config.polyak = 0.995              # target policy update parameter (1-tau)
    config.policy_noise = 0.2          # target policy smoothing noise
    config.noise_clip = 0.5
    config.policy_delay = 2            # delayed policy updates parameter

    # env.seed(1)
    # random.seed(1)
    # np.random.seed(1)
    # torch.manual_seed(1)

    env = gym.make(config.env)

    config.state_dim = env.observation_space.shape[0]

    config.action_type = 'Continuous'
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    print('Continuous', config.state_dim, config.action_dim)

    agent = TD3(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
