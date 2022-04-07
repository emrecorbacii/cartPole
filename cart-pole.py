import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
import random


class DQLAgent:
    def __init__(self,env):
        # All parameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        
    def build_model(self):
        # Neural Network for DeepQLearning
        model = Sequential()
        model.add(Dense(48,input_dim = self.state_size,activation="tanh"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss="mse",optimizer = Adam(lr = self.learning_rate))
        return model
        
    def remember(self,state,action,reward,next_state,done):
        # Storage
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        # Acting
       if random.uniform(0,1) <= self.epsilon:
           return env.action_space.sample()
       else:
           act_values = self.model.predict(state)
           return np.argmax(act_values[0])
    
    def replay(self,batch_size):
        # Training
        if len(self.memory) < batch_size:
            return
        else:
            minibatch = random.sample(self.memory,batch_size)
            for state,action,reward,next_state,done in minibatch:
                if done:
                    target = reward
                    
                else:
                    target = reward+self.gamma*np.amax(self.model.predict(next_state)[0])
                    train_target = self.model.predict(state)
                    train_target[0][action] = target
                    self.model.fit(state,train_target,verbose = 0)
    def adaptiveEGreedy(self):
        
       if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
        
        
        
    
    
    
if __name__ == "__main__":
    
    ####### Initialize env and agent
    env = gym.make("CartPole-v0")
    agent=DQLAgent(env)
    
    
    batch_size = 16
    episodes = 100
    for e in range(episodes):
        # Initialize env
        state = env.reset()
        state = np.reshape(state,[1,4])
        time = 0
        
        while True:
            env.render()
            # Act
            action = agent.act(state)
            # Step
            next_state,reward,done,_=env.step(action)
            next_state = np.reshape(next_state,[1,4])
            # Remember
            agent.remember(state,action,reward,next_state,done)
            
            # Update state
            state=next_state
            # Replay
            agent.replay(batch_size)
            # Adjust epsilon
            agent.adaptiveEGreedy()
            
            #########
            time += 1
            if done:
                print("Episode : {}, Time : {}".format(e,time))
                break
            
    
