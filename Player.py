import random
import math
import pickle
import numpy as np
from State import State
from Memory import Memory

class Player(object):
    def __init__(self, name, env, symbol, memory_capacity, model=None, targets=None, BATCH_SIZE=0, EPSILON_ARGS=(0,0,0), maximize_entropy=False, use_PER=False, PER_hyperparams=(0,0,0)):
        self.GAMMA = 0.9
        self.EPSILON_MAX, self.EPSILON_MIN, self.LAMBDA = EPSILON_ARGS
        self.exploration_rate = self.EPSILON_MAX
        self.BATCH_SIZE = BATCH_SIZE
        self.TEST_SPLIT = 0.9
        self.epsilon_decay_steps = 0
        
        self.name = name
        self.env = env
        self.symbol = symbol
        self.samples = []
        self.model = model
        self.targets = targets
        self.use_PER = use_PER
        self.PER_hyperparams = PER_hyperparams
        self.memory = Memory(memory_capacity, False, self.use_PER, self.PER_hyperparams)
        self.test_memory = Memory(10, True)
        self.accuracy = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []
        self.total_rewards = [0]
        self.average_reward = [0]
        self.invalid_moves = [0]
        self.regret = [1] # optimal reward - actual reward
        self.was_random = False
        self.maximize_entropy = maximize_entropy
        
        self.win = 0
        self.losses = 0
        self.draw = 0
    
    def choose_action(self, state, moves=None, is_random=False):
        action = None
        moves = self.env.get_legal_moves() if moves is None else moves
        if random.random() < self.exploration_rate or is_random:
            # explore
            action = (random.choice(range(self.env.NUM_ROWS)), random.choice(range(self.env.NUM_COLS)))
            self.was_random = True
        else:
            # exploit
            q_values = self.model.predict_one(state.board.reshape(-1)).reshape(3, 3)
            
            maximum = np.amax(q_values)

            # convert from linear to 2D indicies
            location = np.where(q_values==maximum)
            action = list(zip(location[0], location[1]))[0]          
            self.was_random = False
        
        return action
    
    def train(self):
        # train the model based on the reward
        flatten = lambda arr: arr.reshape(-1) if arr is not None else np.zeros(self.env.NUM_ROWS*self.env.NUM_COLS)
        if self.use_PER:
            tree_idxs, samples = self.memory.sample(self.BATCH_SIZE)
        else:
            samples = self.memory.sample(self.BATCH_SIZE)
        
        if len(samples) == 0:
            return
        
        states, actions, rewards, next_states, completes = np.array(samples).T

        states = np.array(list(map(lambda x: flatten(x.board), states)))
        next_states = np.array(list(map(lambda x: flatten(x.board) if x is not None else np.zeros(self.env.NUM_ROWS*self.env.NUM_COLS), next_states)))
        
        q_s_a = self.targets.predict_batch(states)
        q_s_a_p = self.model.predict_batch(next_states)

        # training arrays
        x = np.array(list(map(flatten, states)))
        y = np.array(list(map(flatten, q_s_a)))

        actions = np.array(list(map(lambda x: None if x is None else x[0] * self.env.NUM_COLS + x[1], actions)))

        next_actions = np.argmax(q_s_a_p, axis=1)
        fake_states = next_states.copy()
        fake_states[range(len(next_actions)), next_actions] = self.symbol
        future_q = np.amax(self.targets.predict_batch(fake_states), axis=1)
            
        updated_q = np.add(rewards, (1 - np.array(completes)) * self.GAMMA * future_q)

        y[range(len(actions)), actions] = updated_q
        
        if self.use_PER:
            abs_error = np.abs(q_s_a[range(len(actions)), actions] - updated_q)
            self.memory.update(tree_idxs, abs_error)
            
        data = self.model.train_batch(x, y, self.BATCH_SIZE)
        self.accuracy.append(data.history['accuracy'][0])
        self.loss.append(data.history['loss'][0])
        self.val_loss.append(data.history.get('val_loss', [0])[0])
        self.val_acc.append(data.history.get('val_accuracy', [0])[0])
        self.decay_exploration_rate()
        
    def decay_exploration_rate(self):
        self.exploration_rate = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-1*self.LAMBDA * self.epsilon_decay_steps)
        self.epsilon_decay_steps += 1
        
    def reset(self, reward):
        # samples should be of the form (state, action, reward, next_state, complete)
        while len(self.samples) > 0:
            sample = self.samples[0]
            sample.insert(2, reward)
            if not sample[4]:
                try:
                    sample[3] = self.samples[1][0]
                except IndexError:
                    # if the game wasn't over when the player played, but ended
                    # the next move, have None as the next state
                    sample[3] = None
                    
            else:
                sample[3] = None
            
            self.memory.add_sample(tuple(sample))
            self.samples.pop(0)
            if reward < 0:
                # the agent made an illegal move here
                # this reard shoouldn't affect other states in the game,
                # so we exit the loop
                self.samples = []
                break
                
        self.total_rewards.append(self.total_rewards[-1] + reward)
        self.average_reward.append(self.total_rewards[-1]/len(self.total_rewards))
        self.regret.append(len(self.total_rewards) - self.total_rewards[-1])

    def update_targets(self):
        self.model.copy_weights(self.targets)
    
    def save_policy(self, prefix):
        fout = open("{}policy_{}".format(prefix, self.name), 'wb')
        pickle.dump(self.model, fout)
        fout.close()
    
    def load_policy(self, name, prefix=None):
        self.model = pickle.load(open("{}policy_{}".format(prefix, name), 'rb'))
        
    def get_metrics(self):
        return {'loss': self.loss,
                'accuracy': self.accuracy,
                'reward': self.total_rewards,
                'average_reward': self.average_reward,
                'regret': self.regret,
                'invalid_moves': self.invalid_moves}
    
    def __str__(self):
        return self.name

class Human(Player):
    def __init__(self, name, env, symbol):
        super().__init__(name, env, symbol, 0)
    
    def choose_action(self, state, moves=None):
        action = None
        i=0
        while action not in self.env.get_legal_moves():
            if i != 0:
                print("{} is not valid. Please try again.".format(action))
            row = int(input("Type a row to move to: "))
            col = int(input("Type a col to move to: "))
            action = (row, col)
            i+=1
        return action
