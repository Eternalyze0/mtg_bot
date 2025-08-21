import json

f_name = 'oracle-cards-20250821090338.json'

with open(f_name) as f:
	cards = json.loads(f.read())

standard_cards = [card for card in cards if card['legalities']['standard'] == 'legal']

basic_land_names = ['Swamp', 'Island', 'Forest', 'Mountain', 'Plains']

n_actions = len(standard_cards)

# print(n_actions, 'cards/actions in standard')

def scryfall_to_forge(scryfall_deck, name='Test Deck'):
	forge_deck = {}
	for scryfall_card in scryfall_deck:
		forge_card = scryfall_card['name']+'|'+scryfall_card['set'].upper()
		if scryfall_card['name'] in basic_land_names:
			forge_card += '|1'
		if forge_card in forge_deck:
			forge_deck[forge_card] += 1
		else:
			forge_deck[forge_card] = 1
	forge_deck_string = '[metadata]' + '\nName=' + name + '\n[Main]\n'
	for card, count in sorted(forge_deck.items(), key=lambda x: x[0]):
		forge_deck_string += str(count) + ' ' + card + '\n'
	return forge_deck_string

scryfall_deck = []

for card in standard_cards:
	if card['name'] == 'Swamp':
		for _ in range(30):
			scryfall_deck.append(card)
		break

for card in standard_cards:
	if card['name'] == 'Island':
		for _ in range(30):
			scryfall_deck.append(card)
		break

# print(scryfall_to_forge(scryfall_deck), end='')

# % python3.10 mtg_bot.py
# 3251 cards/actions in standard
# [metadata]
# Name=Test Deck
# [Main]
# 30 Island|TLA|1
# 30 Swamp|TLA|1

import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


class ForgeEnv(gym.Env):
	def __init__(self):
		super(ForgeEnv, self).__init__()
		self.deck = []
		self.state_vector = np.zeros(n_actions)
	def step(self, action): # action is a card selection
		self.deck.append(standard_cards[action]) # first add the new card to the deck list
		self.state_vector[action] += 1.0
		state = self.state_vector
		reward = 0 # if won sim, set to 1, if lost, set to -1
		if len(self.deck) >= 60:
			done = True
			# RUN FORGE SIMULATION HERE
		else:
			done = False
		truncated = False
		info = None
		return state, reward, done, truncated, info
	def reset(self):
		self.deck = []
		self.state_vector = np.zeros(n_actions)
		state = self.state_vector
		return state
	def render(self, mode="human"):
		pass
	def close(self):
		pass

class ReplayBuffer():
	def __init__(self):
		self.buffer = collections.deque(maxlen=buffer_limit)
	
	def put(self, transition):
		self.buffer.append(transition)
	
	def sample(self, n):
		mini_batch = random.sample(self.buffer, n)
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
		
		for transition in mini_batch:
			s, a, r, s_prime, done_mask = transition
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			done_mask_lst.append([done_mask])

		return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
			   torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
			   torch.tensor(done_mask_lst)
	
	def size(self):
		return len(self.buffer)

class Qnet(nn.Module):
	def __init__(self):
		super(Qnet, self).__init__()
		self.fc1 = nn.Linear(n_actions, 128) # state should be the cards cards already in the deck
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, n_actions)
 
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	  
	def sample_action(self, obs, epsilon):
		out = self.forward(obs)
		coin = random.random()
		if coin < epsilon:
			return random.randint(0,1)
		else : 
			return out.argmax().item()
			
def train(q, q_target, memory, optimizer):
	for i in range(10):
		s,a,r,s_prime,done_mask = memory.sample(batch_size)

		q_out = q(s)
		q_a = q_out.gather(1,a)
		max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
		target = r + gamma * max_q_prime * done_mask
		loss = F.smooth_l1_loss(q_a, target)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def main():
	# env = gym.make('CartPole-v1')
	env = ForgeEnv()
	q = Qnet()
	q_target = Qnet()
	q_target.load_state_dict(q.state_dict())
	memory = ReplayBuffer()

	print_interval = 20
	score = 0.0  
	optimizer = optim.Adam(q.parameters(), lr=learning_rate)

	for n_epi in range(1):
		epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
		s = env.reset()
		done = False

		while not done:
			a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
			s_prime, r, done, truncated, info = env.step(a)
			done_mask = 0.0 if done else 1.0
			memory.put((s,a,r/100.0,s_prime, done_mask))
			s = s_prime

			score += r
			if done:
				print(scryfall_to_forge(env.deck))
				break
			
		if memory.size()>=batch_size:
			train(q, q_target, memory, optimizer)

		if n_epi%print_interval==0 and n_epi!=0:
			q_target.load_state_dict(q.state_dict())
			print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
															n_epi, score/print_interval, memory.size(), epsilon*100))
			score = 0.0
	env.close()

if __name__ == '__main__':
	main()


# % python3.10 mtg_bot.py
# [metadata]
# Name=Test Deck
# [Main]
# 5 Choco, Seeker of Paradise|FIN
# 7 Dion, Bahamut's Dominant // Bahamut, Warden of Light|FIN
# 10 Fortress Kin-Guard|TDM
# 29 Geyser Drake|OTJ
# 3 Greta, Sweettooth Scourge|WOE
# 6 Silver Deputy|OTJ



