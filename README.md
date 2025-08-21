# Magic: The Gathering Bot

## Simulating Games with Forge and Scryfall

Forge supports headless AI simulations so it may be easier to interface with (https://github.com/Card-Forge/forge/wiki/ai). In principle the construction AI could construct a deck using oracle bulk data from Scryfall (https://scryfall.com/docs/api/bulk-data) and then simulate games using the Forge built in AI while receiving rewards for its win/loss statistics. It's not as interesting an approach as the original proposal (because the information isn't integrated beyond the win/loss reward signal) but as a minimum proof of concept it can show that interesting decks can emerge through reinforcement learning.

## Testing With JPype

JPype (https://jpype.readthedocs.io/en/latest/userguide.html#introduction) lets you use Java from within Python, thereby simplifying the whole process of interfacing a Python-based AI agent with a Java based game with a Magic: The Gathering rules engine (https://github.com/magefree/mage). In particular, the hope is that it allows one to create a lightweight and headless Gym Environment (https://github.com/openai/gym) for Magic: The Gathering.

### Usage

```
git clone https://github.com/magefree/mage
cd mage
mvn install
cd ..
python3.10 test.py
```

## Proposal

A proposal design for a Magic: The Gathering bot that constructs interesting decks.

The agent is to select cards from a pool and then play them against another such agent so as to maximize P(winning).

The policy network would be 2-headed (1 head for selecting cards from a pool to add to the deck, given the cards already in the deck, and 1 head for selecting the current best action, given the current game state) and 2-tailed (1 tail for receiving the pool of cards as input and 1 tail for receiving the current game state).

Pictorially it would look like this: >---< where the ">" are the tails and the "<" are the heads. Inference flows to the right (--->) gradients flow to the left (<---). "---" in the middle is a shared representation which ensures that information about card selection is not entirely separate from information about playing.

Since game states are naturally variable length in Magic (e.g. hand-size isn't fixed, target options aren't fixed) a sequence based acausal transformer is suggested for the playing tail.

For simplicity, it is proposed that the network initially learns via Q-learning where the reward is 1 for winning and -1 for losing and 0 otherwise.

Deep Q-learning typically looks like this where Q here would be the >---< network:

<img width="744" height="389" alt="Screenshot 2568-08-21 at 11 04 06" src="https://github.com/user-attachments/assets/eaed3f12-e960-4fec-b902-d8d0939cd69f" />

(Image from https://arxiv.org/pdf/1312.5602)

It is suggested that such an approach would yield interesting decks provided the MTG gym environment is sufficiently fast (<1 second for a game with random actions) and the network is sufficiently lightweight (<1 second to perform an action).

An important subtlety is that the win/loss rewards also need to passed to the card pool selection head, so that it may learn.

The skeleton code for >---< might look something like:

```py
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.tail_construct = Block() # acausal transformer block where the sequence is the cards already added to the deck
        self.tail_play = Block() # acausal transformer block where the sequence is the game objects (e.g. hand 1, hand 2, ..., board 1, board 2, ...)
        self.shared = nn.Linear()
        self.head_construct = nn.Linear() # output = a selected card to add to the deck
        self.head_play = nn.Linear() # output = a selected action to perform next in the game
    def forward_construct(self, x):
        x = self.tail_construct(x)
        x = self.shared(x)
        x = self.head_construct(x)
        return x
    def forward_play(self, x):
        x = self.tail_play(x)
        x = self.shared(x)
        x = self.head_play(x)
        return x
```

More on the subtltety: from the perspective of the head_construct, the moment it chooses the last card for the deck it receives a win/loss reward that it is supposed to predict with (and it takes win-maximizing actions during deck construction).

For those unfamiliar with Q-learning: the Qnet produces Q-values which you can think of as the quality associated with each action. Thus it learns to estimate the quality of possible actions and during inference you just take the max over these qualities to determine the best action.

Another subtlety: actions may be naturally variable length in Magic but you will notice that we use a fixed length head_play output. There's no way around this -- actions have to be capped (say 1000-10000 possible actions) and if the game state is such that there are >10000 targets then the game can be considered invalid and notify the experimenter.
