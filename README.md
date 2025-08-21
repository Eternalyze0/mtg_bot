# Magic: The Gathering Bot

A proposal design for a Magic: The Gathering bot that constructs interesting decks.

The agent is to select cards from a pool and then play them against another such agent so as to maximize P(winning).

The policy network would be 2-headed (1 head for selecting cards from a pool to add to the deck, given the cards already in the deck, and 1 head for selecting the current best action, given the current game state) and 2-tailed (1 tail for receiving the pool of cards as input and 1 tail for receiving the current game state).

Pictorially it would look like this: >---< where the ">" are the tails and the "<" are the heads. Inference flows to the right (--->) gradients flow to the left (<---). "---" in the middle is a shared representation which ensures that information about card selection is not entirely separate from information about playing.

Since game states are naturally variable length in Magic (e.g. hand-size isn't fixed, target options aren't fixed) a sequence based acausal transformer is suggested for the playing tail.

For simplicity, it is proposed that the network initially learns via Q-learning where the reward is 1 for winning and -1 for losing and 0 otherwise.

Deep Q-learning typically looks like this where Q here would be the >---< network:

<img width="744" height="389" alt="Screenshot 2568-08-21 at 11 04 06" src="https://github.com/user-attachments/assets/eaed3f12-e960-4fec-b902-d8d0939cd69f" />

It is suggested that such an approach would yield interesting decks provided the MTG gym environment is sufficiently fast (<1 second for a game with random actions) and the network is sufficiently lightweight (<1 second to perform an action).
