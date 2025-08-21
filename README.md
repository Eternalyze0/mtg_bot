# Magic: The Gathering Bot

A proposal design for a Magic: The Gathering bot that constructs interesting decks.

The agent is to select cards from a pool and then play them against another such agent so as to maximize P(winning).

The policy network would be 2-headed (1 head for selecting cards from a pool to add to the deck, given the cards already in the deck, and 1 head for selecting the current best action, given the current game state) and 2-tailed (1 tail for receiving the pool of cards as input and 1 tail for receiving the current game state).

Pictorially it would look like this: >---< where the ">" are the tails and the "<" are the heads. Inference flows to the right (--->) gradients flow to the left (<---). "---" in the middle is a shared representation which ensures that information about card selection is not entirely separate from information about playing.

Since game states are naturally variable length in Magic (e.g. hand-size isn't fixed, target options aren't fixed) a sequence based acausal transformer is suggested for the playing tail.
