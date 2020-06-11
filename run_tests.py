import sys
import pickle
import matplotlib.pyplot as plt
import Environment
import Player
import State
from Model import Model

ROUNDS = 10000

env = Environment.Environment(3, 3)

prefix1 = input("Input the prefix for the first model to be loaded: ")
model_name1 = input("Input the name of the model you want: ") if len(sys.argv) != 2 else sys.argv[1]
print("Loading model with filename {}{} . . .\n".format(prefix1, model_name1))

player1 = Player.Player('player1', env, 1, 0)
player1.load_policy(model_name1, prefix1)

prefix2 = input("Input the prefix for the second model to be loaded: ")
model_name2 = input("Input the name of the model you want: ") if len(sys.argv) != 2 else sys.argv[2]
print("Loading model with filename {}{} . . .\n".format(prefix2, model_name2))

player2 = Player.Player('player2', env, -1, 0)
player2.load_policy(model_name2, prefix2)

players = [player1, player2]

for i in range(ROUNDS):
    while True:
        result = None
        for player in players:
            game_over = False
            # choose action
            action = player.choose_action(env.state, env.turn)
            
            # take action
            next_state, result = env.update(action)
            
            if result != None:
                # the game is over here
                game_over = True
                break
            
        if game_over:
            # dispense rewards and update values
            if result == 0:
                player1.draw += 1
                player2.draw += 1
                
            elif result == 1 or result == -1:
                # if result is 1, p1 wins
                # if result is -1, p2 wins
                
                winner = player1 if result == 1 else player2
                loser  = player1 if result == -1 else player2

                winner.win += 1
                loser.losses += 1
            
            else:
                # this is where someone made an illegal move
                winner = player1 if result == -10 else player2
                loser = player1 if result == 10 else player2
                
                loser.losses += 1
            
            break
        
    if i % (ROUNDS / 10) == 0:
        print("{}% complete. Current score: {}/{}/{}".format(i / ROUNDS * 100, player1.win, player1.draw, player2.win))

print("\n{}{} played {}{}".format(prefix1, model_name1, prefix2, model_name2))
print("\nFinal score: {}/{}/{}".format(player1.win, player1.draw, player2.win))
print("{}{} losses: {}".format(prefix1, model_name1, player1.losses))
print("{}{} losses: {}".format(prefix2, model_name2, player2.losses))