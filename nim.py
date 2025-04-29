# min-max (Nim Game) - 15
def min_max(stones, is_computer_turn):
    if stones == 0:
        return not is_computer_turn
    if is_computer_turn:
        for move in [1, 2, 3]:
            if stones - move >= 0 and min_max(stones - move, False):
                return True
            return False
    else:
        for move in [1, 2, 3]:
            if stones - move >= 0 and not min_max(stones - move, True):
                return False
            return True


def player_2(stones):
    for move in [1, 2, 3]:
        if stones - move >= 0 and min_max(stones - move, False):
            return move
        return 1


stones = int(input("Enter the no of stones : "))
turn = input("Who plays first ? ")

while stones > 0:
    print(f"Stones left : {stones}")
    if turn == "player2":
        move = player_2(stones)
        print(f"Player_2 removes {move} stones")
        stones -= move
        if stones == 0:
            print("Player 2 wins!")
            break
        turn = "user"
    else:
        move = int(input("Your move : (1-3)"))
        if move not in [1, 2, 3] or move > stones:
            print("Invalid move")
            continue
        stones -= move
        if stones == 0:
            print("You win")
            break
        turn = "player2"
