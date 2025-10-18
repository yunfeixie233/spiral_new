import re, random
from typing import Optional, Dict, Tuple, List, Any

import textarena as ta

class StrategoEnv(ta.Env):
    """ A two-player implementation of the board game Stratego """
    def __init__(self):
        """
        Initialize the environment.
        """
        ## set up the board items
        self.piece_counts = {
            'Flag': 1, 'Bomb': 6, 'Spy': 1, 'Scout': 8, 'Miner': 5,
            'Sergeant': 4, 'Lieutenant': 4, 'Captain': 4, 'Major': 3,
            'Colonel': 2, 'General': 1, 'Marshal': 1
        }
        self.piece_ranks = {
            'Flag': 0, 'Bomb': 11, 'Spy': 1, 'Scout': 2, 'Miner': 3,
            'Sergeant': 4, 'Lieutenant': 5, 'Captain': 6, 'Major': 7,
            'Colonel': 8, 'General': 9, 'Marshal': 10
        }
        self.lakes = [(4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)]
        self.player_pieces = {0: [], 1: []}
        self.board = [[None for _ in range(10)] for _ in range(10)]

    @property
    def terminal_render_keys(self):
        return ["rendered_board"]

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the environment to start a new game """
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        
        ## populate the board
        self.board = self._populate_board()

        ## initialise the game state
        rendered_board = self._render_board(player_id=None, full_board=True)
        game_state={"board": self.board, "player_pieces": self.player_pieces, "rendered_board": rendered_board}
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        self._observe_current_state()
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]):
        """
        Generates the player prompt for the current player.

        Args:
            player_id (int): The ID of the current player.
            game_state (Dict[str, Any]): The current game state.
        """
        prompt = (
            f"You are Player {player_id} in Stratego.\n"
            "Your goal is to capture your opponent's Flag or eliminate all of their movable pieces.\n"
            "Your army has been placed for you on the board, including your Flag, Bombs, and other pieces of varying ranks.\n"
            "\n"
            "### Gameplay Instructions\n"
            "1. **Movement Rules:**\n"
            "   - On your turn, you can move one piece by one step to an adjacent square (up, down, left, or right) that is already occupied with your pieces.\n"
            "   - Example: A piece can move from A1 to B1 or A1 to A2 if B1 and A2 are not placed with the player's own pieces.\n"
            "   - If the selected piece is a Bomb or a Flag, it cannot be moved.\n"
            # "   - **Scout Movement:** Scouts, on the other hand, can move multiple steps in a straight line (horizontally or vertically), but strictly only on one condition.\n"
            # "       - The condition is that Scouts cannot jump over any piece (your own or your opponent's).\n"
            # "       - Example: If there is a piece between the Scout and its destination, the Scout cannot move to the destination.\n"
            # "       - This will be indicated as an invalid move which makes you lose the game.\n"
            "2. **Battles:**\n"
            "   - If you move onto a square occupied by an opponent's piece, then a battle will occur:\n"
            "     - The piece with the higher rank wins and eliminates the opponent's piece.\n"
            "     - If the ranks are equal, both pieces are removed from the board.\n"
            "     - **Special Cases:**\n"
            "       - Bombs eliminate most attacking pieces except Miners, which defuse Bombs.\n"
            "       - Spies can defeat the Marshal if the Spy attacks first but lose to all other pieces.\n"
            "3. **Strategic Goals:**\n"
            "   - Identify your opponent's pieces through their movements and battles.\n"
            "   - Protect your Flag while attempting to capture your opponent's Flag.\n"
            "   - Use Scouts strategically to gain information about your opponent's pieces and attack weak ones.\n"
            "\n"
            "### How to Make a Move:\n"
            "1. Specify the coordinates of the piece you want to move and its destination.\n"
            "2. Use the format: [A0 B0], where A0 is the source position, and B0 is the destination.\n"
            "   - Example: To move a piece from row 0, column 0 to row 1, column 0, input [A0 B0].\n"
            "3. Ensure the destination is valid according to the movement rules above.\n"
            "\n"
            "### Important Notes:\n"
            "- The board will show your pieces and their positions, e.g. MN, MS.\n"
            "- The board will also show known positions of your opponent's pieces without revealing their ranks, e.g. ?.\n"
            "- Grids with ~ are lakes and cannot be moved onto.\n"
            "- As a suggestion, start your game by moving your pieces that are on the front lines to gain information about your opponent's pieces. Player 0 and player 1's frontlines are row D and G respectively.\n"
            "\n"
            "Here is the current board state:\n"
        )
        return prompt

    def _observe_current_state(self):
        """
        Observe the current state of the game and update the state with the rendered board
        and gives the available moves for the current player.
        """
        player_id = self.state.current_player_id
        available_moves = []

        for row in range(10):
            for col in range(10):
                piece = self.board[row][col]
                if isinstance(piece, dict) and piece['player'] == player_id:
                    # Skip immovable pieces
                    if piece['rank'].lower() in ['bomb', 'flag']:
                        continue

                    # Check if this is a scout (can move multiple squares)
                    is_scout = piece['rank'].lower() == 'scout'
                    
                    # Check all four directions
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if is_scout:
                            # Scout can move multiple squares in this direction
                            distance = 1
                            while True:
                                new_row = row + (dr * distance)
                                new_col = col + (dc * distance)
                                
                                # Check if still within board bounds
                                if not (0 <= new_row < 10 and 0 <= new_col < 10):
                                    break
                                
                                target = self.board[new_row][new_col]
                                
                                if target is None:
                                    # Empty square - scout can move here and continue
                                    available_moves.append(f"[{chr(row + 65)}{col} {chr(new_row + 65)}{new_col}]")
                                    distance += 1
                                elif isinstance(target, dict) and target['player'] != player_id:
                                    # Enemy piece - scout can attack but cannot continue past
                                    available_moves.append(f"[{chr(row + 65)}{col} {chr(new_row + 65)}{new_col}]")
                                    break
                                else:
                                    # Own piece or other obstacle - scout cannot move here or past
                                    break
                        else:
                            # Regular piece - can only move one square
                            new_row, new_col = row + dr, col + dc
                            if 0 <= new_row < 10 and 0 <= new_col < 10:
                                target = self.board[new_row][new_col]
                                if (target is None or
                                    (isinstance(target, dict) and target['player'] != player_id)):
                                    available_moves.append(f"[{chr(row + 65)}{col} {chr(new_row + 65)}{new_col}]")

        self.state.add_observation(
            message=f"Current Board:\n\n{self._render_board(player_id=player_id, full_board=False)}\nAvailable Moves: " + ", ".join(available_moves),
            observation_type=ta.ObservationType.GAME_BOARD
        )
    
    def _populate_board(self):
        """
        Populates the board with pieces for each player strategically.
        """
        for player in range(2):
            # Define rows for each player
            back_rows = range(0, 2) if player == 0 else range(8, 10)
            front_rows = range(2, 4) if player == 0 else range(7, 9)
            all_rows = range(0, 4) if player == 0 else range(6, 10)

            # Place the Flag strategically
            while True:
                row = random.choice(back_rows)
                col = random.randint(0, 9)
                if (row, col) not in self.lakes and self.board[row][col] is None:
                    self.board[row][col] = {'rank': 'Flag', 'player': player}
                    self.player_pieces[player].append((row, col))
                    flag_position = (row, col)
                    break

            # Place Bombs around the Flag if possible
            bombs_to_place = self.piece_counts['Bomb']
            bomb_positions = [
                (flag_position[0] + dr, flag_position[1] + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Adjacent cells
                if 0 <= flag_position[0] + dr < 10 and 0 <= flag_position[1] + dc < 10
            ]

            for pos in bomb_positions:
                if bombs_to_place > 0 and self.board[pos[0]][pos[1]] is None and pos not in self.lakes:
                    self.board[pos[0]][pos[1]] = {'rank': 'Bomb', 'player': player}
                    self.player_pieces[player].append(pos)
                    bombs_to_place -= 1

            # Place remaining Bombs at the frontline
            for _ in range(bombs_to_place):
                while True:
                    row = random.choice(front_rows)
                    col = random.randint(0, 9)
                    if self.board[row][col] is None and (row, col) not in self.lakes:
                        self.board[row][col] = {'rank': 'Bomb', 'player': player}
                        self.player_pieces[player].append((row, col))
                        break

            # Place other pieces randomly
            for piece, count in self.piece_counts.items():
                if piece in ['Flag', 'Bomb']:
                    continue  # Skip already placed pieces
                for _ in range(count):
                    while True:
                        row = random.choice(all_rows)
                        col = random.randint(0, 9)
                        if self.board[row][col] is None and (row, col) not in self.lakes:
                            self.board[row][col] = {'rank': piece, 'player': player}
                            self.player_pieces[player].append((row, col))
                            break

        # Place the lakes
        for row, col in self.lakes:
            self.board[row][col] = "~"

        return self.board


    
    def _render_board(self, player_id, full_board: bool = False):
        """
        Renders the board state with fixed-width formatting for uniform alignment.

        Args:
            player_id (int): The player viewing the board.
            full_board (bool): Whether to render the full board or just the visible pieces.
        """
        # Define abbreviations for each piece
        piece_abbreviations = {
            'Flag': 'FL', 'Bomb': 'BM', 'Spy': 'SP', 'Scout': 'SC', 'Miner': 'MN',
            'Sergeant': 'SG', 'Lieutenant': 'LT', 'Captain': 'CP', 'Major': 'MJ',
            'Colonel': 'CL', 'General': 'GN', 'Marshal': 'MS'
        }

        res = []
        column_headers = "   " + " ".join([f"{i:>3}" for i in range(10)])  # Align column numbers
        res.append(column_headers + "\n")

        for row in range(10):
            row_label = chr(row + 65)  # Convert row index to a letter (A, B, C, ...)
            row_render = [f"{row_label:<3}"]  # Add row label with fixed width
            for col in range(10):
                if (row, col) in self.lakes:
                    cell = "  ~ "  # Lakes
                elif self.board[row][col] is None:
                    cell = "  . "  # Empty space
                else:
                    piece = self.board[row][col]
                    abbreviation = piece_abbreviations[piece['rank']]
                    if full_board:
                        cell = f" {abbreviation.lower() if piece['player'] == 0 else abbreviation.upper()} "  # Full board view
                    elif piece['player'] == player_id:
                        displayed_piece = abbreviation.upper()
                        cell = f" {displayed_piece} "
                    else:
                        cell = "  ? "  # Hidden opponent piece
                row_render.append(cell)

            res.append("".join(row_render) + "\n")

        return "".join(res)



    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Execute an action in the environment """
        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        ## action search pattern
        action_search_pattern = re.compile(r"\[([A-J])([0-9]) ([A-J])([0-9])\]", re.IGNORECASE)
        match = action_search_pattern.search(action)

        if match is None:
            reason=f"Invalid action format. Player {player_id} did not input a move in the format [A0 B0]."
            self.state.set_invalid_move(reason=reason)
        
        else:
            src_row, src_col, dest_row, dest_col = match.groups()
            src_row, dest_row = src_row.upper(), dest_row.upper()
            source = f"{src_row}{src_col}"
            dest = f"{dest_row}{dest_col}"
            src_row, src_col = ord(src_row) - 65, int(src_col)
            dest_row, dest_col = ord(dest_row) - 65, int(dest_col)
             

            ## check if the source and destination are valid
            if self._validate_move(player_id=player_id, src_row=src_row, src_col=src_col, dest_row=dest_row, dest_col=dest_col):

                attacking_piece = self.board[src_row][src_col]
                target_piece = self.board[dest_row][dest_col]

                if target_piece is None:
                    ## move to an empty square
                    self.board[dest_row][dest_col] = attacking_piece
                    self.board[src_row][src_col] = None
                    self.player_pieces[player_id].remove((src_row, src_col))
                    self.player_pieces[player_id].append((dest_row, dest_col))
                    
                    ## add the observation to both players separately
                    message=f"You have moved your piece from {source} to {dest}."
                    self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                    message=f"Player {player_id} has moved a piece from {source} to {dest}."
                    self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                else:
                    ## battle
                    attacking_rank = self.piece_ranks[attacking_piece['rank']]
                    target_rank = self.piece_ranks[target_piece['rank']]
                    if attacking_rank == target_rank:
                        ## both pieces are removed
                        self.board[src_row][src_col] = None
                        self.board[dest_row][dest_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the ranks are the same, both pieces lost."
                        self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                        message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the ranks are the same, both pieces lost."
                        self.state.add_observation(from_id=-1, to_id=1 - player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                    elif target_piece['rank'] == 'Bomb':
                        if attacking_piece['rank'] == 'Miner':
                            ## Miner defuses the bomb
                            self.board[dest_row][dest_col] = attacking_piece
                            self.board[src_row][src_col] = None
                            self.player_pieces[player_id].remove((src_row, src_col))
                            self.player_pieces[player_id].append((dest_row, dest_col))

                            ## add the observation to both players separately
                            message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As miners can defuse bombs, you won the battle."
                            self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                            message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As miners can defuse bombs, you lost the battle."
                            self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                        else:
                            ## attacking piece is destroyed
                            self.board[src_row][src_col] = None
                            self.player_pieces[player_id].remove((src_row, src_col))

                            ## add the observation to both players separately
                            message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is not a miner, you lost the battle."
                            self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                            message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is not a miner, you won the battle."
                            self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                    elif target_piece['rank'] == 'Flag':
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))
                        ## game over
                        self.state.set_winner(player_id=player_id,reason=[f"Player {player_id} has captured the opponent's flag!"])
                    elif attacking_piece['rank'] == 'Spy' and target_piece['rank'] == 'Marshal':
                        ## Spy beats Marshal only if spy attacks first
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a spy and the destination is a marshall, you won the battle."
                        self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                        message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a spy and the destination is a marshall, you lost the battle."
                        self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                    elif attacking_rank > target_rank:
                        ## attacker wins
                        self.board[dest_row][dest_col] = attacking_piece
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))
                        self.player_pieces[player_id].append((dest_row, dest_col))
                        self.player_pieces[1 - player_id].remove((dest_row, dest_col))

                        ## add the observation to both players separately
                        message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a higher rank than the destination, you won the battle."
                        self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                        message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a higher rank than the destination, you lost the battle."
                        self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                    else:
                        ## defender wins
                        self.board[src_row][src_col] = None
                        self.player_pieces[player_id].remove((src_row, src_col))

                        ## add the observation to both players separately
                        message=f"You have moved your piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a lower rank than the destination, you lost the battle."
                        self.state.add_observation(from_id=-1, to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

                        message=f"Player {player_id} has moved a piece from {source} to {dest}. The attacking piece was {attacking_piece['rank']} and the destination piece was {target_piece['rank']}. As the attacker is a lower rank than the destination, you won the battle."
                        self.state.add_observation(from_id=-1, to_id=1-player_id, message=message, observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)

        ## check if the game is over
        if self._check_winner():
            reason=f"Player {self._check_winner()} wins! Player {1 - self._check_winner()} has no more movable pieces."
            self.state.set_winner(player_id=self._check_winner(), reason=reason)

        ## update the rendered board
        self.state.game_state["rendered_board"] = self._render_board(player_id=player_id, full_board=True)

        result = self.state.step()
        self._observe_current_state()
        return result
    
    def _validate_move(self, player_id, src_row, src_col, dest_row, dest_col):
        """
        Validates the move based on the game rules.

        Args:
            player_id (int): The ID of the player making the move.
            src_row (int): The row of the source position.
            src_col (int): The column of the source position.
            dest_row (int): The row of the destination position.
            dest_col (int): The column of the destination position.
        """
        if not (0 <= src_row < 10 and 0 <= src_col < 10 and 0 <= dest_row < 10 and 0 <= dest_col < 10):
            reason=f"Invalid action format. Player {player_id} did not input valid coordinates."
            self.state.set_invalid_move(reason=reason)
            return False
        
        if self.board[src_row][src_col] is None or self.board[src_row][src_col]['player'] != player_id:
            reason=f"Invalid action format. Player {player_id} must move one of their own pieces."
            self.state.set_invalid_move(reason=reason)
            return False
        
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1 and self.board[src_row][src_col]['rank'].lower() == 'scout':
            ## check if there's a piece in between the source and destination
            if src_row == dest_row:
                for col in range(min(src_col, dest_col) + 1, max(src_col, dest_col)):
                    if self.board[src_row][col] is not None:
                        reason=f"Invalid action format. Player {player_id} cannot move a scout through other pieces."
                        self.state.set_invalid_move(reason=reason)
                        return False
            elif src_col == dest_col:
                for row in range(min(src_row, dest_row) + 1, max(src_row, dest_row)):
                    if self.board[row][src_col] is not None:
                        reason=f"Invalid action format. Player {player_id} cannot move a scout through other pieces."
                        self.state.set_invalid_move(reason=reason)
                        return False
            else:
                reason=f"Invalid action format. Player {player_id} cannot move a scout diagonally."
                self.state.set_invalid_move(reason=reason)
                return False
            
        if abs(src_row - dest_row) + abs(src_col - dest_col) != 1 and self.board[src_row][src_col]['rank'].lower() != 'scout':
            ## !  - by right, only scouts can move more than one square at a time but we are not implementing that yet
            reason=f"Invalid action format. Pieces, apart from scouts, can only move one square at a time."
            self.state.set_invalid_move(reason=reason)
            return False
        
        if self.board[dest_row][dest_col] is not None:
            if (dest_row, dest_col) in self.lakes:
                reason=f"Invalid action format. Player {player_id} cannot move into the lake."
                self.state.set_invalid_move(reason=reason)
                return False
            
            elif self.board[dest_row][dest_col]['player'] == player_id:
                reason=f"Invalid action format. Player {player_id} cannot move onto their own piece."
                self.state.set_invalid_move(reason=reason)
                return False
        
        if self.board[src_row][src_col]['rank'].lower() in ['bomb','flag']:
            reason=f"Invalid action format. Player {player_id} cannot move a bomb or flag."
            self.state.set_invalid_move(reason=reason)
            return False
        
        return True
    
    def _check_winner(self):
        """
        determine which player has no more pieces that are not bombs or flags.
        """
        for player in range(2):
            if all([self.board[row][col]['rank'] in ['Bomb', 'Flag'] for row, col in self.player_pieces[player]]):
                return 1 - player
        return None
    
