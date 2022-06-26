import argparse
import copy
import numpy
import typing
import tqdm


class Board(object):
    '''
    The classic 3 by 3 Tic Tac Toe board interface implementation, which is
    used as a part of Reinforcement Learning environment. It provides the
    necessary routine methods for accessing the internal states and allows
    safely modifying it while maintaining a valid state.

    Cell coordinates are zero-based indices: (x, y). Top left cell's
    coordinates are (0, 0), bottom right - (2, 2), i.e. the whole board looks
    like this:

     (0, 0) | (0, 1) | (0, 2)
    --------------------------
     (1, 0) | (1, 1) | (1, 2)
    --------------------------
     (2, 0) | (2, 1) | (2, 2)
    '''

    def __init__(self, cells: numpy.array = None) -> None:
        # Use classic 3x3 size.
        self.size: int = 3
        self.first_player_turn: bool = True
        self.cells: numpy.array
        if cells is not None:
            assert (cells.shape == (self.size, self.size))
            self.cells = cells
        else:
            self.cells = numpy.zeros((self.size, self.size), dtype=numpy.int8)

    def take_turn(self, cell: typing.Tuple[int, int]):
        '''
        Modifies current board given player's decision.

        Expects given cell to be empty, otherwise produces an exception.
        '''
        assert (self.is_possible(cell))
        player_identifier = 1
        if not self.first_player_turn:
            player_identifier = -1
        self.cells[cell] = player_identifier
        # Switch current player after the turn.
        self.first_player_turn = not self.first_player_turn

    def is_possible(self, action: typing.Tuple[int, int]) -> bool:
        '''
        Checks whether an action is valid on this board.

        Args:
            action: Coordinates of the action to check for validity.

        Returns:
            bool: True if it is possible to put 'X' or 'O' into the given cell,
                False otherwise.
        '''
        return self.cells[action] == 0

    def possible_actions(self) -> numpy.array:
        '''
        Outputs a all possible actions from current board state by choosing the
        ones not previously taken by either player.

        Returns:
            numpy.array: An array of possible actions.
        '''
        return numpy.array([(i, j)
                            for i in range(self.size)
                            for j in range(self.size)
                            if self.is_possible((i, j))])

    def is_over(self) -> typing.Tuple[bool, int]:
        '''
        Determines whether the game is over and hence no possible further
        action can be taken by either side.

        Returns:
            bool: True if the game is over, False otherwise.
            int: If the game is over, returns identifier of the winner (1 or
                -1 for the first and the second player respectively), 0
                otherwise.
        '''
        # Check for all horizontal sequences of 3 consequent non-empty cells
        for i in range(self.size):
            OK = True
            player_id = self.cells[i][0]
            if player_id == 0:
                continue
            for j in range(self.size):
                if self.cells[i][j] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Vertical sequences
        for i in range(self.size):
            OK = True
            player_id = self.cells[0][i]
            if player_id == 0:
                continue
            for j in range(self.size):
                if self.cells[j][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Diagonal: left top to right bottom
        OK = True
        player_id = self.cells[0][0]
        if player_id != 0:
            for i in range(self.size):
                if self.cells[i][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Diagonal: left bottom to right top
        OK = True
        player_id = self.cells[self.size - 1][0]
        if player_id != 0:
            for i in range(self.size):
                if self.cells[self.size - i - 1][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # If there is an empty cell, the game is not over yet.
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] == 0:
                    return False, 0
        # Otherwise all cells are taken and no player has won: it's a draw!
        return True, 0

    def hash(self) -> int:
        '''
        Bijectively maps board state to its unique identifier.

        Returns:
            int: Unique identifier of the current Board state.
        '''
        result = 0
        for i in range(self.size):
            for j in range(self.size):
                result *= 3
                result += self.cells[i][j] % 3
        return result

    def __repr__(self) -> str:
        '''
        Returns the Tic Tac Toe board in a human-readable representation using
        the following form (indices are replaced with 'X's, 'O's and
        whitespaces for empty cells):

         0 | 1 | 2
        -----------
         3 | 4 | 5
        -----------
         6 | 7 | 8
        '''
        result = ''
        mapping = [' ', 'X', 'O']
        for i in range(self.size):
            for j in range(self.size):
                result += ' {} '.format(mapping[self.cells[i][j]])
                if j != self.size - 1:
                    result += '|'
                else:
                    result += '\n'
            if i != self.size - 1:
                result += ('-' * (2 + self.size * self.size)) + '\n'
        return result


def get_all_states() -> typing.Tuple[typing.Set, typing.Set]:
    '''
    Devises all valid board states and computes hashes for each of them. Also
    extracts terminal states useful for the update rule simplification.

    Returns:
        set: A set of all possible boards' hashes.
        set: A set of hashes of all boards after a final turn, i.e. terminal
            boards.
    '''
    boards = [Board()]
    states = set()
    terminal_states = set()
    epoch = 0
    while boards:
        print(f'Epoch: {epoch}')
        epoch += 1
        next_generation = []
        for board in boards:
            board_hash = board.hash()
            if board_hash in states:
                continue
            states.add(board_hash)
            over, _ = board.is_over()
            if over:
                terminal_states.add(board_hash)
                continue
            for action in board.possible_actions():
                next_board = copy.deepcopy(board)
                next_board.take_turn(tuple(action))
                next_generation.append(next_board)
        boards = next_generation
    return states, terminal_states


class TicTacToe(object):
    '''
    TicTacToe is a Reinforcement Learning environment for this game, which
    reacts to players' moves, updates the internal state (Board) and samples
    reward.
    '''

    def __init__(self):
        self.board: Board = Board()

    def step(self,
             action: typing.Tuple[int, int]) -> typing.Tuple[int, Board, bool]:
        '''
        Updates the board given a valid action of the current player.

        Args:
            action: A valid action in a form of cell coordinates.

        Returns:
            int: Reward for the first player.
            Board: Resulting state.
            bool: True if the game is over, False otherwise.
        '''
        over, _ = self.board.is_over()
        assert (self.board.is_possible(action))
        assert (not over)
        self.board.take_turn(action)
        over, winner = self.board.is_over()
        return winner, self.board, over

    def __repr__(self):
        '''
        Returns current board state using a human-readable string
        representation.
        '''
        return self.board.__repr__()

    def reset(self):
        '''
        Empties the board and starts a new game.
        '''
        self.__init__()


class TDAgent(object):
    '''
    Tic Tac Toe-specific Temporal Difference [TD(0)] agent implementation.

    TODO(omtcvxyz): Allow saving and loading value estimates to omit training
    and skip to the interactive session for simplicity.
    '''

    def __init__(self,
                 environment: TicTacToe,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.1) -> None:
        self.environment: TicTacToe = environment
        self.learning_rate: float = learning_rate
        self.exploration_rate: float = exploration_rate
        '''
        TODO(omtcvxyz): Use get_all_states() to allocate memory only for the
        possible states instead of taking space for all possible combinations
        of 9 integers within [0; 2] range. Given that the terminal states are
        known beforehand, these should be also marked beforehand.
        '''
        self.value: numpy.array = numpy.zeros(3 **
            (self.environment.board.size ** 2 + 1))

    def reset_exploration_rate(self):
        '''
        Sets exploration rate to 0. This is useful whenever one would like to
        evaluate the agent's performance.
        '''
        self.exploration_rate = 0

    def consume_experience(self, initial_state: int, reward: int,
                           resulting_state: int, terminal: bool):
        '''
        This code uses formulation from the RL Book, Chapter 1. Although in
        general the TD update rule looks like this:
            V(S) = V(S) + \alpha * [R_t + V(S') - V(S)]
        The environment only samples reward on episode completion and hence the
        value function of terminal states could be set to the sampled reward,
        which would produce the following update rule (as proposed in Chapter 1
        of the RL Book):
            V(S) = V(S) + \alpha * [V(S') - V(S)]
        Which is exactly the same as the one used before if we augment it with
        the prior knowledge of the Tic Tac Toe environment.
        '''
        if terminal:
            self.value[resulting_state] = reward
        self.value[initial_state] += self.learning_rate * (
            self.value[resulting_state] - self.value[initial_state])

    def sample_action(
            self) -> typing.Tuple[typing.Tuple[int, int], bool]:
        '''
        Outputs an action leading to the state with the greatest value with
        probability 1 - self.exploration_rate. Samples random valid action
        with probability self.exploration_rate.


        Returns:
            (int, int): Sampled action.
            bool: True if the sampled action is a result of a "greedy"
                transition, i.e. whether sampled action is not exploratory.
        '''
        possible_actions = self.environment.board.possible_actions()
        if numpy.random.binomial(1, self.exploration_rate):
            random_index = numpy.random.randint(0, len(possible_actions))
            return tuple(possible_actions[random_index]), False
        board_copies = [
            copy.deepcopy(self.environment.board) for _ in possible_actions
        ]
        for action, board in zip(possible_actions, board_copies):
            board.take_turn(tuple(action))
        hashes = [board.hash() for board in board_copies]
        best_state = numpy.argmax(self.value[hashes])
        return tuple(possible_actions[best_state]), True


def learn(episodes_count: int, learning_rate: float,
          verbose: bool) -> typing.Tuple[TDAgent, TDAgent]:
    '''
    Feeds experience generated during Tic Tac Toe games between two similar
    TD(0) agents to these agents while improving their policies.

    Args:
        episodes_count: Samples experience from episodes_count episodes. The
            more experience the agents have, the better learned policies are.
            Approximate rate of running simulations is ~400 games / second
            (16 Gb RAM, Intel Core i7 processor setup).
        learning_rate: Refers to \alpha TD(0) algorithm hyperparameter. The
            more the faster learning process is, but it also becomes less
            "sensetive". Optimally, learning_rate should slowly decrease to a
            very small value over time.
        verbose: Indicates whether progress is shown. tqdm is used for
            convenient terminal experience.

    Returns:
        (TDAgent, TDAgent): Temporal Difference agents trained to play as the
            first and the second player respectively.
    '''
    if verbose:
        print('Training Temporal Difference AI.')
    environment: TicTacToe = TicTacToe()
    first_player: TDAgent = TDAgent(environment, learning_rate)
    second_player: TDAgent = TDAgent(environment, learning_rate)
    episodes = range(episodes_count)
    if verbose:
        episodes = tqdm.tqdm(episodes)
    for episode in episodes:
        first_player_turn: bool = True
        while True:
            if first_player_turn:
                action, greedy = first_player.sample_action()
            else:
                action, greedy = second_player.sample_action()
            first_player_turn = not first_player_turn
            previous_state = environment.board.hash()
            reward, _, over = environment.step(action)
            current_state = environment.board.hash()
            # Don't perform in case the last transition was exploratory.
            if greedy:
                first_player.consume_experience(previous_state, reward,
                                                current_state, over)
                # Second player consumes inverted reward, because it is sampled
                # for the first player.
                second_player.consume_experience(previous_state, -reward,
                                                 current_state, over)
            if over:
                environment.reset()
                break
    return first_player, second_player


def launch_interactive_session(AI: TDAgent, take_first_turn: bool):
    '''
    Launches continuous interactive session, in which human player can
    challenge an Reinforcement Learning agent previously trained using
    self-play.

    Args:
        TDAgent: The Reinforcement Learning agent, which faces the human
            player.
        take_first_turn: If True the human player will always take the first
            turn, the AI will always take the first turn otherwise.
    '''
    AI.reset_exploration_rate()
    environment: TicTacToe = AI.environment
    environment.reset()
    while True:
        print('Playing against AI')
        human_turn = take_first_turn
        print(environment.board)
        while True:
            if human_turn:
                print('Type coordinates (pair of 0-based space-separated '
                      'integers) of the cell you would like to take:')
                while True:
                    try:
                        x, y = map(int, input().split())
                        action: tuple = (x, y)
                        reward, _, over = environment.step(action)
                    except:
                        print('Sorry, the input is invalid. Try again.')
                        continue
                    else:
                        break
                print()
            else:
                action, greedy = AI.sample_action()
                # Learn while playing against human player.
                previous_state = environment.board.hash()
                reward, _, over = environment.step(action)
                current_state = environment.board.hash()
                if take_first_turn:
                    AI.consume_experience(previous_state, -reward,
                                          current_state, over)
                else:
                    AI.consume_experience(previous_state, reward,
                                          current_state, over)
            human_turn = not human_turn
            print(environment.board)
            if over:
                if (reward == 1
                        and take_first_turn) or (reward == -1
                                                 and not take_first_turn):
                    print('You won! Congratulations!')
                elif (reward == -1
                      and take_first_turn) or (reward == 1
                                               and not take_first_turn):
                    print('The AI won! Try again!')
                else:
                    print('It\'s a draw!')
                environment.reset()
                break
        print()
        answer = input('Would you like to play another game? (y/N) ')
        if answer.lower() != 'y' and answer.lower() != 'yes':
            break


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description='''This script implements Temporal Difference agent for
            the classic Tic Tac Toe environment and learns a policy by playing
            against itself. A human player can play against the trained agent
            upon the training completion. TD(0) parameters can be changed via
            command line arguments and options.''')
    parser.add_argument(
        '-v',
        '--verbose',
        help='increase output verbosity',
        action='store_true')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='step-size parameter (alpha); passed value '
        'should be within (0, 1] range [defaults to 0.1]')
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='unless this option is passed, interactive session against AI '
        'is launched after learning the policy')
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='fix numpy random seed for reproducibility [defaults to 42]')
    parser.add_argument(
        '--episodes',
        default=20000,
        type=int,
        help='train temporal difference AI agent for EPISODES games [defaults '
        'to 20000]')
    parser.add_argument(
        '--take_first_turn',
        action='store_true',
        help='always take the first turn; unless passed, human player will '
        'always take the second turn')
    arguments: argparse.Namespace = parser.parse_args()
    assert (0 < arguments.learning_rate and arguments.learning_rate <= 1)
    numpy.random.seed(arguments.seed)
    first_turn_AI, second_turn_AI = learn(
        arguments.episodes, arguments.learning_rate, arguments.verbose)
    if arguments.take_first_turn:
        AI = second_turn_AI
    else:
        AI = first_turn_AI
    if not arguments.no_interactive:
        launch_interactive_session(AI, arguments.take_first_turn)


if __name__ == '__main__':
    main()