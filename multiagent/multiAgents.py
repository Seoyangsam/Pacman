# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# import libraries
import random
import util
from game import Agent
from pacman import GameState


def evaluationFunction(currentGameState: GameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (task 1)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _rmStop(List):
            return [x for x in List if x != 'Stop']

        def _miniMax(s, iterCount):
            if iterCount >= self.depth * numAgent or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iterCount % numAgent != 0:  # Ghost min
                result = 1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = min(result, _miniMax(sdot, iterCount + 1))
                return result
            else:  # Pacman Max
                result = -1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = max(result, _miniMax(sdot, iterCount + 1))
                    if iterCount == 0:
                        ActionScore.append(result)
                return result

        _miniMax(gameState, 0)
        # print _rmStop(gameState.getLegalActions(0)), ActionScore
        return _rmStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (task 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _rmStop(List):
            return [x for x in List if x != 'Stop']

        def _alphaBeta(s, iterCount, alpha, beta):
            if iterCount >= self.depth * numAgent or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iterCount % numAgent != 0:  # Ghost min
                result = 1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = min(result, _alphaBeta(sdot, iterCount + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
                return result
            else:  # Pacman Max
                result = -1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = max(result, _alphaBeta(sdot, iterCount + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if iterCount == 0:
                        ActionScore.append(result)
                    if beta < alpha:
                        break
                return result

        _alphaBeta(gameState, 0, -1e20, 1e20)
        return _rmStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (task 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _rmStop(List):
            return [x for x in List if x != 'Stop']

        def _expectMinimax(s, iterCount):
            if iterCount >= self.depth * numAgent or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iterCount % numAgent != 0:  # Ghost min
                successorScore = []
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = _expectMinimax(sdot, iterCount + 1)
                    successorScore.append(result)
                averageScore = sum([float(x) / len(successorScore) for x in successorScore])
                return averageScore
            else:  # Pacman Max
                result = -1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = max(result, _expectMinimax(sdot, iterCount + 1))
                    if iterCount == 0:
                        ActionScore.append(result)
                return result

        _expectMinimax(gameState, 0)
        return _rmStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]
