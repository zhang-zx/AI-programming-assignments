#!/usr/bin/env python

def depthFirstSearch(problem):
    if(problem.isGoalState(problem.getStartState())): return []
    frontier = util.Stack()
    visited = list()
    visited.append(problem.getStartState())
    frontier.push( (problem.getStartState(), []) )
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        for next_state in problem.getSuccessors(state):
            n_state = next_state[0]
            n_direction = next_state[1]
            if problem.isGoalState(n_state):
                return actions + [n_direction]
            if n_state not in visited:
                frontier.push( (n_state, actions + [n_direction]) )
                visited.append( n_state )
    return []