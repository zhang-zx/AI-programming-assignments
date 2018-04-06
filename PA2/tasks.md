# Multi-Agent Pacman

- http://ai.berkeley.edu/multiagent.html
- https://github.com/jasonwu0731/AI-Pacman/blob/master/Pacman/hw2-multiagent/multiAgents.py
- https://github.com/jasonwu0731/AI-Pacman/tree/master/Pacman/hw2-multiagent

## 测试
- python autograder.py
- python autograder.py -q q2

## 基本任务

- ReflexAgent
    - 改进multiAgents.py 中的 ReflexAgent
    - 使用python pacman.py -p ReﬂexAgent -l openClassic -n 10 -q 进行测试
    - python pacman.py -p ReflexAgent -l testClassic
    - python pacman.py --frameTime 0 -p ReflexAgent -k 1
    - python pacman.py --frameTime 0 -p ReflexAgent -k 2
    - python autograder.py -q q1
- MinimaxAgent
    - 在multiAgents.py中的MinimaxAgent中完成minimax算法
    - 目标为可以使用任意数量的幽灵
    - 使用python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4进行测试
    - python autograder.py -q q2
- AlphaBetaAgent
    - 在multiAgents.py中的AlphaBetaAgent中完成alpha-beta剪枝算法
    - 使用python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic 进行测试
- ExpectimaxAgent
    - 完成ExpectimaxAgent类
    - 使用python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10测试
    - 使用python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10测试

## 加分

- 更好的评估函数
    - 在betterEvaluationFunction写一个更好的评估函数
    - 使用python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 10测试