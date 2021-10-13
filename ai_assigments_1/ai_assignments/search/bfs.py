from .. problem import Problem
from .. datastructures.queue import Queue


# please ignore this
def get_solver_mapping():
    return dict(bfs=BFS)


class BFS(object):
    # TODO, exercise 1:
    # - implement Breadth First Search (BFS)
    # - use 'problem.get_start_node()' to get the node with the start state
    # - use 'problem.is_end(node)' to check whether 'node' is the node with the end state
    # - use a set() to store already visited nodes
    # - use the 'queue' datastructure that is already imported as the 'fringe'/ the 'frontier'
    # - use 'problem.successors(node)' to get a list of nodes containing successor states
    def solve(self, problem: Problem):
        visited = set()
        queue = Queue()
        queue.put(problem.get_start_node())

        current = queue.get()

        while not problem.is_end(current):

            successors = problem.successors(current)

            if current not in visited:
                visited.add(current)  # mark current element as visited
                for s in successors:
                    queue.put(s)    # put successor into queue if it has not been visited

            if queue.has_elements():  # check if queue has elements left
                current = queue.get()

        return current
