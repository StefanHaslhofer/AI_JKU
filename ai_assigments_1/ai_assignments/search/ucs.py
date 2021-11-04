from .. problem import Problem
from .. datastructures.priority_queue import PriorityQueue


def get_solver_mapping():
    return dict(ucs=UCS)


class UCS(object):
    # TODO, excercise 1:
    # - implement Uniform Cost Search (UCS), a variant of Dijkstra's Graph Search
    # - use the provided PriorityQueue where appropriate
    # - to put items into the PriorityQueue, use 'pq.put(<priority>, <item>)'
    # - to get items out of the PriorityQueue, use 'pq.get()'
    # - store visited nodes in a 'set()'
    def solve(self, problem: Problem):
        visited = set()
        queue = PriorityQueue()
        queue.put(0, problem.get_start_node())

        current = queue.get()

        while not problem.is_end(current):

            successors = problem.successors(current)

            if current not in visited:
                visited.add(current)  # mark current element as visited
                for s in successors:
                    queue.put(s.cost, s)  # put successor into queue if it has not been visited

            if queue.has_elements():  # check if queue has elements left
                current = queue.get()

        return current
