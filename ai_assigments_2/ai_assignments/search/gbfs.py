import math
from .. problem import Problem
from .. datastructures.priority_queue import PriorityQueue


# please ignore this
def get_solver_mapping():
    return dict(
        gbfs_ec=GBFS_Euclidean,
        gbfs_mh=GBFS_Manhattan
    )


class GBFS(object):
    # TODO, Exercise 2:
    # - implement Greedy Best First Search (GBFS)
    # - use the provided PriorityQueue where appropriate
    # - to put items into the PriorityQueue, use 'pq.put(<priority>, <item>)'
    # - to get items out of the PriorityQueue, use 'pq.get()'
    # - use a 'set()' to store nodes that were already visited
    def solve(self, problem: Problem):
        visited = set()
        queue = PriorityQueue()
        queue.put(self.heuristic(problem.get_start_node(), problem.get_end_node()), problem.get_start_node())

        current = problem.get_start_node()

        while not problem.is_end(current):

            successors = problem.successors(current)

            if current not in visited:
                visited.add(current)  # mark current element as visited
                for s in successors:
                    queue.put(self.heuristic(s, problem.get_end_node()), s)  # put successor into queue if it has not been visited

            if queue.has_elements():  # check if queue has elements left
                current = queue.get()

        return current


# this is the GBFS variant with the euclidean distance as a heuristic
# it is registered as a solver with the name 'gbfs_ec'

# please note that in an ideal world, this heuristic should actually be part
# of the problem definition, as it assumes domain knowledge about the structure
# of the problem, and defines a distance to the goal state
class GBFS_Euclidean(GBFS):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.sqrt((cy - gy) ** 2 + (cx - gx) ** 2)


# this is the GBFS variant with the manhattan distance as a heuristic
# it is registered as a solver with the name 'gbfs_mh'

# please note that in an ideal world, this heuristic should actually be part
# of the problem definition, as it assumes domain knowledge about the structure
# of the problem, and defines a distance to the goal state
class GBFS_Manhattan(GBFS):
    def heuristic(self, current, goal):
        cy, cx = current.state
        gy, gx = goal.state
        return math.fabs((cy - gy)) + math.fabs(cx - gx)
