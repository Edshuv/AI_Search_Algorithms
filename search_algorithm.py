# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        
        # # define index and tuple to hold a swap tuple
        # i = 0
        # temp_tuple = self.top()
        # ret = self.top()    # to return high priority value
        #
        # # swap last element with top one first and delete last element
        # self.queue[0] = self.queue[self.size() - 1]
        # self.queue.pop()
        #
        # # run while loop up to depth of the queue
        # while (2*i+1) < self.size():
        #     # check if right element (2i+1) is exsist
        #     if (2*i + 2) >= self.size():
        #         #check if the node key is even bigger and then swap
        #         if self.queue[i][0] >= self.queue[2*i+1][0]:
        #             temp_tuple = self.queue[2*i+1]
        #             self.queue[2*i+1] = self.queue[i]
        #             self.queue[i] = temp_tuple
        #             # reset i value to move down
        #             i = (2 * i) + 1
        #         else:
        #             break  # break while loop due to reach the bottom
        #     else:
        #         # compare children nodes which is smallest and do swap (if needed)
        #         if self.queue[2*i + 1][0] <= self.queue[2*i + 2][0]:
        #             if self.queue[i][0] < self.queue[2*i + 1][0]:
        #                 break  # no smaller key node can be found
        #             else:
        #                 temp_tuple = self.queue[2*i + 1]
        #                 self.queue[2*i + 1] = self.queue[i]
        #                 self.queue[i] = temp_tuple
        #                 # reset i value to move down
        #                 i = (2 * i) + 1
        #         else:
        #             if self.queue[i][0] < self.queue[2*i + 2][0]:
        #                 break  # no smaller key node can be found
        #             else:
        #                 temp_tuple = self.queue[2*i + 2]
        #                 self.queue[2*i + 2] = self.queue[i]
        #                 self.queue[i] = temp_tuple
        #                 # reset i value to move down
        #                 i = (2 * i) + 2

        ret = heapq.heappop(self.queue)
        # check FIFO (if left child equal to the top parents -> swap)
        if self.size() > 1:
            if self.queue[0][0] == self.queue[1][0]:
                temp_tuple = self.queue[1]
                self.queue[1] = self.queue[0]
                self.queue[0] = temp_tuple

        return ret
                    
            
#         raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        # prepare queue and node indexes for moving upward on the tree
        # heapq.heappush(self.queue,node)
        ind_n = 0
        self.queue.append(node)
        out = False

        # run while loop to move new node up
        ind_n = (self.size() - 1) - ind_n
        while not out:
            ind_q = int((ind_n - 1) / 2)
            if node[0] < self.queue[ind_q][0]:
                self.queue[ind_n] = self.queue[ind_q]
                self.queue[ind_q] = node
                ind_n = ind_q
            else:
                out = True
        
#         raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!

    path = [start]
    routes = [(start, path)]
    explored = []
    best_path = [(start, path)]

    if start == goal:
        return []

    while routes: # not empty no routs available
        # for node in routes:
        node = min(routes, key=lambda x: len(x[1]))
        # if found the goal store as the best path and delete node. If best path exist - compare and delete.
        if node[0][0] == goal:
            if best_path[0][0] == start:
                best_path = node
                routes.remove(node)
                continue
            elif len(node[1]) < len(best_path[1]):
                best_path = node
                routes.remove(node)
                continue
            else:
                routes.remove(node)
                continue
        if node[0] in explored:
            routes.remove(node)
            continue
        # check if all other routes longer and return best path
        if best_path[0][0] != start:
            # remove all path that would beat the best path
            temp1 = len(node[1])
            temp2 = len(best_path[1])
            if len(node[1]) >= (len(best_path[1]) - 1):
                routes.remove(node)
                continue
            # if there are no paths shorter than best -> retun best path
            if all(len(rt[1]) >= len(best_path[1]) for rt in routes):
                return best_path[1]
        front = graph[node[0]]
        front = sorted(front)
        if goal in front:
            for fr in front:
                if fr == goal:
                    best_path = (fr, node[1] + [fr])
                else:
                    explored.append(fr)
        else:
            for fr in front:
                if fr in node[1]:
                    continue
                else:
                    routes.append((fr, node[1] + [fr]))
        explored.append(node[0])
        routes.remove(node)

    return best_path[1]

    # raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    rq = PriorityQueue()
    explored = []
    routes = rq.queue
    node = (0, [start])
    rq.append(node)
    while routes:
        node = rq.pop()
        if node[1][-1] in explored:
            continue
        explored.append(node[1][-1])
        if node[1][-1] == goal:
            return node[1]
        front = graph[node[1][-1]]
        front = sorted(front)
        for nd in front:
            if nd not in explored:
                cost = node[0] + graph.get_edge_weight(node[1][-1], nd)
                rq.append((cost, node[1] + [nd]))




    # TODO: finish this function!
    # raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    # vx, vy = node coordinates; gx, gy = goal coordinates
    vx, vy = graph.nodes[v]['pos']
    gx, gy = graph.nodes[goal]['pos']
    x = gx - vx
    y = gy - vy
    h = math.sqrt((x * x) + (y * y))
    return h
    # raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    rq = PriorityQueue()
    explored = []
    routes = rq.queue
    node = (0, [start], 0)
    rq.append(node)
    while routes:
        node = rq.pop()
        if node[1][-1] in explored:
            continue
        explored.append(node[1][-1])
        if node[1][-1] == goal:
            return node[1]
        front = graph[node[1][-1]]
        front = sorted(front)
        for nd in front:
            if nd not in explored:
                cost = node[2] + graph.get_edge_weight(node[1][-1], nd)
                h = heuristic(graph, nd, goal)
                rq.append((cost + h, node[1] + [nd], cost))

    # if start == goal:
    #     return []
    # pt = PriorityQueue()
    # explored = []
    # node = (0, [start], 0)
    # explored.append(node[1][-1])
    # # pt.append(node)
    # check = pt.queue
    # while node:
    #     if node[1][-1] == goal:
    #         return node[1]
    #     front = graph[node[1][-1]]
    #     front = sorted(front)
    #     for fr in front:
    #         if fr in explored:
    #             continue
    #         weight = graph.get_edge_weight(node[1][-1], fr)
    #         weight = weight + node[2]
    #         h = heuristic(graph, fr, goal)
    #         # if h == 0:000000000000000000000000000000000000000000000000000
    #         #     return node[1]
    #         pt.append((weight + h, node[1] + [fr], weight))
    #     node = pt.pop()
    #     explored.append(node[1][-1])

    # raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!

    if start == goal:
        return []
    # se up for Start
    rqS  = PriorityQueue()
    exploredS = []
    routesS = rqS.queue
    nodeS = (0, [start])
    rqS.append(nodeS)
    sBest = ()

    if start=='s' and goal=='d':
        drr = 5

    #set up for Goal
    rqG = PriorityQueue()
    exploredG = []
    routesG = rqG.queue
    nodeG = (0, [goal])
    rqG.append(nodeG)
    gBest = ()

    while routesS and routesG:
        # implementation for Start
        nodeS = rqS.pop()
        if exploredS and nodeS[1][-1] in [n[-1] for _, n in exploredS]:
            continue
        # check node and return path and cost of node
        # if exploredG:
        gPath = [n for n in exploredG if n[1][-1] == nodeS[1][-1]]
        if gPath:
            retPath = gPath[0][1]
            retPath = retPath[::-1]  # reverse path to put goal at the end
            retPath = nodeS[1][:-1] + retPath  # ignore last node in node to avoid overlap
            totalCost = gPath[0][0] + nodeS[0]  # for debug
            sBest = (totalCost, retPath)
            for expS in exploredS:
                for expG in exploredG:
                    if expS[1][-1] == expG[1][-1]:
                        nexCost = expS[0] + expG[0]
                        if nexCost < sBest[0]:
                            retPath = expG[1]
                            retPath = retPath[::-1]  # reverse path to put goal at the end
                            retPath = expS[1][:-1] + retPath  # ignore last node in node to avoid overlap
                            totalCost = nexCost  # for debug
                            sBest = (totalCost, retPath)
                for rtG in routesG:
                    if expS[1][-1] == rtG[1][-1]:
                        nexCost = expS[0] + rtG[0]
                        if nexCost < sBest[0]:
                            retPath = rtG[1]
                            retPath = retPath[::-1]  # reverse path to put goal at the end
                            retPath = expS[1][:-1] + retPath  # ignore last node in node to avoid overlap
                            totalCost = nexCost  # for debug
                            sBest = (totalCost, retPath)
            return sBest[1]

        else:
            exploredS.append(nodeS)
            if nodeS[1][-1] == goal:
                return nodeS[1]
            front = graph[nodeS[1][-1]]
            front = sorted(front)
            for nd in front:
                if nd not in [n[-1] for _, n in exploredS]:
                    cost = nodeS[0] + graph.get_edge_weight(nodeS[1][-1], nd)
                    rqS.append((cost, nodeS[1] + [nd]))

        # implementation for Goal
        nodeG = rqG.pop()
        if exploredG and nodeG[1][-1] in [n[-1] for _, n in exploredG]:
            continue
        # check node and return path and cost of node
        # if exploredS:
        sPath = [n for n in exploredS if n[1][-1] == nodeG[1][-1]]
        # gPath = [n for n in exploredS if node[-1] in n[1]]
        if sPath:
            retPath = nodeG[1]
            retPath = retPath[::-1]  # reverse path to put goal at the end
            sp = sPath[0][1]
            retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
            totalCost = sPath[0][0] + nodeG[0]  # for debug
            gBest = (totalCost, retPath)
            for expG in exploredG:
                for expS in exploredS:
                    if expG[1][-1] == expS[1][-1]:
                        nexCost = expG[0] + expS[0]
                        if nexCost < gBest[0]:
                            retPath = expG[1]
                            retPath = retPath[::-1]  # reverse path to put goal at the end
                            sp = expS[1]
                            retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
                            totalCost = nexCost  # for debug
                            gBest = (totalCost, retPath)
                for rtS in routesS:
                    if expG[1][-1] == rtS[1][-1]:
                        nexCost = expG[0] + rtS[0]
                        if nexCost < gBest[0]:
                            retPath = expG[1]
                            retPath = retPath[::-1]  # reverse path to put goal at the end
                            sp = rtS[1]
                            retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
                            totalCost = nexCost  # for debug
                            gBest = (totalCost, retPath)
            return gBest[1]

        else:
            exploredG.append(nodeG)
            if nodeG[1][-1] == start:
                return nodeG[1][::-1]
            front = graph[nodeG[1][-1]]
            front = sorted(front)
            for nd in front:
                if nd not in [n[-1] for _, n in exploredG]:
                    cost = nodeG[0] + graph.get_edge_weight(nodeG[1][-1], nd)
                    rqG.append((cost, nodeG[1] + [nd]))

        # if sBest and gBest:
        #     if sBest <= gBest:
        #         return sBest[1]
        #     else:
        #         return gBest[1]
        # elif sBest:
        #     return sBest[1]
        # elif gBest:
        #     return gBest[1]



    # raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!

    if start == goal:
        return []
    # se up for Start
    rqS  = PriorityQueue()
    exploredS = []
    routesS = rqS.queue
    nodeS = (0, [start],0)
    rqS.append(nodeS)
    sBest = ()
    if start=='s' and goal=='d':
        drr = 5

    #set up for Goal
    rqG = PriorityQueue()
    exploredG = []
    routesG = rqG.queue
    nodeG = (0, [goal],0)
    rqG.append(nodeG)
    gBest = ()

    while routesS and routesG:
        # implementation for Start
        nodeS = rqS.pop()
        nxt = False
        for n in exploredS:
            if n[1][-1] == nodeS[1][-1]:
                nxt = True
        if not nxt:
            # continue
            # check node and return path and cost of node
            # if exploredG:
            gPath = [n for n in exploredG if n[1][-1] == nodeS[1][-1]]
            if gPath:
                retPath = gPath[0][1]
                retPath = retPath[::-1]  # reverse path to put goal at the end
                retPath = nodeS[1][:-1] + retPath  # ignore last node in node to avoid overlap
                totalCost = gPath[0][2] + nodeS[2]  # pure cost without heuristic
                tcost = gPath[0][0] + nodeS[0]
                hcost = heuristic(graph, start, goal)
                sBest = (totalCost, retPath)
                for expS in exploredS:
                    for expG in exploredG:
                        if expS[1][-1] == expG[1][-1]:
                            nexCost = expS[2] + expG[2]
                            if nexCost < sBest[0]:
                                retPath = expG[1]
                                retPath = retPath[::-1]  # reverse path to put goal at the end
                                retPath = expS[1][:-1] + retPath  # ignore last node in node to avoid overlap
                                totalCost = nexCost  # for debug
                                sBest = (totalCost, retPath)
                    for rtG in routesG:
                        if expS[1][-1] == rtG[1][-1]:
                            nexCost = expS[2] + rtG[2]
                            if nexCost < sBest[0]:
                                retPath = rtG[1]
                                retPath = retPath[::-1]  # reverse path to put goal at the end
                                retPath = expS[1][:-1] + retPath  # ignore last node in node to avoid overlap
                                totalCost = nexCost  # for debug
                                sBest = (totalCost, retPath)
                return sBest[1]


            else:
                exploredS.append(nodeS)
                if nodeS[1][-1] == goal:
                    return nodeS[1]
                front = graph[nodeS[1][-1]]
                front = sorted(front)
                for nd in front:
                    skip = False
                    for n in exploredS:
                        if n[1][-1] == nd:
                            skip = True
                    if skip:
                        continue
                    cost = nodeS[2] + graph.get_edge_weight(nodeS[1][-1], nd)
                    h = heuristic(graph, nd, goal)
                    rqS.append((cost + h, nodeS[1] + [nd], cost))

        # implementation for Goal
        nodeG = rqG.pop()
        nxt = False
        for n in exploredG:
            if n[1][-1] == nodeG[1][-1]:
                nxt = True
        if not nxt:
            # continue
            # check node and return path and cost of node
            # if exploredS:
            sPath = [n for n in exploredS if n[1][-1] == nodeG[1][-1]]
            # gPath = [n for n in exploredS if node[-1] in n[1]]
            if sPath:
                retPath = nodeG[1]
                retPath = retPath[::-1]  # reverse path to put goal at the end
                sp = sPath[0][1]
                retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
                totalCost = sPath[0][2] + nodeG[2]  # pure cost without heuristic
                tcost = sPath[0][0] + nodeG[0]
                hcost = heuristic(graph, start, goal)
                gBest = (totalCost, retPath)
                for expG in exploredG:
                    for expS in exploredS:
                        if expG[1][-1] == expS[1][-1]:
                            nexCost = expG[2] + expS[2]
                            if nexCost < gBest[0]:
                                retPath = expG[1]
                                retPath = retPath[::-1]  # reverse path to put goal at the end
                                sp = expS[1]
                                retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
                                totalCost = nexCost  # for debug
                                gBest = (totalCost, retPath)
                    for rtS in routesS:
                        if expG[1][-1] == rtS[1][-1]:
                            nexCost = expG[2] + rtS[2]
                            if nexCost < gBest[0]:
                                retPath = expG[1]
                                retPath = retPath[::-1]  # reverse path to put goal at the end
                                sp = rtS[1]
                                retPath = sp[:-1] + retPath  # ignore last node in node to avoid overlap
                                totalCost = nexCost  # for debug
                                gBest = (totalCost, retPath)
                return gBest[1]

            else:
                exploredG.append(nodeG)
                if nodeG[1][-1] == start:
                    return nodeG[1][::-1]
                front = graph[nodeG[1][-1]]
                front = sorted(front)
                for nd in front:
                    # if [n[1][-1] not in nd for n in exploredG]:
                    skip = False
                    for n in exploredG:
                        if n[1][-1] == nd:
                            skip = True
                    if skip:
                        continue
                    cost = nodeG[2] + graph.get_edge_weight(nodeG[1][-1], nd)
                    h = heuristic(graph, nd, goal)
                    rqG.append((cost + h, nodeG[1] + [nd], cost))

        # if sBest and gBest:
        #     if sBest <= gBest:
        #         return sBest[1]
        #     else:
        #         return gBest[1]
        # elif sBest:
        #     return sBest[1]
        # elif gBest:
        #     return gBest[1]

    # raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function

    # check if all values are equal:
    if all(n == goals[0] for n in goals):
        return []

    a = goals[0]
    b = goals[1]
    c = goals[2]

    # set up for A
    rqA = PriorityQueue()
    exploredA = []
    routesA = rqA.queue
    nodeA = (0, [a])
    rqA.append(nodeA)

    # set up for B
    rqB = PriorityQueue()
    exploredB = []
    routesB = rqB.queue
    nodeB = (0, [b])
    rqB.append(nodeB)

    # set up for C
    rqC = PriorityQueue()
    exploredC = []
    routesC = rqC.queue
    nodeC = (0, [c])
    rqC.append(nodeC)

    abBest = ()
    bcBest = ()
    acBest = ()
    lastFound = ()

    if a=="a" and b=="m" and c=="p":
        tutu = 5


    while routesA or routesB or routesC:
        # implementation for Start
        if routesA:
            nodeA = rqA.pop()
            if nodeA[1][-1] not in [n[-1] for _, n in exploredA]:
                # check B nodes and return path and cost of B node
                exploredA.append(nodeA)
                abPath = [n for n in exploredB if n[1][-1] == nodeA[1][-1]]
                if abPath:
                    abPath = abPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if abBest:
                        if nodeA[0] < abPath[0]: # compare cost of two paths to the node
                            exploredA.append(nodeA)
                            exploredB.remove(abPath)
                    else:
                        retPath = abPath[1]  # B-X
                        retPath = retPath[::-1]  # X-B
                        retPath = nodeA[1][:-1] + retPath  # A-(X-1)-X-B
                        totalCost = abPath[0] + nodeA[0]  # cost
                        abBest = (totalCost, retPath)
                        for expA in exploredA:
                            for expB in exploredB:
                                if expA[1][-1] == expB[1][-1]:
                                    nexCost = expA[0] + expB[0]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]       # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)
                            for rtB in routesB:
                                if expA[1][-1] == rtB[1][-1]:
                                    nexCost = expA[0] + rtB[0]
                                    if nexCost < abBest[0]:
                                        retPath = rtB[1]        # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)

                        if acBest or bcBest:
                            lastFound = abBest

                acPath = [n for n in exploredC if n[1][-1] == nodeA[1][-1]]
                if acPath:
                    acPath = acPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if acBest:
                        if nodeA[0] < acPath[0]:  # compare cost of two paths to the node
                            exploredA.append(nodeA)
                            exploredC.remove(acPath)
                    else:
                        retPath = acPath[1]              # C-X
                        retPath = retPath[::-1]             # X-C
                        retPath = nodeA[1][:-1] + retPath   # A-(X-1)-X-C
                        totalCost = acPath[0] + nodeA[0]
                        acBest = (totalCost, retPath)
                        for expA in exploredA:
                            for expC in exploredC:
                                if expA[1][-1] == expC[1][-1]:
                                    nexCost = expA[0] + expC[0]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]        # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  #
                                        acBest = (totalCost, retPath)
                            for rtC in routesC:
                                if expA[1][-1] == rtC[1][-1]:
                                    nexCost = expA[0] + rtC[0]
                                    if nexCost < acBest[0]:
                                        retPath = rtC[1]         # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)

                        if abBest or bcBest:
                            lastFound = acBest

                else:
                    # exploredA.append(nodeA)
                    front = graph[nodeA[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n in exploredA]:
                            cost = nodeA[0] + graph.get_edge_weight(nodeA[1][-1], nd)
                            rqA.append((cost, nodeA[1] + [nd]))

        # implementation for B
        if routesB:
            nodeB = rqB.pop()
            if nodeB[1][-1] not in [n[-1] for _, n in exploredB]:
                exploredB.append(nodeB)
                # check B nodes and return path and cost of B node
                abPath = [n for n in exploredA if n[1][-1] == nodeB[1][-1]]
                if abPath:
                    abPath = abPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if abBest:
                        if nodeB[0] < abPath[0]:  # compare cost of two paths to the node
                            exploredB.append(nodeB)
                            exploredA.remove(abPath)
                    else:
                        retPath = nodeB[1]  # B-X
                        retPath = retPath[::-1]  # X-B
                        retPath = abPath[1][:-1] + retPath  # A-(x-1)-X-B
                        totalCost = abPath[0] + nodeB[0]  # cost
                        abBest = (totalCost, retPath)
                        for expB in exploredB:
                            for expA in exploredA:
                                if expB[1][-1] == expA[1][-1]:
                                    nexCost = expB[0] + expA[0]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]                 # B-X
                                        retPath = retPath[::-1]           # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)
                            for rtA in routesA:
                                if expB[1][-1] == rtA[1][-1]:
                                    nexCost = expB[0] + rtA[0]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]                # B-X
                                        retPath = retPath[::-1]          # X-B
                                        retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)

                        if acBest or bcBest:
                            lastFound = abBest

                bcPath = [n for n in exploredC if n[1][-1] == nodeB[1][-1]]
                if bcPath:
                    bcPath = bcPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if bcBest:
                        if nodeB[0] < bcPath[0]:  # compare cost of two paths to the node
                            exploredB.append(nodeB)
                            exploredC.remove(bcPath)
                    else:
                        retPath = bcPath[1]               # C-X
                        retPath = retPath[::-1]              # X-C
                        retPath = nodeB[1][:-1] + retPath    # B-(X-1)-X-C
                        totalCost = bcPath[0] + nodeB[0]
                        bcBest = (totalCost, retPath)
                        for expB in exploredB:
                            for expC in exploredC:
                                if expB[1][-1] == expC[1][-1]:
                                    nexCost = expB[0] + expC[0]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]                 # C-X
                                        retPath = retPath[::-1]           # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost
                                        bcBest = (totalCost, retPath)
                            for rtC in routesC:
                                if expB[1][-1] == rtC[1][-1]:
                                    nexCost = expB[0] + rtC[0]
                                    if nexCost < bcBest[0]:
                                        retPath = rtC[1]                  # C-X
                                        retPath = retPath[::-1]           # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost
                                        bcBest = (totalCost, retPath)

                        if acBest or abBest:
                            lastFound = bcBest

                else:
                    # exploredB.append(nodeB)
                    front = graph[nodeB[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n in exploredB]:
                            cost = nodeB[0] + graph.get_edge_weight(nodeB[1][-1], nd)
                            rqB.append((cost, nodeB[1] + [nd]))

        # implementation for C
        if routesC:
            nodeC = rqC.pop()
            if nodeC[1][-1] not in [n[-1] for _, n in exploredC]:
                exploredC.append(nodeC)
                # check B nodes and return path and cost of B node
                acPath = [n for n in exploredA if n[1][-1] == nodeC[1][-1]]
                if acPath:
                    acPath = acPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if acBest:
                        if nodeC[0] < acPath[0]:  # compare cost of two paths to the node
                            exploredC.append(nodeC)
                            exploredA.remove(acPath)
                    else:
                        retPath = nodeC[1]                     # C-X
                        retPath = retPath[::-1]                # X-C
                        retPath = acPath[1][:-1] + retPath  # A-(x-1)-X-C
                        totalCost = acPath[0] + nodeC[0]
                        acBest = (totalCost, retPath)
                        for expC in exploredC:
                            for expA in exploredA:
                                if expC[1][-1] == expA[1][-1]:
                                    nexCost = expC[0] + expA[0]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]                 # C-X
                                        retPath = retPath[::-1]           # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)
                            for rtA in routesA:
                                if expC[1][-1] == rtA[1][-1]:
                                    nexCost = expC[0] + rtA[0]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]                # C-X
                                        retPath = retPath[::-1]          # X-C
                                        retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)

                        if abBest or bcBest:
                            lastFound = acBest

                bcPath = [n for n in exploredB if n[1][-1] == nodeC[1][-1]]
                if bcPath:
                    bcPath = bcPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if bcBest:
                        if nodeC[0] < bcPath[0]:  # compare cost of two paths to the node
                            exploredC.append(nodeC)
                            exploredB.remove(bcPath)
                    else:
                        retPath = nodeC[1]                     # C-X
                        retPath = retPath[::-1]                # X-C
                        retPath = bcPath[1][:-1] + retPath  # B-(x-1)-X-C
                        totalCost = bcPath[0] + nodeC[0]
                        bcBest = (totalCost, retPath)
                        for expC in exploredC:
                            for expB in exploredB:
                                if expC[1][-1] == expB[1][-1]:
                                    nexCost = expC[0] + expB[0]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]                 # C-X
                                        retPath = retPath[::-1]           # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        bcBest = (totalCost, retPath)
                            for rtB in routesB:
                                if expC[1][-1] == rtB[1][-1]:
                                    nexCost = expC[0] + rtB[0]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]                # C-X
                                        retPath = retPath[::-1]          # X-C
                                        retPath = rtB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        bcBest = (totalCost, retPath)

                        if abBest or acBest:
                            lastFound = bcBest
                else:
                    # exploredC.append(nodeC)
                    front = graph[nodeC[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n in exploredC]:
                            cost = nodeC[0] + graph.get_edge_weight(nodeC[1][-1], nd)
                            rqC.append((cost, nodeC[1] + [nd]))

        # when all three path found
        if abBest and acBest and bcBest:
            maxBest = max([abBest, acBest, bcBest], key = lambda x: x[0])
            if maxBest == abBest:
                abBest = ()
            elif maxBest == acBest:
                acBest = ()
            elif maxBest == bcBest:
                bcBest = ()



        # check if AB has a better path
        if acBest and bcBest:
            for expA in exploredA:
                for expB in exploredB:
                    if expA[1][-1] == expB[1][-1]:
                        nexCost = expA[0] + expB[0]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = expB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
                for rtB in routesB:
                    if expA[1][-1] == rtB[1][-1]:
                        nexCost = expA[0] + rtB[0]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = rtB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
            for expB in exploredB:
                for rtA in routesA:
                    if expB[1][-1] == rtA[1][-1]:
                        nexCost = expB[0] + rtA[0]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = expB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
            if abBest:
                if lastFound == acBest and abBest < acBest: # swap ac with ab and build (AB - BC)
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == bcBest and abBest < bcBest: # swap ac with ab and build (AB - AC)
                    nexCost = abBest[0] + acBest[0]
                    retPath = abBest[1]  # A-B
                    retPath = retPath[::-1]  # B-A
                    retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = acBest[0]
                acpath = acBest[1]
                path = [acpath[0]]
                for p in acpath:  # start from A and goes down to C
                    if p == acpath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expB in exploredB:
                        if expB[1][-1] == path[-1]:
                            if tcost > (cost + expB[0]):
                                tcost = cost + expB[0]
                                retPath = expB[1]  # B-X
                                retPath = retPath[::-1]  # X-B
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                abBest = (tcost, retPath)
                if abBest:
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            # (AC - BC)
            nexCost = acBest[0] + bcBest[0]  # A-C-B-C
            retPath = bcBest[1]  # B-C
            retPath = retPath[::-1]  # C-B
            retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
            totalCost = nexCost  # for debug
            retBest = (totalCost, retPath)
            return retBest[1]

        # check if BC has a better path
        if acBest and abBest:
            for expB in exploredB:
                for expC in exploredC:
                    if expB[1][-1] == expC[1][-1]:
                        nexCost = expB[0] + expC[0]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost
                            bcBest = (totalCost, retPath)
                for rtC in routesC:
                    if expB[1][-1] == rtC[1][-1]:
                        nexCost = expB[0] + rtC[0]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = rtC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost
                            bcBest = (totalCost, retPath)
            for expC in exploredC:
                for rtB in routesB:
                    if expC[1][-1] == rtB[1][-1]:
                        nexCost = expC[0] + rtB[0]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = rtB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            bcBest = (totalCost, retPath)
            if bcBest:
                if lastFound == acBest and bcBest < acBest: # swap ac with ab and build (AB - BC)
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == abBest and bcBest < abBest: # swap ac with ab and build (AC - BC)
                    nexCost = acBest[0] + bcBest[0]  # A-C-B-C
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost  # for debug
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = abBest[0]
                bapath = abBest[1][::-1]  # reverse path to B-A
                path = [bapath[0]]
                for p in bapath:  # start from B and goes down to A (to find the best BC connection)
                    if p == bapath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expC in exploredC:
                        if expC[1][-1] == path[-1]:
                            if tcost > (cost + expC[0]):
                                tcost = cost + expC[0]
                                retPath = expC[1]  # C-X
                                retPath = retPath[::-1]  # X-C
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                bcBest = (tcost, retPath)
                if bcBest:
                    nexCost = acBest[0] + bcBest[0]  # A-C-B-C
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost  # for debug
                    retBest = (totalCost, retPath)
                    return retBest[1]

            # (AC - AB)
            nexCost = abBest[0] + acBest[0]
            retPath = abBest[1]  # A-B
            retPath = retPath[::-1]  # B-A
            retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
            totalCost = nexCost  # for debug
            retBest = (totalCost, retPath)
            return retBest[1]

        # check if AC has a better path
        if abBest and bcBest:
            for expA in exploredA:
                for expC in exploredC:
                    if expA[1][-1] == expC[1][-1]:
                        nexCost = expA[0] + expC[0]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  #
                            acBest = (totalCost, retPath)
                for rtC in routesC:
                    if expA[1][-1] == rtC[1][-1]:
                        nexCost = expA[0] + rtC[0]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = rtC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            acBest = (totalCost, retPath)
            for expC in exploredC:
                for rtA in routesA:
                    if expC[1][-1] == rtA[1][-1]:
                        nexCost = expC[0] + rtA[0]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            acBest = (totalCost, retPath)

            if acBest:
                if lastFound == abBest and acBest < abBest:  # swap ac with ab and build (AC - BC)
                    nexCost = bcBest[0] + acBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == bcBest and acBest < bcBest:  # swap ac with ab and build (AB - AC)
                    nexCost = abBest[0] + acBest[0]
                    retPath = abBest[1]  # A-B
                    retPath = retPath[::-1]  # B-A
                    retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = abBest[0]
                abpath = abBest[1]
                path = [abpath[0]]
                for p in abpath:  # start from A and goes down to B
                    if p == abpath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expC in exploredC:
                        if expC[1][-1] == path[-1]:
                            if tcost > (cost + expC[0]):
                                tcost = cost + expC[0]
                                retPath = expC[1]  # C-X
                                retPath = retPath[::-1]  # X-C
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                acBest = (tcost, retPath)
                if acBest:
                    nexCost = bcBest[0] + acBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]

            # (AB - BC)
            nexCost = abBest[0] + bcBest[0]
            retPath = bcBest[1]  # B-C
            retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
            totalCost = nexCost
            retBest = (totalCost, retPath)
            return retBest[1]



    # raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function

    # check if all values are equal:
    if all(n == goals[0] for n in goals):
        return []

    a = goals[0]
    b = goals[1]
    c = goals[2]

    # set up for A
    rqA = PriorityQueue()
    exploredA = []
    routesA = rqA.queue
    nodeA = (0,[a],0)      # (heuristic+cost, path, cost)
    rqA.append(nodeA)

    # set up for B
    rqB = PriorityQueue()
    exploredB = []
    routesB = rqB.queue
    nodeB = (0,[b],0)
    rqB.append(nodeB)

    # set up for C
    rqC = PriorityQueue()
    exploredC = []
    routesC = rqC.queue
    nodeC = (0,[c],0)
    rqC.append(nodeC)

    abBest = ()
    bcBest = ()
    acBest = ()
    lastFound = ()

    if a == "a" and b == "z" and c == "i":
        tutu = 5
    hab = heuristic(graph, a, b)
    hac = heuristic(graph, a, c)
    hbc = heuristic(graph, b, c)

    while routesA or routesB or routesC:
        # implementation for Start
        if routesA:
            nodeA = rqA.pop()
            if nodeA[1][-1] not in [n[-1] for _, n, _ in exploredA]:
                # check B nodes and return path and cost of B node
                exploredA.append(nodeA)
                abPath = [n for n in exploredB if n[1][-1] == nodeA[1][-1]]
                if abPath:
                    abPath = abPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if abBest:
                        if nodeA[2] < abPath[2]:  # compare cost of two paths to the node
                            exploredA.append(nodeA)
                            exploredB.remove(abPath)
                    else:
                        retPath = abPath[1]  # B-X
                        retPath = retPath[::-1]  # X-B
                        retPath = nodeA[1][:-1] + retPath  # A-(X-1)-X-B
                        totalCost = abPath[2] + nodeA[2]  # cost pure
                        abBest = (totalCost, retPath)
                        for expA in exploredA:
                            for expB in exploredB:
                                if expA[1][-1] == expB[1][-1]:
                                    nexCost = expA[2] + expB[2]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]  # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)
                            for rtB in routesB:
                                if expA[1][-1] == rtB[1][-1]:
                                    nexCost = expA[2] + rtB[2]
                                    if nexCost < abBest[0]:
                                        retPath = rtB[1]  # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)

                        if acBest or bcBest:
                            lastFound = abBest

                acPath = [n for n in exploredC if n[1][-1] == nodeA[1][-1]]
                if acPath:
                    acPath = acPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if acBest:
                        if nodeA[2] < acPath[2]:  # compare cost of two paths to the node
                            exploredA.append(nodeA)
                            exploredC.remove(acPath)
                    else:
                        retPath = acPath[1]  # C-X
                        retPath = retPath[::-1]  # X-C
                        retPath = nodeA[1][:-1] + retPath  # A-(X-1)-X-C
                        totalCost = acPath[2] + nodeA[2]
                        acBest = (totalCost, retPath)
                        for expA in exploredA:
                            for expC in exploredC:
                                if expA[1][-1] == expC[1][-1]:
                                    nexCost = expA[2] + expC[2]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  #
                                        acBest = (totalCost, retPath)
                            for rtC in routesC:
                                if expA[1][-1] == rtC[1][-1]:
                                    nexCost = expA[2] + rtC[2]
                                    if nexCost < acBest[0]:
                                        retPath = rtC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)

                        if abBest or bcBest:
                            lastFound = acBest

                # else:
                if (not abPath) and (not acPath):
                    # exploredA.append(nodeA)
                    front = graph[nodeA[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n, _ in exploredA]:
                            cost = nodeA[2] + graph.get_edge_weight(nodeA[1][-1], nd)
                            # check the distance to the two goals and chose the closest one (add to queue)
                            hb = heuristic(graph, nd, b)
                            hc = heuristic(graph, nd, c)
                            if hb <= hc:
                                rqA.append((cost + hb, nodeA[1] + [nd], cost))
                            else:
                                rqA.append((cost + hc, nodeA[1] + [nd], cost))


        # implementation for B
        if routesB:
            nodeB = rqB.pop()
            if nodeB[1][-1] not in [n[-1] for _, n, _ in exploredB]:
                exploredB.append(nodeB)
                # check B nodes and return path and cost of B node
                abPath = [n for n in exploredA if n[1][-1] == nodeB[1][-1]]
                if abPath:
                    abPath = abPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if abBest:
                        if nodeB[2] < abPath[2]:  # compare cost of two paths to the node
                            exploredB.append(nodeB)
                            exploredA.remove(abPath)
                    else:
                        retPath = nodeB[1]  # B-X
                        retPath = retPath[::-1]  # X-B
                        retPath = abPath[1][:-1] + retPath  # A-(x-1)-X-B
                        totalCost = abPath[2] + nodeB[2]  # cost
                        abBest = (totalCost, retPath)
                        for expB in exploredB:
                            for expA in exploredA:
                                if expB[1][-1] == expA[1][-1]:
                                    nexCost = expB[2] + expA[2]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]  # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)
                            for rtA in routesA:
                                if expB[1][-1] == rtA[1][-1]:
                                    nexCost = expB[2] + rtA[2]
                                    if nexCost < abBest[0]:
                                        retPath = expB[1]  # B-X
                                        retPath = retPath[::-1]  # X-B
                                        retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-B
                                        totalCost = nexCost  # for debug
                                        abBest = (totalCost, retPath)

                        if acBest or bcBest:
                            lastFound = abBest

                bcPath = [n for n in exploredC if n[1][-1] == nodeB[1][-1]]
                if bcPath:
                    bcPath = bcPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if bcBest:
                        if nodeB[2] < bcPath[2]:  # compare cost of two paths to the node
                            exploredB.append(nodeB)
                            exploredC.remove(bcPath)
                    else:
                        retPath = bcPath[1]  # C-X
                        retPath = retPath[::-1]  # X-C
                        retPath = nodeB[1][:-1] + retPath  # B-(X-1)-X-C
                        totalCost = bcPath[2] + nodeB[2]
                        bcBest = (totalCost, retPath)
                        for expB in exploredB:
                            for expC in exploredC:
                                if expB[1][-1] == expC[1][-1]:
                                    nexCost = expB[2] + expC[2]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost
                                        bcBest = (totalCost, retPath)
                            for rtC in routesC:
                                if expB[1][-1] == rtC[1][-1]:
                                    nexCost = expB[2] + rtC[2]
                                    if nexCost < bcBest[0]:
                                        retPath = rtC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost
                                        bcBest = (totalCost, retPath)

                        if acBest or abBest:
                            lastFound = bcBest

                # else:
                if (not abPath) and (not bcPath):
                    # exploredB.append(nodeB)
                    front = graph[nodeB[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n, _ in exploredB]:
                            cost = nodeB[2] + graph.get_edge_weight(nodeB[1][-1], nd)
                            # check the distance to the two goals and chose the closest one (add to queue)
                            ha = heuristic(graph, nd, a)
                            hc = heuristic(graph, nd, c)
                            if ha <= hc:
                                rqB.append((cost + ha, nodeB[1] + [nd], cost))
                            else:
                                rqB.append((cost + hc, nodeB[1] + [nd], cost))

        # implementation for C
        if routesC:
            nodeC = rqC.pop()
            if nodeC[1][-1] not in [n[-1] for _, n, _ in exploredC]:
                exploredC.append(nodeC)
                # check B nodes and return path and cost of B node
                acPath = [n for n in exploredA if n[1][-1] == nodeC[1][-1]]
                if acPath:
                    acPath = acPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if acBest:
                        if nodeC[2] < acPath[2]:  # compare cost of two paths to the node
                            exploredC.append(nodeC)
                            exploredA.remove(acPath)
                    else:
                        retPath = nodeC[1]  # C-X
                        retPath = retPath[::-1]  # X-C
                        retPath = acPath[1][:-1] + retPath  # A-(x-1)-X-C
                        totalCost = acPath[2] + nodeC[2]
                        acBest = (totalCost, retPath)
                        for expC in exploredC:
                            for expA in exploredA:
                                if expC[1][-1] == expA[1][-1]:
                                    nexCost = expC[2] + expA[2]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)
                            for rtA in routesA:
                                if expC[1][-1] == rtA[1][-1]:
                                    nexCost = expC[2] + rtA[2]
                                    if nexCost < acBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        acBest = (totalCost, retPath)

                        if abBest or bcBest:
                            lastFound = acBest

                bcPath = [n for n in exploredB if n[1][-1] == nodeC[1][-1]]
                if bcPath:
                    bcPath = bcPath[0]
                    # if A-B the shortest path already found, check which path shorter to the node
                    # and add/keep path in to explored list of the goals with the shortest path to the node
                    if bcBest:
                        if nodeC[2] < bcPath[2]:  # compare cost of two paths to the node
                            exploredC.append(nodeC)
                            exploredB.remove(bcPath)
                    else:
                        retPath = nodeC[1]  # C-X
                        retPath = retPath[::-1]  # X-C
                        retPath = bcPath[1][:-1] + retPath  # B-(x-1)-X-C
                        totalCost = bcPath[2] + nodeC[2]
                        bcBest = (totalCost, retPath)
                        for expC in exploredC:
                            for expB in exploredB:
                                if expC[1][-1] == expB[1][-1]:
                                    nexCost = expC[2] + expB[2]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        bcBest = (totalCost, retPath)
                            for rtB in routesB:
                                if expC[1][-1] == rtB[1][-1]:
                                    nexCost = expC[2] + rtB[2]
                                    if nexCost < bcBest[0]:
                                        retPath = expC[1]  # C-X
                                        retPath = retPath[::-1]  # X-C
                                        retPath = rtB[1][:-1] + retPath  # B-(X-1)-X-C
                                        totalCost = nexCost  # for debug
                                        bcBest = (totalCost, retPath)

                        if abBest or acBest:
                            lastFound = bcBest

                # else:
                if (not acPath) and (not bcPath):
                    # exploredC.append(nodeC)
                    front = graph[nodeC[1][-1]]
                    front = sorted(front)
                    for nd in front:
                        if nd not in [n[-1] for _, n, _ in exploredC]:
                            cost = nodeC[2] + graph.get_edge_weight(nodeC[1][-1], nd)
                            # check the distance to the two goals and chose the closest one (add to queue)
                            ha = heuristic(graph, nd, a)
                            hb = heuristic(graph, nd, b)
                            if ha <= hb:
                                rqC.append((cost + ha, nodeC[1] + [nd], cost))
                            else:
                                rqC.append((cost + hb, nodeC[1] + [nd], cost))


        # when all three path found
        if abBest and acBest and bcBest:
            maxBest = max([abBest, acBest, bcBest], key=lambda x: x[0])
            if maxBest == abBest:
                abBest = ()
            elif maxBest == acBest:
                acBest = ()
            elif maxBest == bcBest:
                bcBest = ()

        # check if AB has a better path
        if acBest and bcBest:
            for expA in exploredA:
                for expB in exploredB:
                    if expA[1][-1] == expB[1][-1]:
                        nexCost = expA[2] + expB[2]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = expB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
                for rtB in routesB:
                    if expA[1][-1] == rtB[1][-1]:
                        nexCost = expA[2] + rtB[2]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = rtB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
            for expB in exploredB:
                for rtA in routesA:
                    if expB[1][-1] == rtA[1][-1]:
                        nexCost = expB[2] + rtA[2]
                        if (not abBest) or (nexCost < abBest[0]):
                            retPath = expB[1]  # B-X
                            retPath = retPath[::-1]  # X-B
                            retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-B
                            totalCost = nexCost  # for debug
                            abBest = (totalCost, retPath)
            if abBest:
                if lastFound == acBest and abBest < acBest:  # swap ac with ab and build (AB - BC)
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == bcBest and abBest < bcBest:  # swap ac with ab and build (AB - AC)
                    nexCost = abBest[0] + acBest[0]
                    retPath = abBest[1]  # A-B
                    retPath = retPath[::-1]  # B-A
                    retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = acBest[0]
                acpath = acBest[1]
                path = [acpath[0]]
                for p in acpath:  # start from A and goes down to C
                    if p == acpath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expB in exploredB:
                        if expB[1][-1] == path[-1]:
                            if tcost > (cost + expB[2]):
                                tcost = cost + expB[2]
                                retPath = expB[1]  # B-X
                                retPath = retPath[::-1]  # X-B
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                abBest = (tcost, retPath)
                if abBest:
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            # (AC - BC)
            nexCost = acBest[0] + bcBest[0]  # A-C-B-C
            retPath = bcBest[1]  # B-C
            retPath = retPath[::-1]  # C-B
            retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
            totalCost = nexCost  # for debug
            retBest = (totalCost, retPath)
            return retBest[1]

        # check if BC has a better path
        if acBest and abBest:
            for expB in exploredB:
                for expC in exploredC:
                    if expB[1][-1] == expC[1][-1]:
                        nexCost = expB[2] + expC[2]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost
                            bcBest = (totalCost, retPath)
                for rtC in routesC:
                    if expB[1][-1] == rtC[1][-1]:
                        nexCost = expB[2] + rtC[2]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = rtC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost
                            bcBest = (totalCost, retPath)
            for expC in exploredC:
                for rtB in routesB:
                    if expC[1][-1] == rtB[1][-1]:
                        nexCost = expC[2] + rtB[2]
                        if (not bcBest) or (nexCost < bcBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = rtB[1][:-1] + retPath  # B-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            bcBest = (totalCost, retPath)
            if bcBest:
                if lastFound == acBest and bcBest < acBest:  # swap ac with ab and build (AB - BC)
                    nexCost = abBest[0] + bcBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == abBest and bcBest < abBest:  # swap ac with ab and build (AC - BC)
                    nexCost = acBest[0] + bcBest[0]  # A-C-B-C
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost  # for debug
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = abBest[0]
                bapath = abBest[1][::-1]  # reverse path to B-A
                path = [bapath[0]]
                for p in bapath:  # start from B and goes down to A (to find the best BC connection)
                    if p == bapath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expC in exploredC:
                        if expC[1][-1] == path[-1]:
                            if tcost > (cost + expC[2]):
                                tcost = cost + expC[2]
                                retPath = expC[1]  # C-X
                                retPath = retPath[::-1]  # X-C
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                bcBest = (tcost, retPath)
                if bcBest:
                    nexCost = acBest[0] + bcBest[0]  # A-C-B-C
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost  # for debug
                    retBest = (totalCost, retPath)
                    return retBest[1]

            # (AC - AB)
            nexCost = abBest[0] + acBest[0]
            retPath = abBest[1]  # A-B
            retPath = retPath[::-1]  # B-A
            retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
            totalCost = nexCost  # for debug
            retBest = (totalCost, retPath)
            return retBest[1]

        # check if AC has a better path
        if abBest and bcBest:
            for expA in exploredA:
                for expC in exploredC:
                    if expA[1][-1] == expC[1][-1]:
                        nexCost = expA[2] + expC[2]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  #
                            acBest = (totalCost, retPath)
                for rtC in routesC:
                    if expA[1][-1] == rtC[1][-1]:
                        nexCost = expA[2] + rtC[2]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = rtC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = expA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            acBest = (totalCost, retPath)
            for expC in exploredC:
                for rtA in routesA:
                    if expC[1][-1] == rtA[1][-1]:
                        nexCost = expC[2] + rtA[2]
                        if (not acBest) or (nexCost < acBest[0]):
                            retPath = expC[1]  # C-X
                            retPath = retPath[::-1]  # X-C
                            retPath = rtA[1][:-1] + retPath  # A-(X-1)-X-C
                            totalCost = nexCost  # for debug
                            acBest = (totalCost, retPath)

            if acBest:
                if lastFound == abBest and acBest < abBest:  # swap ac with ab and build (AC - BC)
                    nexCost = bcBest[0] + acBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
                elif lastFound == bcBest and acBest < bcBest:  # swap ac with ab and build (AB - AC)
                    nexCost = abBest[0] + acBest[0]
                    retPath = abBest[1]  # A-B
                    retPath = retPath[::-1]  # B-A
                    retPath = retPath[:-1] + acBest[1]  # B-(A-1)-A-C
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]
            else:
                cost = 0
                tcost = abBest[0]
                abpath = abBest[1]
                path = [abpath[0]]
                for p in abpath:  # start from A and goes down to B
                    if p == abpath[0]:
                        continue
                    else:
                        cost = cost + graph.get_edge_weight(path[-1], p)
                        path = path + [p]
                    for expC in exploredC:
                        if expC[1][-1] == path[-1]:
                            if tcost > (cost + expC[2]):
                                tcost = cost + expC[2]
                                retPath = expC[1]  # C-X
                                retPath = retPath[::-1]  # X-C
                                retPath = path[:-1] + retPath  # A-(X-1)-X-C
                                acBest = (tcost, retPath)
                if acBest:
                    nexCost = bcBest[0] + acBest[0]
                    retPath = bcBest[1]  # B-C
                    retPath = retPath[::-1]  # C-B
                    retPath = acBest[1][:-1] + retPath  # A-(C-1)-C-B
                    totalCost = nexCost
                    retBest = (totalCost, retPath)
                    return retBest[1]

            # (AB - BC)
            nexCost = abBest[0] + bcBest[0]
            retPath = bcBest[1]  # B-C
            retPath = abBest[1][:-1] + retPath  # A-(B-1)-B-C
            totalCost = nexCost
            retBest = (totalCost, retPath)
            return retBest[1]

    # raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Eduard Shuvaev"
    # raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
