# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function  # force to use print as function print(args)
from __future__ import unicode_literals

from math import ceil, floor
from collections import deque
import numpy as np
import os
import random
import math
import cv2
import copy

def get_aog(cfg):
    aog_param = Param(grid_ht=cfg.RGM.GRID_HEIGHT, grid_wd=cfg.RGM.GRID_WIDTH,
                      min_size=cfg.RGM.MIN_SIZE, control_side_length=cfg.RGM.CONTROL_SIDE_LENGTH,
                      overlap_ratio=cfg.RGM.OVERLAP_RATIO, use_root_TerminalNode=cfg.RGM.USE_ROOT_TERMINAL_NODE,
                      not_use_large_TerminalNode=cfg.RGM.NOT_USE_LARGE_TERMINAL_NODE,
                      turn_off_size_ratio_TerminalNode=cfg.RGM.TURN_OFF_SIZE_RATIO_TERMINAL_NODE,
                      use_tnode_as_alpha_channel=cfg.RGM.USE_TNODE_AS_ALPHA_CHANNEL,
                      use_super_OrNode=cfg.RGM.USE_SUPER_OR_NODE)
    aog = AOGrid(aog_param)
    aog.Create()
    return aog

class NodeType(object):
    OrNode = "OrNode"
    AndNode = "AndNode"
    TerminalNode = "TerminalNode"
    Unknow = "Unknown"


class SplitType(object):
    HorSplit = "Hor"
    VerSplit = "Ver"
    Unknown = "Unknown"


class Param(object):
    """Input parameters for creating an AOGrid
       TODO: put TNodes as alpha-channel nodes of AndNodes
    """

    def __init__(self, grid_ht=3, grid_wd=3, min_size=1, control_side_length=True,
                 overlap_ratio=0, use_root_TerminalNode=False,
                 not_use_large_TerminalNode=False, turn_off_size_ratio_TerminalNode=0.5,
                 use_tnode_as_alpha_channel=False, use_super_OrNode=False,
                 remove_single_child_or_node=False):
        self.grid_ht = grid_ht
        self.grid_wd = grid_wd
        self.min_size = min_size
        self.control_side_length = control_side_length
        self.overlap_ratio = overlap_ratio
        self.use_root_terminal_node = use_root_TerminalNode
        self.not_use_large_terminal_node = not_use_large_TerminalNode
        self.turn_off_size_ratio_terminal_node = turn_off_size_ratio_TerminalNode
        self.use_tnode_as_alpha_channel = use_tnode_as_alpha_channel
        self.use_super_OrNode = use_super_OrNode
        self.remove_single_child_or_node = remove_single_child_or_node
        self.tag = '{}{}{}{}{}{}{}{}'.format(self.min_size, self.control_side_length, self.overlap_ratio,
                                               self.use_root_terminal_node, self.not_use_large_terminal_node,
                                               self.turn_off_size_ratio_terminal_node,
                                             self.use_tnode_as_alpha_channel, self.use_super_OrNode)


class Rect(object):
    """A simple rectangle
    """

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def Width(self):
        return self.x2 - self.x1 + 1

    def Height(self):
        return self.y2 - self.y1 + 1

    def Area(self):
        return self.Width() * self.Height()

    def MinLength(self):
        return min(self.Width(), self.Height())

    def IsOverlap(self, other):
        assert isinstance(other, self.__class__)

        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        if x1 > x2:
            return False

        y1 = max(self.y1, other.y1)
        y2 = min(self.y2, other.y2)
        if y1 > y2:
            return False

        return True

    def IsSame(self, other):
        assert isinstance(other, self.__class__)

        return self.Width() == other.Width() and self.Height() == other.Height()


class Node(object):
    """Types of nodes in an AOGrid
    AND-node (structural decomposition),
    OR-node (alternative decompositions),
    TERMINAL-node (link to data appearance).
    """

    def __init__(self, id=-1, node_type=NodeType.Unknow, rect_idx=-1, child_ids=[], parent_ids=[],
                 split_type=SplitType.Unknown, split_step1=0, split_step2=0):
        self.id = id
        self.node_type = node_type
        self.rect_idx = rect_idx
        self.child_ids = child_ids
        self.parent_ids = parent_ids
        self.split_type = split_type
        self.split_step1 = split_step1
        self.split_step2 = split_step2
        self.on_off = True
        self.out_edge_visited_count = []
        self.which_classes_visited = {}  # key=class name, val=freq_intra_class

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.node_type == other.node_type) and (self.rect_idx == other.rect_idx) and \
                   (self.split_type == other.split_type) and (self.split_step1 == other.split_step1) and \
                   (self.split_step2 == other.split_step2)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))


class AOGrid(object):
    """The AOGrid defines a Directed Acyclic And-Or Grap
    which is used to explore/unfold the space of latent structures
    of a grid (e.g., a 7 * 7 grid for a 100 * 200 lattice)
    """

    def __init__(self, param):
        assert isinstance(param, Param)
        self.param = param
        self.primitive_set = []
        self.node_set = []
        self.num_TNodes = 0
        self.num_AndNodes = 0
        self.num_OrNodes = 0
        self.DFS = []
        self.BFS = []
        self.OrNodeIdxInBFS = {}
        self.TNodeIdxInBFS = {}
        self.TNodeColors = {}
        self._part_instance = None  # a part instance of the type (ht, wd) is defined by (x, y, ht, wd)
        self._part_type = None  # a part type is defined by (ht, wd)
        self._part_instance_with_type = None
        self._matrix_form = None # compact matrix form representing the AOG

    @property
    def part_instance(self):
        return self._part_instance

    @property
    def part_type(self):
        return self._part_type

    @property
    def part_instance_with_type(self):
        return  self._part_instance_with_type

    @property
    def matrix_form(self):
        return self._matrix_form

    def _GetMatrixForm(self):
        max_num_children = 0
        for node in self.node_set:
            max_num_children = max(max_num_children, len(node.child_ids))

        # matrix form: each row (node.id, node.node_type, ornodeIdxInBFS, nb_children, child_ids)
        self._matrix_form = np.zeros((len(self.node_set), 3 + max_num_children + 1), dtype=np.float32)

        for i, node in enumerate(self.node_set):
            self._matrix_form[i, 0] = node.id
            if node.node_type == NodeType.OrNode:
                self._matrix_form[i, 1] = 0
                self._matrix_form[i, 2] = self.OrNodeIdxInBFS[node.id]
            elif node.node_type == NodeType.AndNode:
                self._matrix_form[i, 1] = 1
            else:
                self._matrix_form[i, 1] = 2

            self._matrix_form[i, 3] = len(node.child_ids)
            self._matrix_form[i, 4:4+len(node.child_ids)] = node.child_ids

    def _GetParts(self):
        assert len(self.node_set) > 0
        self._part_instance = np.empty((0, 4), dtype=np.float32)
        self._part_instance_with_type = np.empty((0, 5), dtype=np.float32)
        self._part_type = []
        # TODO: change to use BFS order
        for node in self.node_set:
            if node.node_type == NodeType.TerminalNode:
                rect = self.primitive_set[node.rect_idx]
                self._part_instance = np.vstack((self._part_instance, np.array([rect.x1, rect.y1, rect.x2, rect.y2])))
                p = [rect.Height(), rect.Width()]
                if p not in self._part_type:
                    self._part_type.append(p)
                idx = self._part_type.index(p)
                self._part_instance_with_type = np.vstack((self._part_instance_with_type,
                                                          np.array([rect.x1, rect.y1, rect.x2, rect.y2, idx])))

    def _AddPrimitve(self, rect):
        assert isinstance(rect, Rect)

        if rect in self.primitive_set:
            return self.primitive_set.index(rect)

        self.primitive_set.append(rect)

        return len(self.primitive_set) - 1

    def _AddNode(self, node):
        assert isinstance(node, Node)

        if node in self.node_set:
            node = self.node_set[self.node_set.index(node)]
            return False, node

        node.id = len(self.node_set)
        if node.node_type == NodeType.AndNode:
            self.num_AndNodes += 1
        elif node.node_type == NodeType.OrNode:
            self.num_OrNodes += 1
        elif node.node_type == NodeType.TerminalNode:
            self.num_TNodes += 1
        else:
            raise NotImplementedError

        self.node_set.append(node)

        return True, node

    def _DoSplit(self, rect):
        assert isinstance(rect, Rect)

        if self.param.control_side_length:
            return rect.Width() >= self.param.min_size and rect.Height() >= self.param.min_size

        return rect.Area() > self.param.min_size

    def _SplitStep(self, sz):
        if self.param.control_side_length:
            return self.param.min_size

        if sz >= self.param.min_size:
            return 1
        else:
            return ceil(self.param.min_size / sz)

    def _DFS(self, id, visited):
        if visited[id] == 1:
            raise RuntimeError

        visited[id] = 1
        for i in self.node_set[id].child_ids:
            if visited[i] < 2:
                visited = self._DFS(i, visited)

        self.DFS.append(id)
        visited[id] = 2

        return visited

    def _BFS(self, id, visited):
        if visited[id] == 1:
            raise RuntimeError

        self.BFS.append(id)
        visited[id] = 1

        for i in self.node_set[id].child_ids:
            if visited[i] < 2:
                visited = self._BFS(i, visited)

        visited[id] = 2

        return visited

    def _AssignParentIds(self):
        for i in range(len(self.node_set)):
            self.node_set[i].parent_ids = []

        for node in self.node_set:
            for i in node.child_ids:
                self.node_set[i].parent_ids.append(node.id)

    def TurnOnOffNodes(self, on_off):
        for i in range(len(self.node_set)):
            self.node_set[i].on_off = on_off

    def UpdateOnOffNodes(self, pg, offset_using_part_type, class_name=''):
        BFS = [self.BFS[0]]
        pg_used = np.ones((1, len(pg)), dtype=np.int) * -1
        configuration = []
        tnode_offset_indx = []
        while len(BFS):
            id = BFS.pop()
            node = self.node_set[id]
            self.node_set[id].on_off = True
            if len(class_name):
                if class_name in node.which_classes_visited.keys():
                    self.node_set[id].which_classes_visited[class_name] += 1.0
                else:
                    self.node_set[id].which_classes_visited[class_name] = 0

            if node.node_type == NodeType.OrNode:
                idx = self.OrNodeIdxInBFS[node.id]
                BFS.append(node.child_ids[int(pg[idx])])
                pg_used[0, idx] = int(pg[idx])
                if len(self.node_set[id].out_edge_visited_count):
                    self.node_set[id].out_edge_visited_count[int(pg[idx])] += 1.0
                else:
                    self.node_set[id].out_edge_visited_count = np.zeros((len(node.child_ids),), dtype=np.float32)
            elif node.node_type == NodeType.AndNode:
                BFS += node.child_ids

            else:
                configuration.append(node.id)

                offset_ind = 0
                if not offset_using_part_type:
                    for node1 in self.node_set:
                        if node1.node_type == NodeType.TerminalNode:  # change to BFS after _part_instance is changed to BFS
                            if node1.id == node.id:
                                break
                            offset_ind += 1
                else:
                    rect = self.primitive_set[node.rect_idx]
                    offset_ind = self.part_type.index([rect.Height(), rect.Width()])

                tnode_offset_indx.append(offset_ind)

        configuration.sort()
        cfg = np.ones((1, self.num_TNodes), dtype=np.int) * -1
        cfg[0, :len(configuration)] = configuration
        return pg_used, cfg, tnode_offset_indx

    def ResetOutEdgeVisitedCountNodes(self):
        for i in range(len(self.node_set)):
            self.node_set[i].out_edge_visited_count = []

    def NormalizeOutEdgeVisitedCountNodes(self, count=0):
        if count == 0:
            for i in range(len(self.node_set)):
                if len(self.node_set[i].out_edge_visited_count):
                    count = max(count, max(self.node_set[i].out_edge_visited_count))

        if count == 0:
            return

        for i in range(len(self.node_set)):
            if len(self.node_set[i].out_edge_visited_count):
                self.node_set[i].out_edge_visited_count /= count

    def ResetWhichClassesVisitedNodes(self):
        for i in range(len(self.node_set)):
            self.node_set[i].which_classes_visited = {}

    def NormalizeWhichClassesVisitedNodes(self, class_name, count):
        assert count > 0
        for i in range(len(self.node_set)):
            if class_name in self.node_set[i].which_classes_visited.keys():
                self.node_set[i].which_classes_visited[class_name] /= count

    def PictureWhichClassesVisitedNodes(self, save_dir, color_map):
        import matplotlib.pyplot as plt

        save_dir = os.path.join(save_dir, 'pictureWhichClassesVisitedNodes')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # # pie slice
        # N = len(color_map)
        # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        # width = np.pi * 1.8 / N
        #
        # for i in range(len(self.node_set)):
        #     if len(self.node_set[i].which_classes_visited):
        #         # Compute pie slices
        #         radii = []
        #         colors = []
        #         labels = []
        #         for k in self.node_set[i].which_classes_visited.keys():
        #             radii.append(self.node_set[i].which_classes_visited[k])
        #             colors.append(color_map[k])
        #             labels.append(k)
        #
        #         for j in range(len(radii), N):
        #             radii.append(0)
        #             colors.append((1.0, 1.0, 1.0))
        #             labels.append('')
        #
        #         ax = plt.subplot(111, projection='polar')
        #         bars = ax.bar(theta, radii, width=width, bottom=0.0, tick_label=labels)
        #
        #         # Use custom colors and opacity
        #         j = 0
        #         for r, bar in zip(radii, bars):
        #             bar.set_facecolor(colors[j])
        #             bar.set_alpha(0.5)
        #             j += 1
        #
        #         # plt.axis('equal')
        #         plt.tight_layout()
        #         #plt.show()
        #         plt.savefig(os.path.join(save_dir, '{:4d}.png'.format(self.node_set[i].id)))
        #         plt.clf()

        width = 0.9
        ind = range(len(color_map))
        name2id = {}

        for i, k in enumerate(color_map.keys()):
            name2id[k] = int(i)

        for i in range(len(self.node_set)):
            if len(self.node_set[i].which_classes_visited):
                # Compute pie slices
                val = np.zeros((len(ind),), dtype=np.float)
                colors = [np.ones((3,), dtype=np.float) for _ in range(len(ind))]
                labels = ['' for _ in range(len(ind))]
                for j, k in enumerate(self.node_set[i].which_classes_visited.keys()):
                    cur_ind = name2id[k]
                    val[cur_ind] = self.node_set[i].which_classes_visited[k]
                    colors[cur_ind] = color_map[k]
                    labels[cur_ind] = k

                ax = plt.subplot(111)
                bars = ax.bar(ind, val, width=width, bottom=0.0)

                ax.set_xticks(ind)
                ax.set_xticklabels(labels, rotation=45)

                # Use custom colors and opacity
                for j, bar in enumerate(bars):
                    bar.set_facecolor(colors[j])
                    # bar.set_alpha(0.5)

                # plt.axis('equal')
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(save_dir, '{:04d}.png'.format(self.node_set[i].id)))
                plt.clf()

    def Create(self):
        print("======= creating AOGrid, could take a while")

        # the root OrNode
        rect = Rect(0, 0, self.param.grid_wd - 1, self.param.grid_ht - 1)
        self.primitive_set.append(rect)
        node = Node(node_type=NodeType.OrNode, rect_idx=0)
        self._AddNode(node)

        BFS = deque()
        BFS.append(0)
        while len(BFS) > 0:
            curId = BFS.popleft()
            curNode = self.node_set[curId]
            curRect = self.primitive_set[curNode.rect_idx]
            curWd = curRect.Width()
            curHt = curRect.Height()

            childIds = []

            if curNode.node_type == NodeType.OrNode:
                # add a terminal node for a non-root OrNode
                allowTerminate = not (self.param.not_use_large_terminal_node and
                                      (curWd * curHt) / (self.param.grid_ht * self.param.grid_wd) >
                                      self.param.turn_off_size_ratio_terminal_node)

                if curId > 0 and allowTerminate:
                    node = Node(node_type=NodeType.TerminalNode, rect_idx=curNode.rect_idx)
                    suc, node = self._AddNode(node)
                    childIds.append(node.id)

                # add all AndNodes for horizontal and vertical binary splits
                if not self._DoSplit(curRect):
                    continue

                # horizontal splits
                step = self._SplitStep(curWd)
                for topHt in range(step, curHt - step + 1):
                    bottomHt = curHt - topHt
                    if self.param.overlap_ratio > 0:
                        numSplit = int(1 + floor(topHt * self.param.overlap_ratio))
                    else:
                        numSplit = 1
                    for b in range(0, numSplit):
                        node = Node(node_type=NodeType.AndNode, rect_idx=curNode.rect_idx,
                                    split_type=SplitType.HorSplit,
                                    split_step1=topHt, split_step2=curHt - bottomHt)
                        suc, node = self._AddNode(node)
                        if suc:
                            BFS.append(node.id)

                        childIds.append(node.id)
                        bottomHt += 1

                # vertical splits
                step = self._SplitStep(curHt)
                for leftWd in range(step, curWd - step + 1):
                    rightWd = curWd - leftWd
                    if self.param.overlap_ratio > 0:
                        numSplit = int(1 + floor(leftWd * self.param.overlap_ratio))
                    else:
                        numSplit = 1
                    for r in range(0, numSplit):
                        node = Node(node_type=NodeType.AndNode, rect_idx=curNode.rect_idx,
                                    split_type=SplitType.VerSplit,
                                    split_step1=leftWd, split_step2=curWd - rightWd)
                        suc, node = self._AddNode(node)
                        if suc:
                            BFS.append(node.id)

                        childIds.append(node.id)
                        rightWd += 1
            elif curNode.node_type == NodeType.AndNode:
                # add two child OrNodes
                if curNode.split_type == SplitType.HorSplit:
                    top = Rect(x1=curRect.x1, y1=curRect.y1,
                               x2=curRect.x2, y2=curRect.y1 + curNode.split_step1 - 1)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(top))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                    bottom = Rect(x1=curRect.x1, y1=curRect.y1 + curNode.split_step2,
                                  x2=curRect.x2, y2=curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(bottom))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)
                elif curNode.split_type == SplitType.VerSplit:
                    left = Rect(curRect.x1, curRect.y1,
                                curRect.x1 + curNode.split_step1 - 1, curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(left))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                    right = Rect(curRect.x1 + curNode.split_step2, curRect.y1,
                                 curRect.x2, curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(right))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

            childIds = list(set(childIds))
            self.node_set[curId].child_ids = childIds

        root_id = 0
        if self.param.use_root_terminal_node:
            tnode = Node(node_type=NodeType.TerminalNode, rect_idx=0)
            _, tnode = self._AddNode(tnode)
            root = Node(node_type=NodeType.AndNode, rect_idx=0)
            root.child_ids = [tnode.id, 0]
            _, rnode = self._AddNode(root)
            root_id = rnode.id

        self._AssignParentIds()

        visited = np.zeros(len(self.node_set))
        self._DFS(root_id, visited)
        visited = np.zeros(len(self.node_set))
        self._BFS(root_id, visited)

        if self.param.use_tnode_as_alpha_channel:
            for id in self.BFS:
                node = self.node_set[id]
                if node.node_type == NodeType.OrNode and len(node.child_ids) > 1:
                    for ch in node.child_ids:
                        ch_node = self.node_set[ch]
                        if ch_node.node_type == NodeType.TerminalNode:
                            and_node = Node(node_type=NodeType.AndNode, rect_idx=ch_node.rect_idx)
                            _, and_node = self._AddNode(and_node)
                            and_node.child_ids = [ch_node.id, node.id]

                            for pr in node.parent_ids:
                                pr_node = self.node_set[pr]
                                for i, pr_ch in enumerate(pr_node.child_ids):
                                    if pr_ch == node.id:
                                        pr_node.child_ids[i] = and_node.id
                                        break

                            node.child_ids.remove(ch)

                            break

            self._AssignParentIds()

            self.DFS = []
            self.BFS = []
            visited = np.zeros(len(self.node_set))
            self._DFS(root_id, visited)
            visited = np.zeros(len(self.node_set))
            self._BFS(root_id, visited)

        if self.param.use_super_OrNode:
            super_or_node = Node(node_type=NodeType.OrNode, rect_idx=-1)
            _, super_or_node = self._AddNode(super_or_node)
            super_or_node.child_ids = []
            for node in self.node_set:
                if node.node_type == NodeType.OrNode and node.rect_idx != -1:
                    rect = self.primitive_set[node.rect_idx]
                    r = float(rect.Area()) / float(self.param.grid_ht * self.param.grid_wd)
                    if r > 0.5:
                        super_or_node.child_ids.append(node.id)


            root_id = super_or_node.id

            self._AssignParentIds()

            self.DFS = []
            self.BFS = []
            visited = np.zeros(len(self.node_set))
            self._DFS(root_id, visited)
            visited[:] = 0
            self._BFS(root_id, visited)
            
        #remove or-nodes with single child node
        if self.param.remove_single_child_or_node:
            remove_ids = []
            for node in self.node_set:
                if node.node_type == NodeType.OrNode and len(node.child_ids) == 1:
                    for pr in node.parent_ids:
                        pr_node = self.node_set[pr]
                        for i, pr_ch in enumerate(pr_node.child_ids):
                            if pr_ch == node.id:
                                pr_node.child_ids[i] = node.child_ids[0]
                                break

                    remove_ids.append(node.id)
                    node.child_ids = []

            remove_ids.sort()
            remove_ids.reverse()

            for id in remove_ids:
                for node in self.node_set:
                    if node.id > id:
                        node.id -= 1
                    for i, ch in enumerate(node.child_ids):
                        if ch > id:
                            node.child_ids[i] -= 1

                if root_id > id:
                    root_id -= 1

            for id in remove_ids:
                del self.node_set[id]

            self._AssignParentIds()

            self.DFS = []
            self.BFS = []
            visited = np.zeros(len(self.node_set))
            self._DFS(root_id, visited)
            visited = np.zeros(len(self.node_set))
            self._BFS(root_id, visited)


        # generate colors for terminal nodes
        self.TNodeColors = {}
        for node in self.node_set:
            if node.node_type == NodeType.TerminalNode:
                self.TNodeColors[node.id] = (
                    random.random(), random.random(), random.random())  # generate a random color

        # index of Or-nodes in BFS
        self.OrNodeIdxInBFS = {}
        self.TNodeIdxInBFS = {}
        idx_or = 0
        idx_t = 0
        for id in self.BFS:
            node = self.node_set[id]
            if node.node_type == NodeType.OrNode:
                self.OrNodeIdxInBFS[node.id] = idx_or
                idx_or += 1
            elif node.node_type == NodeType.TerminalNode:
                self.TNodeIdxInBFS[node.id] = idx_t
                idx_t += 1

        self._GetParts()
        self._GetMatrixForm()

        print("======= create AOGrid, done")

    def GetConfiguration(self, det, pg, tnode_scores, offset, trans_std):
        wd = det[2] - det[0] + 1
        ht = det[3] - det[1] + 1
        cell_wd = wd / self.param.grid_wd
        cell_ht = ht / self.param.grid_ht

        # get the parse graph
        configuration = np.empty((0, 4), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float32)
        tnode_score = []
        BFS = [self.BFS[0]]
        # print('---------------------------------------------------')
        while len(BFS):
            id = BFS.pop()
            node = self.node_set[id]
            if node.node_type == NodeType.OrNode:
                idx = self.OrNodeIdxInBFS[node.id]
                BFS.append(node.child_ids[int(pg[idx])])
            elif node.node_type == NodeType.AndNode:
                BFS += node.child_ids
            else:
                # if self.param.use_tnode_as_alpha_channel and len(node.parent_ids) == 1 and self.node_set[
                #     node.parent_ids[0]].node_type == NodeType.AndNode:
                #     continue
                rect = self.primitive_set[node.rect_idx]
                if offset is not None:
                    offset_ind = 0
                    if offset.shape[1] == len(self.part_type):
                        offset_ind = self.part_type.index([rect.Height(), rect.Width()])
                    elif offset.shape[1] == self.part_instance.shape[0]:
                        for node1 in self.node_set:
                            if node1.node_type == NodeType.TerminalNode:  # change to BFS after _part_instance is changed to BFS
                                if node1.id == node.id:
                                    break
                                offset_ind += 1
                    else:
                        raise ValueError("Wrong offsets")

                    # print(offset[0, offset_ind, 0], offset[1, offset_ind, 0])
                    offset_x = offset[0, offset_ind, 0] * trans_std * wd
                    offset_y = offset[1, offset_ind, 0] * trans_std * ht

                    tile_half_w = (rect.x2 - rect.x1 + 1) * cell_wd / 2.0
                    tile_half_h = (rect.y2 - rect.y1 + 1) * cell_ht / 2.0

                    # offset_x = min(max(offset_x, -tile_half_w), tile_half_w)
                    # offset_y = min(max(offset_y, -tile_half_h), tile_half_h)
                    #
                    offset_x = 0
                    offset_y = 0

                else:
                    offset_x = 0
                    offset_y = 0

                box = [det[0] + rect.x1 * cell_wd + offset_x,
                       det[1] + rect.y1 * cell_ht + offset_y,
                       det[0] + (rect.x2 + 1) * cell_wd + offset_x,
                       det[1] + (rect.y2 + 1) * cell_ht + offset_y]
                configuration = np.vstack((configuration, box))
                colors = np.vstack((colors, self.TNodeColors[node.id]))

                idx = self.TNodeIdxInBFS[node.id]
                score = tnode_scores[idx]
                tnode_score.append(score)


        return configuration, colors, tnode_score

    def PictureConfiguration(self, config, save_name, input_bbox=None):
        save_dir = os.path.split(save_name)[0]
        assert os.path.exists(save_dir), 'not found {}'.format(save_dir)

        if input_bbox is None:
            input_bbox = np.array([self.param.grid_ht, self.param.grid_wd]) * 20

        bin_ht = min(40, max(20, int(round(input_bbox[0] / self.param.grid_ht))))
        bin_wd = min(40, max(20, int(round(input_bbox[1] / self.param.grid_wd))))

        line_wd = 3

        ht = self.param.grid_ht * (bin_ht + line_wd) + line_wd
        wd = self.param.grid_wd * (bin_wd + line_wd) + line_wd


        templ = np.ones((ht, wd, 3), dtype=np.uint8) * 255
        xx = 0
        for x in range(self.param.grid_wd + 1):
            templ[:, xx:(xx + line_wd), :] = 0
            xx += bin_wd + line_wd

        yy = 0
        for y in range(self.param.grid_ht + 1):
            templ[yy:(yy + line_wd), :, :] = 0
            yy += bin_ht + line_wd


        for id in config[:]:
            if id == -1:
                break
            node = self.node_set[id]
            assert node.node_type == NodeType.TerminalNode

            rect = self.primitive_set[node.rect_idx]
            x1 = int(rect.x1 * (bin_wd + line_wd) + line_wd)
            y1 = int(rect.y1 * (bin_ht + line_wd) + line_wd)
            x2 = int((rect.x2 + 1) * (bin_wd + line_wd) + 1)
            y2 = int((rect.y2 + 1) * (bin_ht + line_wd) + 1)

            for c in range(3):
                templ[y1:y2, x1:x2, c] = int(min(255, self.TNodeColors[id][c] * 255))

        cv2.imwrite(save_name, templ)


    def PictureNodes(self, save_dir, input_bbox=None):
        assert os.path.exists(save_dir), 'not found {}'.format(save_dir)
        if input_bbox is None:
            input_bbox = np.array([self.param.grid_ht, self.param.grid_wd]) * 20

        bin_ht = min(40, max(20, int(round(input_bbox[0] / self.param.grid_ht))))
        bin_wd = min(40, max(20, int(round(input_bbox[1] / self.param.grid_wd))))

        line_wd = 3

        ht = self.param.grid_ht * (bin_ht + line_wd) + line_wd
        wd = self.param.grid_wd * (bin_wd + line_wd) + line_wd

        save_dir = os.path.join(save_dir, 'pictureAOG_{}_{}_{}'.format(ht, wd, self.param.tag))
        if os.path.exists(save_dir):
            return save_dir
        else:
            os.makedirs(save_dir)

        templ = np.ones((ht, wd, 3), dtype=np.uint8) * 255
        xx = 0
        for x in range(self.param.grid_wd + 1):
            templ[:, xx:(xx + line_wd), :] = 0
            xx += bin_wd + line_wd

        yy = 0
        for y in range(self.param.grid_ht + 1):
            templ[yy:(yy + line_wd), :, :] = 0
            yy += bin_ht + line_wd

        filename = os.path.join(save_dir, 'grid.png')
        cv2.imwrite(filename, templ)

        # images for T-nodes and Or-nodes
        for node in self.node_set:
            if node.node_type == NodeType.AndNode or node.rect_idx==-1:
                continue

            rect = self.primitive_set[node.rect_idx]
            x1 = int(rect.x1 * (bin_wd + line_wd) + line_wd)
            y1 = int(rect.y1 * (bin_ht + line_wd) + line_wd)
            x2 = int((rect.x2 + 1) * (bin_wd + line_wd) + 1)
            y2 = int((rect.y2 + 1) * (bin_ht + line_wd) + 1)

            img = templ.copy()
            if node.node_type == NodeType.TerminalNode:
                diff = 120
            else:
                diff = 80
            img[y1:y2, x1:x2, :] -= diff

            filename = os.path.join(save_dir, '{:04d}.png'.format(node.id))
            cv2.imwrite(filename, img)

        # images for And-nodes
        margin = 1
        for node in self.node_set:
            if node.node_type != NodeType.AndNode:
                continue

            # first child
            ch = self.node_set[node.child_ids[0]]
            rect = self.primitive_set[ch.rect_idx]

            x1 = int(rect.x1 * (bin_wd + line_wd) + line_wd)
            y1 = int(rect.y1 * (bin_ht + line_wd) + line_wd)
            x2 = int((rect.x2 + 1) * (bin_wd + line_wd) + 1)
            y2 = int((rect.y2 + 1) * (bin_ht + line_wd) + 1)

            tx1 = min(x1 + margin, wd - 1)
            ty1 = min(y1 + margin, ht - 1)
            tx2 = max(x2 - margin, 0)
            ty2 = max(y2 - margin, 0)

            img = templ.copy()
            img[ty1:ty2, tx1:tx2, :] -= 100

            # second child
            ch = self.node_set[node.child_ids[1]]
            rect = self.primitive_set[ch.rect_idx]

            xx1 = int(rect.x1 * (bin_wd + line_wd) + line_wd)
            yy1 = int(rect.y1 * (bin_ht + line_wd) + line_wd)
            xx2 = int((rect.x2 + 1) * (bin_wd + line_wd) + 1)
            yy2 = int((rect.y2 + 1) * (bin_ht + line_wd) + 1)

            tx1 = min(xx1 + margin, wd - 1)
            ty1 = min(yy1 + margin, ht - 1)
            tx2 = max(xx2 - margin, 0)
            ty2 = max(yy2 - margin, 0)

            img[ty1:ty2, tx1:tx2, :] -= 160

            # overlapping area
            val = 200

            img[y1:y2, x1, :] = [0, val, val]
            img[y1:y2, x2, :] = [0, val, val]
            img[y1, x1:x2, :] = [0, val, val]
            img[y2, x2:x2, :] = [0, val, val]

            img[yy1:yy2, xx1, :] = [val, val, 0]
            img[yy1:yy2, xx2, :] = [val, val, 0]
            img[yy1, xx1:xx2, :] = [val, val, 0]
            img[yy2, xx1:xx2, :] = [val, val, 0]

            ox1 = max(x1, xx1)
            oy1 = max(y1, yy1)
            ox2 = min(x2, xx2)
            oy2 = min(y2, yy2)

            if ox1 <= ox2 and oy1 <= oy2:
                tx1 = min(ox1 + margin, wd - 1)
                ty1 = min(oy1 + margin, ht - 1)
                tx2 = max(ox2 - margin, 0)
                ty2 = max(oy2 - margin, 0)

                img[ty1:ty2, tx1:tx2, :] -= val

            filename = os.path.join(save_dir, '{:04d}.png'.format(node.id))
            cv2.imwrite(filename, img)

        return save_dir

    def Visualize(self, save_dir, filename=None, use_weighted_edge=True):
        if filename is None and not os.path.exists(save_dir):
            print("filename is not specified or not found save dir {}".format(save_dir))
            return

        if filename is None:
            filename = os.path.join(save_dir, "AOG_{}_{}.dot".format(self.param.grid_ht, self.param.grid_wd))
        elif not os.path.exists(save_dir):
            save_dir = os.path.split(filename)[0]

        node_img_dir = self.PictureNodes(save_dir)
        #node_img_dir = self.PictureNodes(os.path.dirname(__file__))
        class_distr_img_dir = os.path.join(save_dir, 'pictureWhichClassesVisitedNodes')

        with open(filename, "w") as f:
            f.write('digraph AOG {\n')
            for node in self.node_set:
                if node.on_off:
                    img_file = os.path.join(node_img_dir, '{:04d}.png'.format(node.id))
                    clss_distr_img_file = os.path.join(class_distr_img_dir, '{:04d}.png'.format(node.id))
                    if node.node_type == NodeType.OrNode:
                        if node.rect_idx == -1:
                            f.write(
                                'node{} [shape=ellipse, style=bold, color=green]\n'.format(node.id))
                        else:
                            if node.id == 0:
                                img_file = os.path.join(node_img_dir, 'grid.png')
                            if not os.path.exists(clss_distr_img_file) or len(node.child_ids) == 1:
                                f.write(
                                    'node{} [shape=ellipse, style=bold, color=green, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                        node.id, img_file))
                            else:
                                f.write(
                                    'node{} [shape=ellipse, style=bold, color=green, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                        node.id, img_file, clss_distr_img_file))

                    elif node.node_type == NodeType.AndNode:
                        if node.id == self.node_set[self.BFS[0]].id:
                            img_file = os.path.join(node_img_dir, 'grid.png')
                        if not os.path.exists(clss_distr_img_file):
                            f.write(
                                'node{} [shape=ellipse, style=filled, color=blue, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                    node.id, img_file))
                        else:
                            f.write(
                                'node{} [shape=ellipse, style=filled, color=blue, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                    node.id, img_file, clss_distr_img_file))
                    elif node.node_type == NodeType.TerminalNode:
                        # if self.param.use_tnode_as_alpha_channel and len(node.parent_ids) == 1 and self.node_set[
                        #     node.parent_ids[0]].node_type == NodeType.AndNode:
                        #     continue

                        if not os.path.exists(clss_distr_img_file):
                            f.write(
                                'node{} [shape=box, style=bold, color=red, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                    node.id, img_file))
                        else:
                            f.write(
                                'node{} [shape=box, style=bold, color=red, label=<<TABLE border=\"0\" cellborder=\"0\"><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR><TR><TD PORT=\"f0\"><IMG SRC=\"{}\"/></TD></TR></TABLE>>]\n'.format(
                                    node.id, img_file, clss_distr_img_file))
                    else:
                        print("Wrong node type")
                        raise RuntimeError

            for node in self.node_set:
                if node.on_off:
                    if node.node_type == NodeType.OrNode:
                        f.write('edge [style=bold, color=green]\n')
                    elif node.node_type == NodeType.AndNode:
                        f.write('edge [style=bold, color=blue]\n')
                    elif node.node_type == NodeType.TerminalNode:
                        f.write('edge [style=bold, color=red]\n')
                    else:
                        print("Wrong node type")
                        raise RuntimeError

                    for c, i in enumerate(node.child_ids):
                        # if self.param.use_tnode_as_alpha_channel and node.node_type==NodeType.AndNode \
                        #         and self.node_set[i].node_type == NodeType.TerminalNode:
                        #     continue

                        if self.node_set[i].on_off:
                            if len(node.out_edge_visited_count) and node.out_edge_visited_count[
                                c] > 0 and use_weighted_edge:
                                penwidth = max(1, math.log10(node.out_edge_visited_count[c]))
                                f.write(
                                    'node{} -> node{} [penwidth={}, label=\"{:.4f}\"]\n'.format(node.id, i, penwidth,
                                                                                                node.out_edge_visited_count[
                                                                                                    c]))
                            else:
                                f.write('node{} -> node{}\n'.format(node.id, i))

            f.write('}')

        return filename


if __name__ == '__main__':
    param = Param(grid_ht=3, grid_wd=3, not_use_large_TerminalNode=False, turn_off_size_ratio_TerminalNode=0.33,
                  use_root_TerminalNode=False, use_tnode_as_alpha_channel=False)
    g = AOGrid(param)
    g.Create()
    print(g.matrix_form)
    filename = g.Visualize("plot-2d")
    os.system('dot -Tpdf {} -o {}.pdf'.format(filename, filename))
