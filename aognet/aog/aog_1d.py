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

def get_aog(dim, min_size=1, use_root_tnode=True, tnode_max_size=100, turn_off_unit_or_node=False):
    aog = AOGrid(dim=dim, min_size=min_size, use_root_terminal_node=use_root_tnode, tnode_max_size=tnode_max_size,
                 turn_off_unit_or_node=turn_off_unit_or_node)
    aog.Create()
    return aog

class NodeType(object):
    OrNode = "OrNode"
    AndNode = "AndNode"
    TerminalNode = "TerminalNode"
    Unknow = "Unknown"

class SplitType(object):
    Split = "Split"
    Unknown = "Unknown"

class Array(object):
    """A simple rectangle
    """

    def __init__(self, x1=0, x2=0):
        self.x1 = x1
        self.x2 = x2

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

    def Length(self):
        return self.x2 - self.x1 + 1

    def IsOverlap(self, other):
        assert isinstance(other, self.__class__)

        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        if x1 > x2:
            return False

        return True

    def IsSame(self, other):
        assert isinstance(other, self.__class__)

        return self.Length() == other.Length()


class Node(object):
    """Types of nodes in an AOGrid
    AND-node (structural decomposition),
    OR-node (alternative decompositions),
    TERMINAL-node (link to data appearance).
    """

    def __init__(self, id=-1, node_type=NodeType.Unknow, array_idx=-1, child_ids=[], parent_ids=[],
                 split_type=SplitType.Unknown, split_step1=0, split_step2=0):
        self.id = id
        self.node_type = node_type
        self.array_idx = array_idx
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
            return (self.node_type == other.node_type) and (self.array_idx == other.array_idx) and \
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
    """The 1d AOGrid defines a Directed Acyclic And-Or Grap
    which is used to explore/unfold the space of latent structures
    of a 1d grid (e.g., a array with size 5)
    """

    def __init__(self, dim, min_size=1, overlap_ratio=0, use_root_terminal_node=False, tnode_max_size=100,
                 turn_off_unit_or_node=True):
        self.dim = dim
        self.min_size = min_size
        self.tnode_max_size = tnode_max_size
        self.overlap_ratio = overlap_ratio
        self.use_root_terminal_node = use_root_terminal_node
        self.turn_off_unit_or_node = turn_off_unit_or_node
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
        self._part_instance = None  # a part instance of the type (len,) is defined by (x, len,)
        self._part_type = None  # a part type is defined by (len,)
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
        self._part_instance = np.empty((0, 2), dtype=np.float32)
        self._part_instance_with_type = np.empty((0, 3), dtype=np.float32)
        self._part_type = []
        # TODO: change to use BFS order
        for node in self.node_set:
            if node.node_type == NodeType.TerminalNode:
                arr = self.primitive_set[node.array_idx]
                self._part_instance = np.vstack((self._part_instance, np.array([arr.x1, arr.x2])))
                p = arr.Length()
                if p not in self._part_type:
                    self._part_type.append(p)
                idx = self._part_type.index(p)
                self._part_instance_with_type = np.vstack((self._part_instance_with_type,
                                                          np.array([arr.x1, arr.x2, idx])))

    def _AddPrimitve(self, arr):
        assert isinstance(arr, Array)

        if arr in self.primitive_set:
            return self.primitive_set.index(arr)

        self.primitive_set.append(arr)

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

    def _DoSplit(self, arr):
        assert isinstance(arr, Array)
        return arr.Length() >= self.min_size

    def _SplitStep(self, sz):
        if sz >= self.min_size:
            return 1
        else:
            return ceil(self.min_size / sz)    ## ???????

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


    def Create(self):
        print("======= creating AOGrid, could take a while")

        # the root OrNode
        arr = Array(0, self.dim-1)
        self.primitive_set.append(arr)
        node = Node(node_type=NodeType.OrNode, array_idx=0)
        self._AddNode(node)

        BFS = deque()
        BFS.append(0)
        while len(BFS) > 0:
            curId = BFS.popleft()
            curNode = self.node_set[curId]
            curArr = self.primitive_set[curNode.array_idx]
            curLen = curArr.Length()

            childIds = []

            if curNode.node_type == NodeType.OrNode:
                # add a terminal node for a non-root OrNode
                if self.use_root_terminal_node:
                    cmp = -1
                else:
                    cmp = 0
                if curId > cmp and curLen <= self.tnode_max_size:
                    node = Node(node_type=NodeType.TerminalNode, array_idx=curNode.array_idx)
                    suc, node = self._AddNode(node)
                    childIds.append(node.id)

                # add all AndNodes for horizontal and vertical binary splits
                if not self._DoSplit(curArr):
                    print(curId)
                    continue

                # splits
                step = self._SplitStep(curLen)
                for leftLen in range(step, curLen - step + 1):
                    rightLen = curLen - leftLen
                    if self.overlap_ratio > 0:
                        numSplit = int(1 + floor(leftLen * self.overlap_ratio))
                    else:
                        numSplit = 1
                    for b in range(0, numSplit):
                        node = Node(node_type=NodeType.AndNode, array_idx=curNode.array_idx,
                                    split_type=SplitType.Split,
                                    split_step1=leftLen, split_step2=curLen - rightLen)
                        suc, node = self._AddNode(node)
                        if suc:
                            BFS.append(node.id)

                        childIds.append(node.id)
                        rightLen += 1

                childIds = list(set(childIds))

            elif curNode.node_type == NodeType.AndNode:
                # add two child OrNodes
                if curNode.split_type == SplitType.Split:
                    left = Array(curArr.x1, curArr.x1 + curNode.split_step1 - 1)
                    if self.turn_off_unit_or_node and left.Length() == 1:
                        node = Node(node_type=NodeType.TerminalNode, array_idx=self._AddPrimitve(left))
                    else:
                        node = Node(node_type=NodeType.OrNode, array_idx=self._AddPrimitve(left))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                    right = Array(curArr.x1 + curNode.split_step2, curArr.x2)
                    if self.turn_off_unit_or_node and right.Length() == 1:
                        node = Node(node_type=NodeType.TerminalNode, array_idx=self._AddPrimitve(right))
                    else:
                        node = Node(node_type=NodeType.OrNode, array_idx=self._AddPrimitve(right))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

            self.node_set[curId].child_ids = childIds

        root_id = 0

        self._AssignParentIds()

        visited = np.zeros(len(self.node_set))
        self._DFS(root_id, visited)
        visited = np.zeros(len(self.node_set))
        self._BFS(root_id, visited)

        # deleted use_tnode_as_alpha_channel and use_super_or_nodes

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

    def PictureNodes(self, save_dir, input_bbox=None):
        assert os.path.exists(save_dir), 'not found {}'.format(save_dir)
        if input_bbox is None:
            input_bbox = np.array([1, self.dim]) * 20

        bin_ht = min(40, max(20, int(round(input_bbox[0] / 1))))
        bin_wd = min(40, max(20, int(round(input_bbox[1] / self.dim))))

        line_wd = 3

        ht = 1 * (bin_ht + line_wd) + line_wd
        wd = self.dim * (bin_wd + line_wd) + line_wd

        save_dir = os.path.join(save_dir, 'pictureAOG_{}_{}'.format(ht, wd))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        templ = np.ones((ht, wd, 3), dtype=np.uint8) * 255
        xx = 0
        for x in range(self.dim + 1):
            templ[:, xx:(xx + line_wd), :] = 0
            xx += bin_wd + line_wd

        yy = 0
        for y in range(2):
            templ[yy:(yy + line_wd), :, :] = 0
            yy += bin_ht + line_wd

        filename = os.path.join(save_dir, 'grid.png')
        cv2.imwrite(filename, templ)

        # images for T-nodes and Or-nodes
        for node in self.node_set:
            if node.node_type == NodeType.AndNode or node.array_idx==-1:
                continue

            arr = self.primitive_set[node.array_idx]
            x1 = int(arr.x1 * (bin_wd + line_wd) + line_wd)
            y1 = int(0 * (bin_ht + line_wd) + line_wd)
            x2 = int((arr.x2 + 1) * (bin_wd + line_wd) + 1)
            y2 = int((0 + 1) * (bin_ht + line_wd) + 1)

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
            arr = self.primitive_set[ch.array_idx]

            x1 = int(arr.x1 * (bin_wd + line_wd) + line_wd)
            y1 = int(0 * (bin_ht + line_wd) + line_wd)
            x2 = int((arr.x2 + 1) * (bin_wd + line_wd) + 1)
            y2 = int((0 + 1) * (bin_ht + line_wd) + 1)

            tx1 = min(x1 + margin, wd - 1)
            ty1 = min(y1 + margin, ht - 1)
            tx2 = max(x2 - margin, 0)
            ty2 = max(y2 - margin, 0)

            img = templ.copy()
            img[ty1:ty2, tx1:tx2, :] -= 100

            # second child
            ch = self.node_set[node.child_ids[1]]
            arr = self.primitive_set[ch.array_idx]

            xx1 = int(arr.x1 * (bin_wd + line_wd) + line_wd)
            yy1 = int(0 * (bin_ht + line_wd) + line_wd)
            xx2 = int((arr.x2 + 1) * (bin_wd + line_wd) + 1)
            yy2 = int((0 + 1) * (bin_ht + line_wd) + 1)

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
            filename = os.path.join(save_dir, "AOG_{}.dot".format(self.dim))
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
                        if node.array_idx == -1:
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
    aog = get_aog(dim=4, min_size=1, use_root_tnode=True, tnode_max_size=4, turn_off_unit_or_node=False)
    print(aog.matrix_form)
    filename = aog.Visualize("plot-1d")
    os.system('dot -Tpdf {} -o {}.pdf'.format(filename, filename))
