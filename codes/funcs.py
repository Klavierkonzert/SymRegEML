# -*- coding: utf-8 -*-
"""

copyright Ying Jin, Weilin Fu, Jian Kang, Jiadong Guo, Jian Guo
@20190804 added epsilon to avoid singular matrix
@20190804 fix bug, change scale=np.max(XX) to scale = np.max(np.abs(XX))
@20190804 added running time record
@20190804 added stop condition in the algo's loop to prevent running for too long
"""

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.stats import norm
import sklearn
from sklearn.datasets import load_iris
import copy
#import pymc3 as pm
import random
#import data_generate_funcs as dg
import time

SAFE_VALUE = 1e12
SAFE_LOG_VALUE = np.log(SAFE_VALUE)
SAFE_SQUARE_INPUT = np.sqrt(SAFE_VALUE)
SAFE_CUBIC_INPUT = np.cbrt(SAFE_VALUE)


def safe_array(data):
    if np.iscomplexobj(data):
        r = np.nan_to_num(np.clip(data.real, -SAFE_VALUE, SAFE_VALUE), nan=0.0, posinf=SAFE_VALUE, neginf=-SAFE_VALUE)
        i = np.nan_to_num(np.clip(data.imag, -SAFE_VALUE, SAFE_VALUE), nan=0.0, posinf=SAFE_VALUE, neginf=-SAFE_VALUE)
        return r + 1j * i
    else:
        return np.nan_to_num(np.clip(data, -SAFE_VALUE, SAFE_VALUE), nan=0.0, posinf=SAFE_VALUE, neginf=-SAFE_VALUE)


class Operator:
    def __init__(self, name, function, arity):
        self.name = name
        self.func = function
        self.arity = arity  # num of inputs


class Node:
    def __init__(self, depth):
        # tree structure attributes
        self.type = -1
        # -1 represents newly grown node (not decided yet)
        # 0 represents no child, as a terminal node
        # 1 represents one child,
        # 2 represents 2 children
        self.order = 0
        self.left = None
        self.right = None
        # if type=1, the left child is the only one
        self.depth = depth
        self.parent = None

        # calculation attributes
        self.operator = None
        self.op_ind = None
        # operator is a string, either "+","*","ln","exp","inv"
        self.data = None
        self.feature = None
        self.constant = None
        # feature is a int indicating the index of feature in the input data
        # possible parameters
        self.a = None
        self.b = None

    def inform(self):
        print("order:", self.order)
        print("type:", self.type)
        print("depth:", self.depth)
        print("operator:", self.operator)
        print("data:", self.data)
        print("feature:", self.feature)
        if self.operator == 'ln':
            print(" ln_a:", self.a)
            print(" ln_b:", self.b)

        return


# =============================================================================
# # grow from a node, assign an operator or stop as terminal
# =============================================================================

def terminal_count(nfeature, const_terminal=False):
    return nfeature + int(const_terminal)


def get_terminal_logprob(chosen_val, prior_dict, nfeature, const_terminal):
    # Build allowed terminals
    terminals = []
    if const_terminal:
        terminals.append(1.0)
    for i in range(nfeature):
        terminals.append(i)
        
    # Unnormalized probabilities
    probs = []
    prior_keys = list(prior_dict.keys())
    valid_prior_keys = [k for k in prior_keys if k in terminals]
    sum_prior_prob = sum(prior_dict[k] for k in valid_prior_keys)
    
    remaining_terminals = [t for t in terminals if t not in valid_prior_keys]
    
    for t in terminals:
        if t in valid_prior_keys:
            probs.append(prior_dict[t])
        else:
            if len(remaining_terminals) > 0:
                probs.append(max(0.0, 1.0 - sum_prior_prob) / len(remaining_terminals))
            else:
                probs.append(0.0)
                
    probs = np.array(probs)
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = np.ones(len(terminals)) / len(terminals)
        
    idx = -1
    for i, t in enumerate(terminals):
        if type(t) == type(chosen_val) and t == chosen_val:
            idx = i
            break
    if idx == -1:
        for i, t in enumerate(terminals):
            if t == chosen_val:
                idx = i
                break
    if idx != -1:
        return np.log(max(1e-15, probs[idx]))
    else:
        return -np.log(len(terminals))


def get_root(node):
    curr = node
    while curr.parent is not None:
        curr = curr.parent
    return curr

def get_right_left_prior(node, eml_unary_chain_prior=False):
    if not eml_unary_chain_prior or node.parent is None:
        return 1.0, None
        
    parent_op = getattr(node.parent.operator, 'name', node.parent.operator)
    if parent_op != 'exp_minus_log':
        return 1.0, None
        
    if node == node.parent.left:
        # Left child uses default symmetric split scale
        # But strongly prefers 1.0 if it decides to terminate
        return 1.0, {1.0: 0.95} 
    else:
        # Right child is conditionally dependent on the left node
        left_node = node.parent.left
        if left_node is None:
            return 1.0, {1.0: 0.95}
            
        if left_node.type == 0:
            # Left node terminated. Right node MUST split.
            return 20.0, {1.0: 0.95}
        else:
            # Left node split. Right node MUST terminate.
            return 0.05, {1.0: 0.95}

def assign_terminal(node, nfeature, const_terminal=False, left_prior=None, right_prior=None):
    # Check parent operator
    prior_dict = None
    eml_unary_chain_prior = getattr(get_root(node), 'eml_unary_chain_prior', False)
    
    if node.parent is not None:
        parent_op = getattr(node.parent.operator, 'name', node.parent.operator)
        
        if eml_unary_chain_prior and parent_op == 'exp_minus_log':
            _, prior_dict_cond = get_right_left_prior(node, eml_unary_chain_prior)
            if prior_dict_cond is not None:
                prior_dict = prior_dict_cond
        else:
            # If node is left child
            if node == node.parent.left:
                if left_prior is not None and parent_op in left_prior:
                    prior_dict = left_prior[parent_op]
            # If node is right child
            elif node == node.parent.right:
                if right_prior is not None and parent_op in right_prior:
                    prior_dict = right_prior[parent_op]

    if prior_dict is not None:
        # Build allowed terminals
        terminals = []
        if const_terminal:
            terminals.append(1.0)
        for i in range(nfeature):
            terminals.append(i)
            
        # Sum of probabilities of terminals explicitly specified in prior_dict
        prior_keys = list(prior_dict.keys())
        valid_prior_keys = [k for k in prior_keys if k in terminals]
        sum_prior_prob = sum(prior_dict[k] for k in valid_prior_keys)
        
        remaining_terminals = [t for t in terminals if t not in valid_prior_keys]
        
        probs = []
        for t in terminals:
            if t in valid_prior_keys:
                probs.append(prior_dict[t])
            else:
                if len(remaining_terminals) > 0:
                    probs.append(max(0.0, 1.0 - sum_prior_prob) / len(remaining_terminals))
                else:
                    probs.append(0.0)
                    
        probs = np.array(probs)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(len(terminals)) / len(terminals)
            
        idx = np.random.choice(len(terminals), p=probs)
        selected_terminal = terminals[idx]
    else:
        # Default uniform choice
        num_terms = terminal_count(nfeature, const_terminal)
        terminal = np.random.randint(0, num_terms, size=1)[0]
        if const_terminal:
            if terminal == nfeature:
                selected_terminal = 1.0
            else:
                selected_terminal = int(terminal)
        else:
            selected_terminal = int(terminal)
            
    node.type = 0
    node.operator = None
    node.left = None
    node.right = None
    node.a = None
    node.b = None
    if selected_terminal == 1.0 and isinstance(selected_terminal, float):
        node.feature = None
        node.constant = 1.0
    else:
        node.feature = np.array([selected_terminal])
        node.constant = None
    return


def grow(node, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal=False, max_depth=None, max_complexity=None, node_count=None, left_prior=None, right_prior=None):
    depth = node.depth

    # Decide budget
    budget = max_complexity - node_count[0] if (max_complexity is not None and node_count is not None) else float('inf')

    # deciding the number of child nodes
    if max_depth is not None and node.depth >= max_depth:
        assign_terminal(node, nfeature, const_terminal, left_prior, right_prior)
    elif node.depth > 0:
        eml_unary_chain_prior = getattr(get_root(node), 'eml_unary_chain_prior', False)
        scale, _ = get_right_left_prior(node, eml_unary_chain_prior)
        prob = min(0.9999, scale / np.power((1 + depth), -beta))

        test = np.random.uniform(0, 1, 1)
        if test > prob or budget <= 0:  # terminal
            assign_terminal(node, nfeature, const_terminal, left_prior, right_prior)
        else:
            # Filter operators based on remaining budget
            allowed_indices = []
            for idx, op_type in enumerate(Op_type):
                if op_type == 1 and budget >= 1:
                    allowed_indices.append(idx)
                elif op_type == 2 and budget >= 2:
                    allowed_indices.append(idx)
            
            if len(allowed_indices) == 0:
                assign_terminal(node, nfeature, const_terminal, left_prior, right_prior)
            else:
                allowed_weights = np.array([Op_weights[idx] for idx in allowed_indices])
                sum_weights = allowed_weights.sum()
                if sum_weights > 0:
                    allowed_weights = allowed_weights / sum_weights
                else:
                    allowed_weights = np.ones(len(allowed_indices)) / len(allowed_indices)
                
                op_ind = np.random.choice(allowed_indices, p=allowed_weights)
                node.operator = Ops[op_ind]
                node.type = Op_type[op_ind]
                node.op_ind = op_ind

    else:  # root node, sure to split
        # Filter operators based on remaining budget
        allowed_indices = []
        for idx, op_type in enumerate(Op_type):
            if op_type == 1 and budget >= 1:
                allowed_indices.append(idx)
            elif op_type == 2 and budget >= 2:
                allowed_indices.append(idx)
        
        if len(allowed_indices) == 0:
            assign_terminal(node, nfeature, const_terminal, left_prior, right_prior)
        else:
            allowed_weights = np.array([Op_weights[idx] for idx in allowed_indices])
            sum_weights = allowed_weights.sum()
            if sum_weights > 0:
                allowed_weights = allowed_weights / sum_weights
            else:
                allowed_weights = np.ones(len(allowed_indices)) / len(allowed_indices)
            
            op_ind = np.random.choice(allowed_indices, p=allowed_weights)
            node.operator = Ops[op_ind]
            node.type = Op_type[op_ind]
            node.op_ind = op_ind

    # grow recursively
    if node.type == 0:
        if node.feature is None and node.constant is None:
            assign_terminal(node, nfeature, const_terminal, left_prior, right_prior)

    elif node.type == 1:
        if node_count is not None:
            node_count[0] += 1
        node.left = Node(depth + 1)
        node.left.parent = node
        if node.operator == 'ln':  # linear parameters
            node.a = norm.rvs(loc=1, scale=np.sqrt(sigma_a))
            node.b = norm.rvs(loc=0, scale=np.sqrt(sigma_b))
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, max_complexity, node_count, left_prior, right_prior)

    else:  # node.type=2
        if node_count is not None:
            node_count[0] += 2
        node.left = Node(depth + 1)
        node.left.parent = node
        # node.left.order = len(Tree)
        node.right = Node(depth + 1)
        node.right.parent = node
        # node.right.order = len(Tree)
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, max_complexity, node_count, left_prior, right_prior)
        grow(node.right, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, max_complexity, node_count, left_prior, right_prior)

    return


# =============================================================================
# # generate a list storing the nodes in the tree
# # nodes are stored by induction
# # orders are assigned accordingly
# =============================================================================
def genList(node):
    lst = []
    # terminal node
    if node.left is None:
        lst.append(node)
    else:
        if node.right is None:  # with one child
            lst.append(node)
            lst = lst + genList(node.left)
        else:  # with two children
            lst.append(node)
            lst = lst + genList(node.left)
            lst = lst + genList(node.right)
    for i in np.arange(0, len(lst)):
        lst[i].order = i
    return (lst)


# =============================================================================
# # cut all child nodes of a current node
# # turn the node into a terminal one
# =============================================================================
def shrink(node):
    if node.left is None:
        print("Already a terminal node!")
    else:
        node.left = None
        node.right = None
        node.type = 0
        node.operator = None  # delete operator
        node.a = None  # delete parameters
        node.b = None
    return


# =============================================================================
# # upgrade 'order' attribute of nodes in Tree
# # Tree is a list containing nodes of a tree
# =============================================================================
def upgOd(Tree):
    for i in np.arange(0, len(Tree)):
        Tree[i].order = i
    return


# =============================================================================
# # calculate a tree output from node
# =============================================================================
def allcal(node, indata):
    if node.type == 0:  # terminal node
        if indata is not None:
            if node.constant is not None:
                node.data = np.ones((indata.shape[0], 1), dtype=indata.values.dtype) * node.constant
            else:
                node.data = np.array(indata.iloc[:, node.feature], dtype=indata.values.dtype)
    elif node.type == 1:  # one child node
        if node.operator == 'ln':
            node.data = node.a * allcal(node.left, indata) + node.b
        elif node.operator == 'exp':
            left = allcal(node.left, indata)
            if np.iscomplexobj(left):
                node.data = np.exp(np.clip(left.real, -SAFE_LOG_VALUE, SAFE_LOG_VALUE) + 1j * left.imag)
            else:
                node.data = np.exp(np.clip(left, -SAFE_LOG_VALUE, SAFE_LOG_VALUE))
        elif node.operator == 'one':
            child_data = allcal(node.left, indata)
            node.data = np.ones_like(child_data)
        elif node.operator == 'inv':
            node.data = allcal(node.left, indata)
            with np.errstate(divide='ignore', invalid='ignore'):
                node.data = np.where(np.abs(node.data) < 1e-12, 0.0, 1.0 / node.data)
        elif node.operator == 'neg':
            node.data = -1 * allcal(node.left, indata)
        elif node.operator == 'sin':
            node.data = np.sin(allcal(node.left, indata))
        elif node.operator == 'cos':
            node.data = np.cos(allcal(node.left, indata))
        elif node.operator == 'square': ## operator added by fwl
            left = allcal(node.left, indata)
            if np.iscomplexobj(left):
                left = np.clip(left.real, -SAFE_SQUARE_INPUT, SAFE_SQUARE_INPUT) + 1j * np.clip(left.imag, -SAFE_SQUARE_INPUT, SAFE_SQUARE_INPUT)
            else:
                left = np.clip(left, -SAFE_SQUARE_INPUT, SAFE_SQUARE_INPUT)
            node.data = np.square(left)
        elif node.operator == 'cubic': ## operator added by fwl
            left = allcal(node.left, indata)
            if np.iscomplexobj(left):
                left = np.clip(left.real, -SAFE_CUBIC_INPUT, SAFE_CUBIC_INPUT) + 1j * np.clip(left.imag, -SAFE_CUBIC_INPUT, SAFE_CUBIC_INPUT)
            else:
                left = np.clip(left, -SAFE_CUBIC_INPUT, SAFE_CUBIC_INPUT)
            node.data = np.power(left, 3)
        else:
            print("No matching type and operator!")
    elif node.type == 2:  # two child nodes
        if node.operator == '+':
            node.data = allcal(node.left, indata) + allcal(node.right, indata)
        elif node.operator == '*':
            node.data = allcal(node.left, indata) * allcal(node.right, indata)
        elif node.operator == 'exp_minus_log':
            left = allcal(node.left, indata)
            right = allcal(node.right, indata)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                if np.iscomplexobj(left) or np.iscomplexobj(right):
                    if np.iscomplexobj(left):
                        left = np.clip(left.real, -SAFE_LOG_VALUE, SAFE_LOG_VALUE) + 1j * left.imag
                    else:
                        left = np.clip(left, -SAFE_LOG_VALUE, SAFE_LOG_VALUE)
                    node.data = np.exp(left) - np.log(right)
                else:
                    node.data = np.exp(np.clip(left, -SAFE_LOG_VALUE, SAFE_LOG_VALUE)) - np.log(right)
        else:
            print("No matching type and operator!")
    elif node.type == -1:  # not grown
        print("Not a grown tree!")
    else:
        print("No legal node type!")

    node.data = safe_array(node.data)
    return node.data


# =============================================================================
# # display the structure of the tree, each node displays operator
# # Tree is a list storing the nodes
# =============================================================================
def display(Tree):
    tree_depth = -1
    for i in np.arange(0, len(Tree)):
        if Tree[i].depth > tree_depth:
            tree_depth = Tree[i].depth
    dlists = []
    for d in np.arange(0, tree_depth + 1):
        dlists.append([])
    for i in np.arange(0, len(Tree)):
        dlists[Tree[i].depth].append(Tree[i])

    for d in np.arange(0, len(dlists)):
        st = " "
        for i in np.arange(0, len(dlists[d])):
            if dlists[d][i].type > 0:
                st = st + dlists[d][i].operator + " "
            else:
                st = st + str(dlists[d][i].feature) + " "
        print(st)
    return


# =============================================================================
# # get the height of a (sub)tree with node being root node
# # equivalently, the maximum distance from node to its descendent
# # only a root node has height 0
# # terminal nodes has height 0
# =============================================================================
def getHeight(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        return getHeight(node.left) + 1
    else:
        lheight = getHeight(node.left)
        rheight = getHeight(node.right)
        return max(lheight, rheight) + 1


# =============================================================================
# # get the number of nodes of a (sub)tree with node being root
# =============================================================================
def getNum(node):
    if node.type == 0:
        return 1
    elif node.type == 1:
        return getNum(node.left) + 1
    else:
        lnum = getNum(node.left)
        rnum = getNum(node.right)
        return (lnum + rnum + 1)


# =============================================================================
# # get the number of lt() operators of a (sub)tree with node being root
# =============================================================================
def numLT(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        if node.operator == 'ln':
            return 1 + numLT(node.left)
        else:
            return numLT(node.left)
    else:
        return numLT(node.left) + numLT(node.right)


# =============================================================================
# # update depth of all nodes
# =============================================================================
def upDepth(Root):
    if Root.parent is None:
        Root.depth = 0
    else:
        Root.depth = Root.parent.depth + 1

    if Root.left is not None:
        upDepth(Root.left)
        if Root.right is not None:
            upDepth(Root.right)


# =============================================================================
# # returns a string of the expression of the tree
# # node is the root of the tree
# =============================================================================
def Express(node):
    expr = ""
    if node.type == 0:  # terminal
        if node.constant is not None:
            expr = "1"
        else:
            expr = "x" + str(node.feature)
        return (expr)
    elif node.type == 1:
        if node.operator == 'exp':
            expr = "exp(" + Express(node.left) + ")"
        elif node.operator == 'one':
            expr = "1"
        elif node.operator == 'ln':
            expr = str(round(node.a, 4)) + "*(" + Express(node.left) + ")+" + str(round(node.b, 4))
        elif node.operator == 'inv':  # node.operator == 'inv':
            expr = "1/[" + Express(node.left) + "]"
        elif node.operator == 'sin':
            expr = "sin(" + Express(node.left) + ")"
        elif node.operator == 'cos':
            expr = "cos(" + Express(node.left) + ")"
        elif node.operator == 'square': ## added by fwl
            expr = "(" + Express(node.left) + ")^2"
        elif node.operator == 'cubic': ## added by fwl
            expr = "(" + Express(node.left) + ")^3"
        else:  # note.operate=='neg'
            expr = "-(" + Express(node.left) + ")"

    else:  # node.type==2
        if node.operator == '+':
            expr = Express(node.left) + "+" + Express(node.right)
        elif node.operator == 'exp_minus_log':
            expr = "exp(" + Express(node.left) + ")-log(" + Express(node.right) + ")"
        else:
            expr = "(" + Express(node.left) + ")*(" + Express(node.right) + ")"
    return (expr)


# =============================================================================
# # compute the likelihood of tree structure f(S)
# # P(M,T)*P(theta|M,T)*P(theta|sigma_theta)*P(sigma_theta)*P(theta)
# =============================================================================
def fStruc(node, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal=False, left_prior=None, right_prior=None):
    loglike = 0  # log.likelihood of structure S=(T,M)
    loglike_para = 0  # log.likelihood of linear paras

    '''
    # contribution of hyperparameter sigma_theta
    if node.depth == 0:#root node
        loglike += np.log(invgamma.pdf(node.sigma_a,1))
        loglike += np.log(invgamma.pdf(node.sigma_b,1))
    '''

    # contribution of splitting the node or becoming terminal
    eml_unary_chain_prior = getattr(get_root(node), 'eml_unary_chain_prior', False)
    scale, _ = get_right_left_prior(node, eml_unary_chain_prior)
    prob_split = min(0.9999, scale / np.power((1 + node.depth), -beta))

    if node.type == 0:  # terminal node
        loglike += np.log(1 - prob_split)  # contribution of choosing terminal
        # Determine if parent has conditional prior on this side
        prior_dict = None
        if node.parent is not None:
            parent_op = getattr(node.parent.operator, 'name', node.parent.operator)
            is_left = (node == node.parent.left)
            is_right = (node == node.parent.right)
            
            if eml_unary_chain_prior and parent_op == 'exp_minus_log':
                _, prior_dict_cond = get_right_left_prior(node, eml_unary_chain_prior)
                if prior_dict_cond is not None:
                    prior_dict = prior_dict_cond
            else:
                if is_left and left_prior is not None and parent_op in left_prior:
                    prior_dict = left_prior[parent_op]
                elif is_right and right_prior is not None and parent_op in right_prior:
                    prior_dict = right_prior[parent_op]
                
        if prior_dict is not None:
            # get chosen terminal value
            if node.constant is not None:
                chosen_val = float(node.constant)
            else:
                chosen_val = int(node.feature[0]) if isinstance(node.feature, (list, np.ndarray)) else int(node.feature)
            loglike += get_terminal_logprob(chosen_val, prior_dict, n_feature, const_terminal)
        else:
            loglike -= np.log(terminal_count(n_feature, const_terminal))  # contribution of feature/constant selection
    elif node.type == 1:  # unitary operator
        # contribution of splitting
        if node.depth == 0:  # root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log(prob_split) + np.log(Op_weights[node.op_ind])
        # contribution of parameters of linear nodes
        if node.operator == 'ln':
            loglike_para -= np.power((node.a - 1), 2) / (2 * sigma_a)
            loglike_para -= np.power(node.b, 2) / (2 * sigma_b)
            loglike_para -= 0.5 * np.log(2 * np.pi * sigma_a)
            loglike_para -= 0.5 * np.log(2 * np.pi * sigma_b)
    else:  # binary operator
        # contribution of splitting
        if node.depth == 0:  # root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log(prob_split) + np.log(Op_weights[node.op_ind])

    # contribution of child nodes
    if node.left is None:  # no child nodes
        return [loglike, loglike_para]
    else:
        fleft = fStruc(node.left, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
        loglike += fleft[0]
        loglike_para += fleft[1]
        if node.right is None:  # only one child
            return [loglike, loglike_para]
        else:
            fright = fStruc(node.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
            loglike += fright[0]
            loglike_para += fright[1]

    return [loglike, loglike_para]


# =============================================================================
# # propose a new tree from existing Root
# # and calculate the ratio
# # five possible actions: stay, grow, prune, ReassignOp, ReassignFea.
# =============================================================================
def Prop(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal=False, max_depth=None, max_complexity=None, other_complexity=0, left_prior=None, right_prior=None):
    ###############################
    ######### preparations ########
    ###############################

    # make a copy of Root
    oldRoot = copy.deepcopy(Root)
    # get necessary auxiliary information
    depth = -1
    Tree = genList(Root)
    old_node_count = len(Tree)
    tree_max_complexity = max_complexity - other_complexity if max_complexity is not None else None
    for i in np.arange(0, len(Tree)):
        if Tree[i].depth > depth:
            depth = Tree[i].depth

    # preserve pointers to originally linear nodes
    lnPointers = []
    last_a = []
    last_b = []
    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            lnPointers.append(Tree[i])
            last_a.append(Tree[i].a)
            last_b.append(Tree[i].b)

    # get the list of terminal nodes
    Term = []  # terminal
    Nterm = []  # non-terminal
    cnode = None
    for i in np.arange(0, len(Tree)):
        if Tree[i].type == 0:
            Term.append(Tree[i])
        else:
            Nterm.append(Tree[i])

    # get the list of lt() nodes
    Lins = []
    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            Lins.append(Tree[i])
    ltNum = len(Lins)

    # record expansion and shrinkage
    # expansion occurs when num of lt() increases
    # shrinkage occurs when num of lt() decreases
    change = ''
    Q = Qinv = 1

    # qualified candidates for detransformation
    detcd = []
    # for detransformation: not root or root but child nodes are not all terminal
    # transformation can be applied to any node
    for i in np.arange(len(Tree)):
        flag = True
        if Tree[i].type == 0:  # terminal is not allowed
            flag = False
        elif Tree[i].parent is None:  # root
            if Tree[i].right is None and Tree[i].left is not None and Tree[i].left.type == 0:
                flag = False
            elif Tree[i].left is not None and Tree[i].right is not None and Tree[i].left.type == 0 and Tree[i].right.type == 0:
                flag = False

        if flag == True:
            detcd.append(Tree[i])

    ###############################
    # decide which action to take #
    ###############################

    # probs of each action
    p_stay = 0.25 * ltNum / (ltNum + 3)
    p_grow = (1 - p_stay) * min(1, 4 / (len(Nterm) + 2)) / 3
    p_prune = (1 - p_stay) / 3 - p_grow
    p_detr = (1 - p_stay) * (1 / 3) * len(detcd) / (3 + len(detcd))
    p_trans = (1 - p_stay) / 3 - p_detr
    p_rop = (1 - p_stay) / 6

    # auxiliary
    test = np.random.uniform(0, 1, 1)[0]

    ###############################
    ########### take action #######
    ###############################

    # stay
    if test <= p_stay:
        action = 'stay'
        # print('action:',action)
        # calculate Q and Qinv
        Q = p_stay
        Qinv = p_stay
        # update all linear nodes 
        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                Tree[i].a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
                Tree[i].b = norm.rvs(loc=0,scale=np.sqrt(sigma_b))

    # grow
    elif test <= p_stay + p_grow:
        action = 'grow'
        # print("action:",action)

        # pick a terminal node
        pod = np.random.randint(0, len(Term), 1)[0]
        # grow the node
        # the likelihood is exactly the same as fStruc(), starting from assigning operator
        if tree_max_complexity is not None:
            grow(Term[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth,
                 max_complexity=tree_max_complexity - (old_node_count - 1), node_count=[1], left_prior=left_prior, right_prior=right_prior)
        else:
            grow(Term[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, left_prior=left_prior, right_prior=right_prior)

        if Term[pod].type == 0:  # grow to be terminal
            Q = Qinv = 1
        else:
            # calculate Q
            fstrc = fStruc(Term[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
            Q = p_grow * np.exp(fstrc[0]) / len(Term)
            # calculate Qinv (equiv to prune)
            new_ltNum = numLT(Root)
            new_height = getHeight(Root)
            new_nodeNum = getNum(Root)
            newTerm = []  # terminal
            newTree = genList(Root)
            new_nterm = []  # non-terminal
            for i in np.arange(0, len(newTree)):
                if newTree[i].type == 0:
                    newTerm.append(newTree[i])
                else:
                    new_nterm.append(newTree[i])
            new_termNum = len(newTerm)
            new_p = (1 - 0.25 * new_ltNum / (new_ltNum + 3)) * (1 - min(1, 4 / (len(new_nterm) + 2))) / 3
            Qinv = new_p / max(1, (new_nodeNum - new_termNum - 1))  # except root node

            if new_ltNum > ltNum:
                change = 'expansion'

    # prune
    elif test <= p_stay + p_grow + p_prune:
        action = 'prune'
        # print("action:",action)

        # pick a node to prune
        pod = np.random.randint(1, len(Nterm), 1)[0]  # except root node
        fstrc = fStruc(Nterm[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
        pruned = copy.deepcopy(Nterm[pod])  # preserve a copy

        # preserve pointers to all cutted ln
        p_ltNum = numLT(pruned)
        if p_ltNum > 0:
            change = 'shrinkage'

        # prune the node
        Nterm[pod].left = None
        Nterm[pod].right = None
        Nterm[pod].operator = None
        Nterm[pod].type = 0
        assign_terminal(Nterm[pod], n_feature, const_terminal, left_prior, right_prior)
        # print("prune and assign feature:",Par[pod].feature)

        # quantities for new tree
        new_ltNum = numLT(Root)
        new_height = getHeight(Root)
        new_nodeNum = getNum(Root)
        newTerm = []  # terminal
        new_nTerm = []  # non-terminal
        newTree = genList(Root)
        for i in np.arange(0, len(newTree)):
            if newTree[i].type == 0:
                newTerm.append(newTree[i])
            else:
                new_nTerm.append(newTree[i])

        # calculate Q
        Q = p_prune / ((len(Nterm) - 1) * terminal_count(n_feature, const_terminal))

        # calculate Qinv (correspond to grow)
        pg = 1 - 0.25 * new_ltNum / (new_ltNum + 3) * 0.75 * min(1, 4 / (len(new_nTerm) + 2))
        Qinv = pg * np.exp(fstrc[0]) / len(newTerm)

    # detransformation
    elif test <= p_stay + p_grow + p_prune + p_detr:
        action = 'detransform'

        det_od = np.random.randint(0, len(detcd), 1)[0]
        det_node = detcd[det_od]
        cutt = None

        # print("cutted op:",det_node.operator)

        Q = p_detr / len(detcd)

        if det_node.parent is None:  # root
            if det_node.right is None:  # one child
                Root = Root.left
            else:  # two children
                if det_node.left.type == 0:  # left is terminal
                    cutt = Root.left
                    Root = Root.right
                elif det_node.right.type == 0:  # right is terminal
                    cutt = Root.right
                    Root = Root.left
                else:  # both are non-terminal
                    aa = np.random.uniform(0, 1, 1)[0]
                    if aa <= 0.5:  # preserve left
                        cutt = Root.right
                        Root = Root.left
                    else:
                        cutt = Root.left
                        Root = Root.right
                    Q = Q / 2
            Root.parent = None
            upDepth(Root)
        else:  # not root, non-terminal
            if det_node.type == 1:  # unary
                if det_node.parent.left is det_node:  # left child of its parent
                    det_node.parent.left = det_node.left
                    det_node.left.parent = det_node.parent
                else:
                    det_node.parent.right = det_node.left
                    det_node.left.parent = det_node.parent
            else:  # binary
                aa = np.random.uniform(0, 1, 1)[0]
                if aa <= 0.5:  # preserve left
                    cutt = det_node.right
                    if det_node.parent.left is det_node:
                        det_node.parent.left = det_node.left
                        det_node.left.parent = det_node.parent
                    else:
                        det_node.parent.right = det_node.left
                        det_node.left.parent = det_node.parent
                else:  # preserve right
                    cutt = det_node.left
                    if det_node.parent.left is det_node:
                        det_node.parent.left = det_node.right
                        det_node.right.parent = det_node.parent
                    else:
                        det_node.parent.right = det_node.right
                        det_node.right.parent = det_node.parent
                Q = Q / 2
            Root.parent = None
            upDepth(Root)

        # the number of linear nodes may decrease
        new_ltnum = 0
        new_tree = genList(Root)
        for i in np.arange(len(new_tree)):
            if new_tree[i].operator == 'ln':
                new_ltnum += 1
        if new_ltnum < ltNum:
            change = 'shrinkage'

        new_pstay = 0.25 * new_ltnum / (new_ltnum + 3)

        # calculate Qinv (correspond to transform)
        new_detcd = []
        for i in np.arange(len(new_tree)):
            flag = True
            if new_tree[i].type == 0:  # terminal is not allowed
                flag = False
            elif new_tree[i].parent is None:  # root
                if new_tree[i].right is None and new_tree[i].left is not None and new_tree[i].left.type == 0:
                    flag = False
                elif new_tree[i].left is not None and new_tree[i].right is not None and new_tree[i].left.type == 0 and new_tree[i].right.type == 0:
                    flag = False
            if flag == True:
                new_detcd.append(new_tree[i])
        new_pdetr = (1 - new_pstay) * (1 / 3) * len(new_detcd) / (len(new_detcd) + 3)
        new_ptr = (1 - new_pstay) / 3 - new_pdetr
        Qinv = new_ptr * Op_weights[det_node.op_ind] / len(new_tree)
        if cutt is not None:
            fstrc = fStruc(cutt, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
            Qinv = Qinv * np.exp(fstrc[0])




    # transformation
    elif test <= p_stay + p_grow + p_prune + p_detr + p_trans:
        action = "transform"

        # transform means uniformly pick a node and add an operator as its parent
        # probability of operators is specified by Op_weights
        # adding unary operator is simple
        # adding binary operator needs to generate a new sub-tree
        Tree = genList(Root)
        ins_ind = np.random.randint(0, len(Tree), 1)[0]
        ins_node = Tree[ins_ind]
        ins_opind = np.random.choice(np.arange(0, len(Ops)), p=Op_weights)
        ins_op = Ops[ins_opind]
        ins_type = Op_type[ins_opind]
        ins_opweight = Op_weights[ins_opind]

        # create new node
        new_node = Node(ins_node.depth)
        new_node.operator = ins_op
        new_node.type = ins_type
        new_node.op_ind = ins_opind

        if ins_type == 1:  # unary
            if ins_op == 'ln':  # linear node
                change = 'expansion'
                # new_node.a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
                # new_node.b = norm.rvs(loc=0,scale=np.sqrt(sigma_b))
            if ins_node.parent is None:  # inserted node is root
                Root = new_node
                new_node.left = ins_node
                ins_node.parent = new_node
            else:  # inserted node is not root
                if ins_node.parent.left is ins_node:
                    ins_node.parent.left = new_node
                else:
                    ins_node.parent.right = new_node
                new_node.parent = ins_node.parent
                new_node.left = ins_node
                ins_node.parent = new_node
            upDepth(Root)
            # calculate Q
            Q = p_trans * ins_opweight / len(Tree)

        else:  # binary
            if ins_node.parent is None:  # inserted node is root
                Root = new_node
                new_node.left = ins_node
                ins_node.parent = new_node
                new_right = Node(1)
                new_node.right = new_right
                new_right.parent = new_node
                upDepth(Root)
                if tree_max_complexity is not None:
                    grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth,
                         max_complexity=tree_max_complexity - old_node_count - 1, node_count=[1], left_prior=left_prior, right_prior=right_prior)
                else:
                    grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, left_prior=left_prior, right_prior=right_prior)
                fstrc = fStruc(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
                # calculate Q
                Q = p_trans * ins_opweight * np.exp(fstrc[0]) / len(Tree)
            else:  # inserted node is not root
                # place the new node
                if ins_node.parent.left is ins_node:
                    ins_node.parent.left = new_node
                    new_node.parent = ins_node.parent
                else:
                    ins_node.parent.right = new_node
                    new_node.parent = ins_node.parent

                new_node.left = ins_node
                ins_node.parent = new_node
                new_right = Node(new_node.depth + 1)
                new_node.right = new_right
                new_right.parent = new_node
                upDepth(Root)
                if tree_max_complexity is not None:
                    grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth,
                         max_complexity=tree_max_complexity - old_node_count - 1, node_count=[1], left_prior=left_prior, right_prior=right_prior)
                else:
                    grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, left_prior=left_prior, right_prior=right_prior)
                fstrc = fStruc(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)

                # calculate Q
                Q = p_trans * ins_opweight * np.exp(fstrc[0]) / len(Tree)

        # the number of linear nodes may decrease
        new_ltnum = 0
        new_tree = genList(Root)
        for i in np.arange(len(new_tree)):
            if new_tree[i].operator == 'ln':
                new_ltnum += 1
        if new_ltnum > ltNum:
            change = 'expansion'

        # calculate Qinv (correspond to detransform)
        new_pstay = 0.25 * new_ltnum / (new_ltnum + 3)

        new_detcd = []
        for i in np.arange(len(new_tree)):
            flag = True
            if new_tree[i].type == 0:  # terminal is not allowed
                flag = False
            elif new_tree[i].parent is None:  # root
                if new_tree[i].right is None and new_tree[i].left is not None and new_tree[i].left.type == 0:
                    flag = False
                elif new_tree[i].left is not None and new_tree[i].right is not None and new_tree[i].left.type == 0 and new_tree[i].right.type == 0:
                    flag = False
            if flag == True:
                new_detcd.append(new_tree[i])

        new_pdetr = (1 - new_pstay) * (1 / 3) * len(new_detcd) / (len(new_detcd) + 3)
        new_ptr = (1 - new_pstay) / 3 - new_pdetr

        Qinv = new_pdetr / len(new_detcd) if len(new_detcd) > 0 else 0.0
        if new_node.type == 2:
            if new_node.left.type > 0 and new_node.right.type > 0:
                Qinv = Qinv / 2



    # reassignOperator
    elif test <= p_stay + p_grow + p_prune + p_detr + p_trans + p_rop:
        if len(Nterm) == 0:
            action = 'stay'
            Q = 1.0
            Qinv = 1.0
            cnode = None
        else:
            action = 'ReassignOperator'
            # print("action:",action)
            pod = np.random.randint(0, len(Nterm), 1)[0]
            last_op = Nterm[pod].operator
            last_op_ind = Nterm[pod].op_ind
            last_type = Nterm[pod].type
            # print('replaced operator:',last_op)
            cnode = Nterm[pod]  ########pointer to the node changed#######
            # a deep copy of the changed node and its descendents
            replaced = copy.deepcopy(Nterm[pod])

            new_od = np.random.choice(np.arange(0, len(Ops)), p=Op_weights)
            new_op = Ops[new_od]
            # print('assign new operator:',new_op)
            new_type = Op_type[new_od]

            # originally unary
            if last_type == 1:
                if new_type == 1:  # unary to unary
                    # assign operator and type
                    Nterm[pod].operator = new_op
                    if last_op == 'ln':  # originally linear
                        if new_op != 'ln':  # change from linear to other ops
                            cnode.a = None
                            cnode.b = None
                            change = 'shrinkage'
                    else:  # orignally not linear
                        if new_op == 'ln':  # linear increases by 1
                            ###### a and b is not sampled
                            change = 'expansion'

                    # calculate Q, Qinv (equal)
                    Q = Op_weights[new_od]
                    Qinv = Op_weights[last_op_ind]

                else:  # unary to binary
                    # assign operator and type
                    cnode.operator = new_op
                    cnode.type = 2
                    if last_op == 'ln':
                        cnode.a = None
                        cnode.b = None
                        # grow a new sub-tree rooted at right child
                    cnode.right = Node(cnode.depth + 1)
                    cnode.right.parent = cnode
                    if tree_max_complexity is not None:
                        grow(cnode.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth,
                             max_complexity=tree_max_complexity - old_node_count, node_count=[1], left_prior=left_prior, right_prior=right_prior)
                    else:
                        grow(cnode.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, left_prior=left_prior, right_prior=right_prior)
                    fstrc = fStruc(cnode.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)

                    # calculate Q
                    Q = p_rop * np.exp(fstrc[0]) * Op_weights[new_od] / (len(Nterm))
                    # calculate Qinv
                    # get necessary quantities
                    new_height = getHeight(Root)
                    new_nodeNum = getNum(Root)
                    newTerm = []  # terminal
                    newTree = genList(Root)
                    new_ltNum = numLT(Root)
                    for i in np.arange(0, len(newTree)):
                        if newTree[i].type == 0:
                            newTerm.append(newTree[i])
                        # reversed action is binary to unary
                    new_p0 = new_ltNum / (4 * (new_ltNum + 3))
                    Qinv = 0.125 * (1 - new_p0) * Op_weights[last_op_ind] / (new_nodeNum - len(newTerm))

                    # record change of dim
                    if new_ltNum > ltNum:
                        change = 'expansion'
                    elif new_ltNum < ltNum:
                        change = 'shrinkage'




            # originally binary
            else:
                if new_type == 1:  # binary to unary
                    # assign operator and type
                    cutted = copy.deepcopy(cnode.right)  # deep copy root of the cutted subtree
                    # preserve pointers to all cutted ln
                    p_ltNum = numLT(cutted)
                    if p_ltNum > 1:
                        change = 'shrinkage'
                    elif new_op == 'ln':
                        if p_ltNum == 0:
                            change = 'expansion'

                    cnode.right = None
                    cnode.operator = new_op
                    cnode.type = new_type

                    # calculate Q
                    Q = p_rop * Op_weights[new_od] / len(Nterm)
                    # calculate Qinv
                    # necessary quantities
                    new_nodeNum = getNum(Root)
                    newTerm = []  # terminal
                    newTree = genList(Root)
                    new_ltNum = numLT(Root)
                    # reversed action is unary to binary and grow
                    new_p0 = new_ltNum / (4 * (new_ltNum + 3))
                    fstrc = fStruc(cutted, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
                    Qinv = 0.125 * (1 - new_p0) * np.exp(fstrc[0]) * Op_weights[last_op_ind] / (
                    (new_nodeNum - len(newTerm)))



                else:  # binary to binary
                    # assign operator
                    cnode.operator = new_op
                    # calculate Q,Qinv(equal)
                    Q = Op_weights[new_od]
                    Qinv = Op_weights[last_op_ind]


    # reassign feature
    else:
        action = 'ReassignFeature'
        # print("action:",action)

        # pick a terminal node
        pod = np.random.randint(0, len(Term), 1)[0]
        # pick a feature and reassign
        assign_terminal(Term[pod], n_feature, const_terminal, left_prior, right_prior)
        # calculate Q,Qinv (equal)
        Q = Qinv = 1

    Root.parent = None
    upDepth(Root)
    # print(action)

    return [oldRoot, Root, lnPointers, change, Q, Qinv, last_a, last_b, cnode]


# =============================================================================
# # calculate the likelihood of genenerating auxiliary variable
# # change is a string with value of 'expansion' or 'shrinkage'
# # oldRoot is the root of the original Tree
# # Root is the root of the new tree
# # cnode is a pointer to the node just changed (expand or shrink 'ln') (if applicable)
# # last_a and last_b is the parameters for the shrinked node (if applicable)
# # lnPointers is list of pointers to originally linear nodes in Tree before changing
# =============================================================================
def auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a, last_b, cnode=None):
    # record the informations of linear nodes other than the shrinked or expanded
    odList = []  # list of orders of linear nodes
    Tree = genList(Root)

    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            odList.append(i)

    # sample new sigma_a2 and sigma_b2
    new_sa2 = invgamma.rvs(1)
    new_sb2 = invgamma.rvs(1)
    old_sa2 = sigma_a
    old_sb2 = sigma_b

    if change == 'shrinkage':
        prsv_aList = []
        prsv_bList = []
        cut_aList = []
        cut_bList = []
        # find the preserved a's
        for i in np.arange(0, len(lnPointers)):
            if lnPointers[i].operator == 'ln':  # still linear
                prsv_aList.append(last_a[i])
                prsv_bList.append(last_b[i])
            else:  # no longer linear
                cut_aList.append(last_a[i])
                cut_bList.append(last_b[i])
        # substitute those cutted with newly added if cut and add
        for i in np.arange(0, len(odList) - len(prsv_aList)):
            prsv_aList.append(cut_aList[i])
            prsv_bList.append(cut_bList[i])

        n0 = len(prsv_aList)

        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0, n0):
            UaList.append(norm.rvs(loc=0, scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # generate inverse auxiliary U*
        NaList = []  # Theta* with a
        NbList = []  # Theta* with b
        NUaList = []  # U* with a
        NUbList = []  # U* with b
        for i in np.arange(0, n0):
            NaList.append(prsv_aList[i] + UaList[i])
            NbList.append(prsv_bList[i] + UbList[i])
            NUaList.append(prsv_aList[i] - UaList[i])
            NUbList.append(prsv_bList[i] - UbList[i])
        NUaList = NUaList + last_a
        NUbList = NUbList + last_b

        # hstar is h(U*|Theta*,S*,S^t) (here we calculate the log) corresponding the shorter para
        # h is h(U|Theta,S^t,S*) corresponding longer para
        # Theta* is the
        logh = 0
        loghstar = 0

        # contribution of new_sa2 and new_sb2
        logh += invgamma.logpdf(new_sa2, 1)
        logh += invgamma.logpdf(new_sb2, 1)
        loghstar += invgamma.logpdf(old_sa2, 1)
        loghstar += invgamma.logpdf(old_sb2, 1)

        for i in np.arange(0, len(UaList)):
            # contribution of UaList and UbList
            logh += norm.logpdf(UaList[i], loc=0, scale=np.sqrt(new_sa2))
            logh += norm.logpdf(UbList[i], loc=0, scale=np.sqrt(new_sb2))

        for i in np.arange(0, len(NUaList)):
            # contribution of NUaList and NUbList
            loghstar += norm.logpdf(NUaList[i], loc=0, scale=np.sqrt(old_sa2))
            loghstar += norm.logpdf(NUbList[i], loc=0, scale=np.sqrt(old_sb2))

        hratio = np.exp(loghstar - logh)
        # print("hratio:",hratio)

        # determinant of jacobian
        detjacob = np.power(2, 2 * len(prsv_aList))
        # print("detjacob:",detjacob)

        #### assign Theta* to the variables
        # new values of Theta
        # new sigma_a, sigma_b are directly returned
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [hratio, detjacob, new_sa2, new_sb2]



    elif change == 'expansion':
        # sample new sigma_a2 and sigma_b2
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        old_sa2 = sigma_a
        old_sb2 = sigma_b

        # lists of theta_0 and expanded ones
        # last_a is the list of all original a's
        # last_b is the list of all original b's
        odList = []
        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)

        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0, len(last_a)):
            UaList.append(norm.rvs(loc=0, scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # generate inverse auxiliary U* and new para Theta*
        NaList = []  # Theta*_a
        NbList = []  # Theta*_b
        NUaList = []  # U*_a
        NUbList = []  # U*_b
        for i in np.arange(0, len(last_a)):
            NaList.append((last_a[i] + UaList[i]) / 2)
            NbList.append((last_b[i] + UbList[i]) / 2)
            NUaList.append((last_a[i] - UaList[i]) / 2)
            NUbList.append((last_b[i] - UbList[i]) / 2)

        # append newly generated a and b into NaList and NbList
        nn = len(odList) - len(last_a)  # number of newly added ln
        for i in np.arange(0, nn):
            u_a = norm.rvs(loc=1, scale=np.sqrt(new_sa2))
            u_b = norm.rvs(loc=0, scale=np.sqrt(new_sb2))
            NaList.append(u_a)
            NbList.append(u_b)

        # calculate h ratio
        # logh is h(U|Theta,S,S*) correspond to jump from short to long (new)
        # loghstar is h(Ustar|Theta*,S*,S) correspond to jump from long to short
        logh = 0
        loghstar = 0

        # contribution of sigma_ab
        logh += invgamma.logpdf(new_sa2, 1)
        logh += invgamma.logpdf(new_sb2, 1)
        loghstar += invgamma.logpdf(old_sa2, 1)
        loghstar += invgamma.logpdf(old_sb2, 1)

        # contribution of u_a, u_b
        for i in np.arange(len(last_a), nn):
            logh += norm.logpdf(NaList[i], loc=1, scale=np.sqrt(new_sa2))
            logh += norm.logpdf(NbList[i], loc=0, scale=np.sqrt(new_sb2))

        # contribution of U_theta
        for i in np.arange(0, len(UaList)):
            logh += norm.logpdf(UaList[i], loc=0, scale=np.sqrt(new_sa2))
            logh += norm.logpdf(UbList[i], loc=0, scale=np.sqrt(new_sb2))

        for i in np.arange(0, len(NUaList)):
            loghstar += norm.logpdf(NUaList[i], loc=0, scale=np.sqrt(old_sa2))
            loghstar += norm.logpdf(NUbList[i], loc=0, scale=np.sqrt(old_sb2))

        # compute h ratio
        hratio = np.exp(loghstar - logh)

        # determinant of jacobian
        detjacob = 1 / np.power(2, 2 * len(last_a))

        #### assign Theta* to the variables
        # new values of sigma_a sigma_b
        # new values of Theta
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [hratio, detjacob, new_sa2, new_sb2]


    else:  # same set of parameters
        # record the informations of linear nodes other than the shrinked or expanded
        odList = []  # list of orders of linear nodes
        Tree = genList(Root)

        old_sa2 = sigma_a
        old_sb2 = sigma_b

        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)

        NaList = []
        NbList = []
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        for i in np.arange(0, len(odList)):
            NaList.append(norm.rvs(loc=1, scale=np.sqrt(new_sa2)))
            NbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # new values of Theta
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [new_sa2, new_sb2]


# =============================================================================
# # calculate the log likelihood f(y|S,Theta,x)
# # (S,Theta) is represented by node Root
# # prior is y ~ N(output,sigma)
# # output is the matrix of outputs corresponding to different roots
# =============================================================================
def ylogLike(y, outputs, sigma):
    XX = copy.deepcopy(outputs)
    if np.iscomplexobj(XX):
        XX = np.real(XX)
    constant = np.ones((XX.shape[0], 1))
    XX = np.concatenate((constant, XX), axis=1)
    scale = np.max(np.abs(XX), axis=0)
    scale[scale == 0] = 1.0
    XX = XX / scale
    epsilon = np.eye(XX.shape[1])*1e-6
    yy = np.array(y)
    yy.shape = (yy.shape[0], 1)
    Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
    Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))

    output = np.matmul(XX, Beta)

    # error = 0
    # for i in np.arange(0, len(y)):
    #     error += (y[i] - output[i, 0]) * (y[i] - output[i, 0])
    error = np.sum(np.square(y-output[:,0]))
    #error = np.sqrt(error / len(y))
    # print("mean error:",error)

    # log_sum = 0
    # for i in np.arange(0, len(y)):
    #     temp = np.power(y[i] - output[i, 0], 2)  # np.log(norm.pdf(y[i],loc=output[i,0],scale=np.sqrt(sigma)))
    #     # print(i,temp)
    #     log_sum += temp
    log_sum = error
    log_sum = -log_sum / (2 * sigma * sigma)
    log_sum -= 0.5 * len(y) * np.log(2 * np.pi * sigma * sigma)
    return (log_sum)


# =============================================================================
# # prop new structure, sample new parameters and decide whether to accept
# # Root is the root node of the tree to be changed
# # RootLists stores the list of roots of K trees
# # sigma is for output to y
# # sigma_a, sigma_b are (squared) hyper-paras for linear paras
# =============================================================================
def newProp(
    Roots,
    count,
    sigma,
    y,
    indata,
    n_feature,
    Ops,
    Op_weights,
    Op_type,
    beta,
    sigma_a,
    sigma_b,
    const_terminal=False,
    max_complexity=None,
    max_depth=None,
    temperature=1.0,
    left_prior=None,
    right_prior=None,
):
    # number of components
    K = len(Roots)
    # the root to edit
    Root = copy.deepcopy(Roots[count])
    
    other_complexity = 0
    for i in np.arange(K):
        if i != count:
            other_complexity += getNum(Roots[i])
            
    [oldRoot, Root, lnPointers, change, Q, Qinv, last_a, last_b, cnode] = Prop(
        Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth,
        max_complexity=max_complexity, other_complexity=other_complexity, left_prior=left_prior, right_prior=right_prior
    )
    if max_complexity is not None:
        proposed_complexity = 0
        for i in np.arange(len(Roots)):
            proposed_complexity += getNum(Root if i == count else Roots[i])
        if proposed_complexity > max_complexity:
            return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]
    if max_depth is not None:
        proposed_depth = getHeight(Root)
        if proposed_depth > max_depth:
            return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]
    # print("change:",change)
    # display(genList(Root))
    # allcal(Root,train_data)
    sig = 4
    new_sigma = invgamma.rvs(sig)

    # matrix of outputs
    new_outputs = np.zeros((len(y), K), dtype=indata.values.dtype)
    old_outputs = np.zeros((len(y), K), dtype=indata.values.dtype)

    # auxiliary propose
    if change == 'shrinkage':
        [hratio, detjacob, new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a,
                                                       last_b, cnode)
    elif change == 'expansion':
        [hratio, detjacob, new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a,
                                                       last_b, cnode)
    else:  # no dimension jump
        # the parameters are upgraded as well
        [new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a, last_b, cnode)

    for i in np.arange(K):
        if i == count:
            temp = allcal(Root, indata)
            temp.shape = (temp.shape[0])
            new_outputs[:, i] = temp
            temp = allcal(oldRoot, indata)
            temp.shape = (temp.shape[0])
            old_outputs[:, i] = temp
        else:
            temp = allcal(Roots[i], indata)
            temp.shape = (temp.shape[0])
            new_outputs[:, i] = temp
            old_outputs[:, i] = temp

    if np.linalg.matrix_rank(new_outputs) < K:
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]

    if change == 'shrinkage':
        # contribution of f(y|S,Theta,x)
        # print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y, new_outputs, new_sigma)
        # print("sigma:",round(sigma,3))
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = (yllstar - yll) / temperature
        # print("log yratio:",log_yratio)

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2, const_terminal, left_prior, right_prior)
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
        sl = strucl[0] + strucl[1]
        slstar = struclstar[0] + struclstar[1]
        log_strucratio = slstar - sl  # struclstar / strucl
        # print("log strucratio:",log_strucratio)

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))
        # print("log qratio:",log_qratio)

        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(max(1e-5,hratio)) + np.log(max(1e-5,detjacob))
        logR = logR + invgamma.logpdf(new_sigma, sig) - invgamma.logpdf(sigma, sig)
        # print("logR:",logR)

    elif change == 'expansion':
        # contribution of f(y|S,Theta,x)
        yllstar = ylogLike(y, new_outputs, new_sigma)
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = (yllstar - yll) / temperature

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2, const_terminal, left_prior, right_prior)
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)
        sl = strucl[0] + strucl[1]
        slstar = struclstar[0] + struclstar[1]
        log_strucratio = slstar - sl  # struclstar / strucl

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))

        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(max(1e-5,hratio)) + np.log(max(1e-5,detjacob))
        logR = logR + invgamma.logpdf(new_sigma, sig) - invgamma.logpdf(sigma, sig)

    else:  # no dimension jump
        # contribution of f(y|S,Theta,x)
        yllstar = ylogLike(y, new_outputs, new_sigma)
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = (yllstar - yll) / temperature
        # yratio = np.exp(yllstar-yll)

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2, const_terminal, left_prior, right_prior)[0]
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, left_prior, right_prior)[0]
        log_strucratio = struclstar - strucl

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))

        # R
        logR = log_yratio + log_strucratio + log_qratio
        logR = logR + invgamma.logpdf(new_sigma, sig) - invgamma.logpdf(sigma, sig)

    alpha = min(logR, 0)
    test = np.random.uniform(low=0, high=1, size=1)[0]
    if np.log(test) >= alpha:  # no accept
        # print("no accept")
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]
    else:
        # print("||||||accepted||||||")
        return [True, new_sigma, copy.deepcopy(Root), new_sa2, new_sb2]
