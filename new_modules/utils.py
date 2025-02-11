import numpy as np
import numba as nb

import networkx as nx
import matplotlib.pyplot as plt

import torch

def softmax(x, axis = 0):
    """
    Computes the softmax of an array x along a given axis.

    Parameters:
    --- x: np.array
        Array to be softmaxed.
    --- axis: int or tuple of ints
        Axis or axes along which the softmax is computed.

    Returns:
    --- np.array
        Softmaxed array, of the same shape as x.
    """
    max_x = np.max(x, axis = axis, keepdims = True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis = axis, keepdims = True)
    return exp_x / sum_exp_x

def torch_softmax_2dim(x, dims):
    """
    Computes the softmax of a tensor x along a given axis.

    Parameters:
    --- x: torch.Tensor
        Tensor to be softmaxed.
    --- dims: tuple
        Dimensions along which the softmax is computed.

    Returns:
    --- torch.Tensor
        Softmaxed tensor, of the same shape as x.
    """
    max_x = torch.max(x, dim = dims[0], keepdim = True)[0]
    max_x = torch.max(max_x, dim = dims[1], keepdim = True)[0]
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim = dims, keepdim = True)
    return exp_x / sum_exp_x

@nb.njit
def numba_random_choice(vals, probs):
    """
    Chooses a value from vals with probabilities given by probs.

    Parameters:
    --- vals: np.array
        Array of values to choose from.
    --- probs: np.array
        Array of probabilities for each value in vals.

    Returns:
    --- object
        Value chosen from vals.
    """
    r = np.random.rand()
    cum_probs = np.cumsum(probs)
    for idx in range(len(probs)):
        if r < cum_probs[idx]:
            return vals[idx]

def combine_spaces(space1, space2):
    """
    Combines two spaces into a single space. Useful to index the combined
    space with a single index.

    Parameters:
    --- space1: np.array
        First space to be combined.
    --- space2: np.array
        Second space to be combined.

    Returns:
    --- np.array
        Combined space, with shape (space1.size * space2.size, 2).
    """
    return np.array(np.meshgrid(space1, space2)).T.reshape(-1, 2)

def plot_FSC_network(ax, piprob, pitilde_prob,
                     memory1_color='gray', memory2_color='gray',
                     action_r_color='lightblue', action_t_color='salmon',
                     title=""):
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add memory nodes
    G.add_node('$M_1$', shape='o', color=memory1_color)
    G.add_node('$M_2$', shape='o', color=memory2_color)
    
    # Add action nodes
    G.add_node('R1', shape='s', color=action_r_color)
    G.add_node('T1', shape='s', color=action_t_color)
    G.add_node('R2', shape='s', color=action_r_color)
    G.add_node('T2', shape='s', color=action_t_color)
    
    # Add edges from memory nodes to action nodes
    G.add_edges_from([('$M_1$', 'R1'), ('$M_1$', 'T1'),
                      ('$M_2$', 'R2'), ('$M_2$', 'T2')])
    
    # Add edges from action nodes to memory nodes
    G.add_edges_from([('R1', '$M_1$'), ('R1', '$M_2$'),
                      ('T1', '$M_1$'), ('T1', '$M_2$'),
                      ('R2', '$M_1$'), ('R2', '$M_2$'),
                      ('T2', '$M_1$'), ('T2', '$M_2$')])
    
    # Define node shapes
    node_shapes = {'o': 'o', 's': 's'}
    
    # Define custom labels for plotting
    labels = {'$M_1$': '$M_1$', '$M_2$': '$M_2$', 'R1': 'R', 'T1': 'T', 'R2': 'R', 'T2': 'T'}
    
    # Define positions for the nodes
    pos = {
        '$M_1$': (0, 0),
        'R1': (0.2, 1),
        'T1': (0.2, -1),
        '$M_2$': (2, 0),
        'R2': (1.8, 1),
        'T2': (1.8, -1)
    }

    ax.axis('off')
    
    # Draw the graph
    for shape in node_shapes:
        nodes = [n for n in G.nodes if G.nodes[n]['shape'] == shape]
        colors = [G.nodes[n]['color'] for n in nodes]
        node_size = 2000 if shape == 'o' else 400  # Larger size for memory nodes, smaller for action nodes
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=node_shapes[shape],
                               node_size=node_size, node_color=colors, ax=ax)
    
    # Draw edges with arrows and curved connections
    memory_to_action_edges_left = [('$M_2$', 'R2', piprob[1, 0]), ('$M_1$', 'T1', piprob[0, 1])]
    memory_to_action_edges_right = [('$M_1$', 'R1', piprob[0, 0]), ('$M_2$', 'T2', piprob[1, 1])]
    memory_to_action_edges = [memory_to_action_edges_left, memory_to_action_edges_right]

    action_to_memory_stay_edges_1 = [('R1', '$M_1$', pitilde_prob[0, 0, 0]), ('T2', '$M_2$', pitilde_prob[1, 1, 1])]
    action_to_memory_stay_edges_2 = [('R2', '$M_2$', pitilde_prob[1, 1, 0]), ('T1', '$M_1$', pitilde_prob[0, 0, 1])]

    action_to_memory_stay_edges = [action_to_memory_stay_edges_1, action_to_memory_stay_edges_2]

    action_to_memory_edges = [('R1', '$M_2$', pitilde_prob[0, 1, 0]), ('T1', '$M_2$', pitilde_prob[0, 1, 1]),
                              ('R2', '$M_1$', pitilde_prob[1, 0, 0]), ('T2', '$M_1$', pitilde_prob[1, 0, 1])]
    
    # Draw edges with different colors and connection styles

    xshifts = {'$M_1$': -0.33, '$M_2$': 0.18}
    pscale = 5
    rads = [0.5, -0.5]

    for i in range(len(memory_to_action_edges)):
        for j, (u, v, prob) in enumerate(memory_to_action_edges[i]):
            width = prob * pscale

            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rads[i]}', width=width,
                                arrows=True, arrowsize=20, min_source_margin=25, min_target_margin=15, edge_color='gray', ax=ax)
            
            x = (pos[u][0] + pos[v][0]) / 2 + xshifts[u]
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, f'{prob:.3f}', fontsize=10, color='gray')

    rads = [-0.2, 0.2]
    xshifts = {'R1': 0.1, 'T1': 0.08, 'R2': -0.27, 'T2': -0.25}
    yshifts = {'R1': 0.08, 'T1': -0., 'R2': 0., 'T2': -0.15}
    colors = {'R1': 'lightblue', 'T1': 'salmon', 'R2': 'lightblue', 'T2': 'salmon'}

    for i in range(len(action_to_memory_stay_edges)):
        for j, (u, v, prob) in enumerate(action_to_memory_stay_edges[i]):
            width = prob * pscale

            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rads[i]}', width=width,
                                   arrows=True, arrowsize=20, min_source_margin=15, min_target_margin=25, edge_color=colors[u], ax=ax)
            
            x = (pos[u][0] + pos[v][0]) / 2 + xshifts[u]
            y = (pos[u][1] + pos[v][1]) / 2 + yshifts[u]
            ax.text(x, y, f'{prob:.3f}', fontsize=10, color=colors[u])

    xshifts = {'R1': 0., 'T1': -0.5, 'R2': -0.27, 'T2': -0.3}
    yshifts = {'R1': -0.35, 'T1': -0.45, 'R2': 0.45, 'T2': 0.58}

    for i, (u, v, prob) in enumerate(action_to_memory_edges):
        width = prob * pscale

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle='arc3,rad=0.2', width=width,
                               arrows=True, arrowsize=20, min_source_margin=15, min_target_margin=25, edge_color=colors[u], ax=ax)
        
        x = (pos[u][0] + pos[v][0]) / 2 + xshifts[u]
        y = (pos[u][1] + pos[v][1]) / 2 + yshifts[u]

        ax.text(x, y, f'{prob:.3f}', fontsize=10, color=colors[u])
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white', ax=ax)
    
    # Add title
    ax.set_title(title)

    cut = 1.1
    xmax= cut*max(xx for xx,yy in pos.values())
    ax.set_xlim(-0.2,xmax)


def extract_durations(trajectory):
    durations = []

    current_value = trajectory[0]
    current_duration = 0

    for value in trajectory:
        if value == current_value:
            current_duration += 1
        else:
            durations.append((current_value, current_duration))
            current_value = value
            current_duration = 1

    durations.append((current_value, current_duration))
    return durations

def filter_durations(durations, target_value):
    return np.array([duration for value, duration in durations if value == target_value])

def segment_trajectories(trajectory, label = "actions"):
    possible_actions = np.unique(trajectory[label])
    idxs = {action: np.where(trajectory[label] == action)[0] for action in possible_actions}

    # Segment the trajectory by action, so that each action corresponds to a list of arrays of indexes corresponding to that action
    segmented_trajectory = {action: [] for action in possible_actions}

    for action in possible_actions:
        idxs_action = idxs[action]
        current_segment = [idxs_action[0]]
        for i in range(1, len(idxs_action)):
            if idxs_action[i] == idxs_action[i - 1] + 1:
                current_segment.append(idxs_action[i])
            else:
                segmented_trajectory[action].append(np.array(current_segment))
                current_segment = [idxs_action[i]]
        segmented_trajectory[action].append(np.array(current_segment))

    return segmented_trajectory

def get_cumulative(data):
    values, counts = np.unique(data, return_counts=True)
    cumulative = np.cumsum(counts)
    cumulative = cumulative / cumulative[-1]  # Normalize the cumulative values
    return values, cumulative

def get_inverse_cumulative(data):
    values, counts = np.unique(data, return_counts=True)
    cumulative = np.cumsum(counts[::-1])[::-1]
    cumulative = cumulative / cumulative[0]  # Normalize the cumulative values
    return values, cumulative

def expcum_fit(x, a):
    return 1 - np.exp(-a * x)

def plot_FSC_network_3(ax, piprob, pitilde_prob,
                       memory1_color='gray', memory2_color='gray', memory3_color='gray',
                       action_r_color='lightblue', action_t_color='salmon',
                       title=""):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add memory nodes
    G.add_node('$M_1$', shape='o', color=memory1_color)
    G.add_node('$M_2$', shape='o', color=memory2_color)
    G.add_node('$M_3$', shape='o', color=memory3_color)
    
    # Add action nodes
    G.add_node('R1', shape='s', color=action_r_color)
    G.add_node('T1', shape='s', color=action_t_color)
    G.add_node('R2', shape='s', color=action_r_color)
    G.add_node('T2', shape='s', color=action_t_color)
    G.add_node('R3', shape='s', color=action_r_color)
    G.add_node('T3', shape='s', color=action_t_color)
    
    # Add edges from memory nodes to action nodes
    G.add_edges_from([('$M_1$', 'R1'), ('$M_1$', 'T1'),
                      ('$M_2$', 'R2'), ('$M_2$', 'T2'),
                      ('$M_3$', 'R3'), ('$M_3$', 'T3')])
    
    # Add edges from action nodes to memory nodes
    G.add_edges_from([('R1', '$M_1$'), ('R1', '$M_2$'), ('R1', '$M_3$'),
                      ('T1', '$M_1$'), ('T1', '$M_2$'), ('T1', '$M_3$'),
                      ('R2', '$M_1$'), ('R2', '$M_2$'), ('R2', '$M_3$'),
                      ('T2', '$M_1$'), ('T2', '$M_2$'), ('T2', '$M_3$'),
                      ('R3', '$M_1$'), ('R3', '$M_2$'), ('R3', '$M_3$'),
                      ('T3', '$M_1$'), ('T3', '$M_2$'), ('T3', '$M_3$')])
    
    # Define node shapes
    node_shapes = {'o': 'o', 's': 's'}
    
    # Define custom labels for plotting
    labels = {'$M_1$': '$M_1$', '$M_2$': '$M_2$', '$M_3$': '$M_3$', 
              'R1': 'R', 'T1': 'T', 'R2': 'R', 'T2': 'T', 'R3': 'R', 'T3': 'T'}
    
    # Define positions for the nodes
    pos = {
        '$M_1$': (0, 0),
        'R1': (0, 1),
        'T1': (0, -1),
        '$M_2$': (2, 0),
        'R2': (2, 1),
        'T2': (2, -1),
        '$M_3$': (4, 0),
        'R3': (4, 1),
        'T3': (4, -1)
    }

    ax.axis('off')
    
    # Draw the graph
    for shape in node_shapes:
        nodes = [n for n in G.nodes if G.nodes[n]['shape'] == shape]
        colors = [G.nodes[n]['color'] for n in nodes]
        node_size = 2000 if shape == 'o' else 400  # Larger size for memory nodes, smaller for action nodes
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=node_shapes[shape],
                               node_size=node_size, node_color=colors, ax=ax)
    
    # Draw edges with arrows and curved connections
    memory_to_action_edges = [
        [('$M_1$', 'R1', piprob[0, 0]), ('$M_1$', 'T1', piprob[0, 1])],
        [('$M_2$', 'R2', piprob[1, 0]), ('$M_2$', 'T2', piprob[1, 1])],
        [('$M_3$', 'R3', piprob[2, 0]), ('$M_3$', 'T3', piprob[2, 1])]
    ]

    action_to_memory_edges = [
        [('R1', '$M_1$', pitilde_prob[0, 0, 0]), ('R1', '$M_2$', pitilde_prob[0, 1, 0]), ('R1', '$M_3$', pitilde_prob[0, 2, 0])],
        [('T1', '$M_1$', pitilde_prob[0, 0, 1]), ('T1', '$M_2$', pitilde_prob[0, 1, 1]), ('T1', '$M_3$', pitilde_prob[0, 2, 1])],
        [('R2', '$M_1$', pitilde_prob[1, 0, 0]), ('R2', '$M_2$', pitilde_prob[1, 1, 0]), ('R2', '$M_3$', pitilde_prob[1, 2, 0])],
        [('T2', '$M_1$', pitilde_prob[1, 0, 1]), ('T2', '$M_2$', pitilde_prob[1, 1, 1]), ('T2', '$M_3$', pitilde_prob[1, 2, 1])],
        [('R3', '$M_1$', pitilde_prob[2, 0, 0]), ('R3', '$M_2$', pitilde_prob[2, 1, 0]), ('R3', '$M_3$', pitilde_prob[2, 2, 0])],
        [('T3', '$M_1$', pitilde_prob[2, 0, 1]), ('T3', '$M_2$', pitilde_prob[2, 1, 1]), ('T3', '$M_3$', pitilde_prob[2, 2, 1])]
    ]
    
    # Draw edges with different colors and connection styles
    pscale = 5
    rads_memory_to_action = [0.5, 0.5, 0.5]
    rads_action_to_memory = [-0.2, 0.2, -0.2]

    for i in range(len(memory_to_action_edges)):
        for j, (u, v, prob) in enumerate(memory_to_action_edges[i]):
            width = prob * pscale

            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rads_memory_to_action[i]}', width=width,
                                arrows=True, arrowsize=20, min_source_margin=25, min_target_margin=15, edge_color='gray', ax=ax)
            
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, f'{prob:.3f}', fontsize=10, color='gray')

    colors = {'R1': 'lightblue', 'T1': 'salmon', 'R2': 'lightblue', 'T2': 'salmon', 'R3': 'lightblue', 'T3': 'salmon'}

    for i in range(len(action_to_memory_edges)):
        for j, (u, v, prob) in enumerate(action_to_memory_edges[i]):
            width = prob * pscale

            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rads_action_to_memory[i % 3]}', width=width,
                                   arrows=True, arrowsize=20, min_source_margin=15, min_target_margin=25, edge_color=colors[u], ax=ax)
            
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, f'{prob:.3f}', fontsize=10, color=colors[u])
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white', ax=ax)
    
    # Add title
    ax.set_title(title)

    cut = 1.2
    xmax= cut*max(xx for xx,yy in pos.values())
    ax.set_xlim(-0.5,xmax)
