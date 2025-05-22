import networkx as nx
import numpy as np
import time
import multiprocessing
import pycombo
from itertools import combinations
from collections import defaultdict
from gurobipy import *
from networkx.algorithms.connectivity import minimum_st_node_cut
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from itertools import chain
import random
import EstimateUB
from typing import Optional, Tuple, Dict, List

# global var for pycombo execution
start_separate = None


def is_graph(graph) -> bool:
    """
    Copied from PyCombo source code
    """
    graph_names = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}
    return type(graph).__name__ in graph_names


def is_weighted(G, edge: Optional[tuple] = None, weight: str = "weight"):
    """
    Copied from PyCombo source code
    """
    if edge is not None:
        data = G.get_edge_data(*edge)
        if data is None:
            raise ValueError(f"Edge {edge!r} does not exist.")
        return weight in data

    if not any(G.adj.values()):  # nx.is_empty()
        # Special handling required since: all([]) == True
        return False

    return all(weight in data for u, v, data in G.edges(data=True))


def deconstruct_graph(graph, weight: Optional[str] = None):
    """
    Copied from PyCombo source code
    """
    default_: int = 1

    if weight is not None:
        if is_weighted(graph, weight=weight):
            default_ = 0
        else:
            print(f"No property found: `{weight}`. Using as unweighted graph")

    nodenum, nodes = dict(), dict()
    for i, n in enumerate(graph.nodes()):
        nodenum[n] = i
        nodes[i] = n

    edges = []
    for edge in graph.edges(
        data=True
    ):  # NOTE: could switch to data=False and save a few milliseconds for unweighted graph
        edges.append(
            (nodenum[edge[0]], nodenum[edge[1]], edge[2].get(weight, default_))
        )
    return nodes, edges


def clique_filtering(G):
    """
    Returns G' which is a clique reduction on the input graph G
    """
    lcc_dict = {node: get_local_clustering_coefficient(G, node) for node in G.nodes()}
    shrink_dict = {}
    for node in G.nodes():
        if node in shrink_dict:
            continue
        neighbors = list(G.neighbors(node))

        # merge if pendant node has positive-degree
        if len(neighbors) == 1:
            neighbor = neighbors[0]
            if G[node][neighbor]['weight'] > 0:
                shrink_dict[node] = neighbor
            continue
    
        count_of_ones = 0
        count_of_not_one = 0
        not_one_neighbour = -1

        for neighbour in neighbors:
            if lcc_dict[neighbour] == 1:
                count_of_ones += 1
            else:
                count_of_not_one += 1
                not_one_neighbour = neighbour

        # only merge iff node and its neighbors form a pendant clique with only positive internal edges
        if count_of_ones == len(neighbors) - 1 and count_of_not_one == 1 and lcc_dict[node] == 1:
            all_positive_weights = all(G[i][j]['weight'] > 0 for i in neighbors for j in neighbors if i != j) and \
                            all(G.edges[(node, i)]['weight'] > 0 for i in neighbors)
            if all_positive_weights:
                shrink_dict[node] = not_one_neighbour
        if node not in shrink_dict:
            shrink_dict[node] = node

    G_prime = G.copy()
    for key in shrink_dict:
        supernode = shrink_dict[key]
        if key != supernode:
            if supernode not in G_prime:
                continue

            G_prime.nodes[supernode]['super node of'].extend(G_prime.nodes[key].get('super node of', []))
            
            edges_to_del = list(G_prime.edges(key))
            total_weight = 0
            for edge in edges_to_del:
                total_weight += G_prime.edges[edge]['weight']
            
            if G_prime.has_edge(supernode, supernode):
                G_prime.edges[(supernode, supernode)]['weight'] += total_weight
                G_prime.edges[(supernode, supernode)]['constrained_weight'] = False
            else:
                G_prime.add_edge(supernode, supernode, weight=total_weight, constrained_weight=False)

            G_prime.remove_node(key)

    G_prime = nx.convert_node_labels_to_integers(G_prime, first_label=0)
    return G_prime

def find_list_of_cut_triads(G):
    list_of_cut_triads = []

    pairs = set(combinations(np.sort(list((G).nodes())), 2))
    self_edges = set([(i, i) for i in (G).nodes()])
    pairs_with_edges = set((G).edges()) - self_edges
    pairs_without_edges = list(pairs - pairs_with_edges)
    pairs_with_edges = list(pairs_with_edges)

    # JOBLIB
    with parallel_backend(backend='loky', n_jobs=-1):
        res = Parallel()(delayed(separating_set_parallel)(pair[0], pair[1], G) for pair in pairs_without_edges)
    list_of_cut_triads = list(chain(*res))
    for pair in pairs_with_edges:
        i = pair[0]
        j = pair[1]
        removed_edge = False
        if G.has_edge(i, j):
            removed_edge = True
            attr_dict = G.edges[i, j]
            G.remove_edge(i, j)
        minimum_vertex_cut = minimum_st_node_cut(G, i, j)
        for k in minimum_vertex_cut:
            list_of_cut_triads.append(list(np.sort([i, j, k])))
        if removed_edge:
            G.add_edge(i, j, weight=attr_dict["weight"], constrained_weight=attr_dict["constrained_weight"])

    return list_of_cut_triads


def get_local_clustering_coefficient(G, node: int):
    """
    Returns the clustering coefficient for the input node in the input graph
    """
    neighbors = list(nx.neighbors(G, node))
    if len(neighbors) <= 1:
        return 0.0
    num_actual_edges = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if G.has_edge(neighbors[i], neighbors[j]):
                num_actual_edges += 1
    num_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
    clustering_coefficient = num_actual_edges / num_possible_edges
    return clustering_coefficient


def find_in_list_of_list(mylist, char):
    """
    Searches for 'char' in each sublist of 'mylist' and returns the index of the first sublist containing 'char'
    """
    for i, sub_list in enumerate(mylist):
        if char in sub_list:
            return i
    raise ValueError("'{char}' is not in list".format(char=char))


def model_to_communities(var_vals, graph):
    """
    Using DFS, divides graph into communities using Gurobi IP solution, returning a list of clusters
    """
    communities = []
    visited = set()

    def dfs(node, community):
        visited.add(node)
        community.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                edge_key = (str(node)+','+str(neighbor)) \
                    if (str(node)+','+str(neighbor)) in var_vals \
                    else (str(neighbor)+','+str(node)) 
                if edge_key in var_vals:
                    var = var_vals[edge_key]
                    if var < 0.5:
                        dfs(neighbor, community)

    for node in sorted(graph.nodes()):
        if node not in visited:
            community = set()
            dfs(node, community) 
            communities.append(list(community))
    return [list(np.sort(com)) for com in communities]


def decluster_communities(group, Graph, isolated_nodes):
    """
    Method to get communities based on the original graph. Note, input Graph is the reduced graph.
    """
    group_declustered = []
    for comm in group:
        new_comm = []
        for node in comm:
            if 'super node of' in Graph.nodes[int(node)]:
                node_list = Graph.nodes[int(node)]['super node of']
                new_comm = new_comm + node_list
            else:
                new_comm.append(int(node))
        group_declustered.append(new_comm)
    for n in isolated_nodes:
        group_declustered.append([n])
    for c in range(len(group_declustered)):
        group_declustered[c].sort()
    group_declustered.sort()
    return group_declustered


def separating_set_parallel(i, j, Graph):
    """
    Function to find sepearating set for 2 nodes in graph
    """
    triads = []
    minimum_vertex_cut = minimum_st_node_cut(Graph, i, j)
    for k in minimum_vertex_cut:
        triads.append(list(np.sort([i, j, k])))
    return triads


def post_processing(var_vals, Graph):
    '''
    The processing step to ammend the results of the RP*(G) formulation
    '''
    G_new = defaultdict(set)
    for (i, j) in Graph.edges():
        i, j = tuple(sorted((i, j)))
        if var_vals.get(str(i) + ',' + str(j)) == 0 and Graph[i][j]['weight'] > 0:
            G_new[i].add(j)
            G_new[j].add(i)

    def dfs(g, start, visited):
        visited.add(start)
        for neighbor in g[start]:
            if neighbor not in visited:
                dfs(g, neighbor, visited)

    visited = set()
    components = []
    for node in G_new:
        if node not in visited:
            component = set()
            dfs(G_new, node, component)
            components.append(component)
            visited.update(component)
    var_vals_pp = {}
    for key in var_vals.keys():
        var_vals_pp[key] = 1.0
    OFV = 0.0
    for c in components:
        c = list(np.sort(list(c)))
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                node_i = c[i]
                node_j = c[j]
                var_vals_pp[str(node_i) + ',' + str(node_j)] = 0.0
                if Graph.has_edge(node_i, node_j):
                    OFV += Graph[node_i][node_j]['weight']
    return OFV, var_vals_pp


def lp_formulation(Graph, list_of_cut_triads, lp_method, warmstart=int(0), branching_priotiy=int(0)):
    """
    Method to create the LP model and run it for the root node
    """
    formulation_time_start = time.time()

    x = {}
    model = Model("Clique Partitioning")
    model.setParam(GRB.param.OutputFlag, 0)
    model.setParam(GRB.param.Method, lp_method)
    model.setParam(GRB.Param.Crossover, 0)
    model.setParam(GRB.Param.Threads, min(64, multiprocessing.cpu_count()))

    for i in range(Graph.number_of_nodes()):
        for j in range(i + 1, Graph.number_of_nodes()):
            x[(i, j)] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=str(i) + ',' + str(j))
    model.update()

    OFV = sum(Graph[i][j]['weight'] * (1 - x[tuple(sorted((i, j)))]) for i, j in Graph.edges() if (i, j) in x or (j, i) in x)
    model.setObjective(OFV, GRB.MAXIMIZE)

    for [i, j, k] in list_of_cut_triads:
        if (Graph.has_edge(i, k) and Graph[i][k]['weight'] > 0) or (Graph.has_edge(j, k) and Graph[j][k]['weight'] > 0):
            model.addConstr(x[(i, k)] + x[(j, k)] >= x[(i, j)], 'triangle1' + ',' + str(i) + ',' + str(j) + ',' + str(k))
        if (Graph.has_edge(j, k) and Graph[j][k]['weight'] > 0) or (Graph.has_edge(i, j) and Graph[i][j]['weight'] > 0):
            model.addConstr(x[(j, k)] + x[(i, j)] >= x[(i, k)], 'triangle2' + ',' + str(i) + ',' + str(j) + ',' + str(k))
        if (Graph.has_edge(i, j) and Graph[i][j]['weight'] > 0) or (Graph.has_edge(i, k) and Graph[i][k]['weight'] > 0):
            model.addConstr(x[(i, j)] + x[(i, k)] >= x[(j, k)], 'triangle3' + ',' + str(i) + ',' + str(j) + ',' + str(k))

    formulation_time = time.time() - formulation_time_start
    model.update()

    # Using a warm start
    # A known partition can be used as a starting point to "warm-start" the algorithm.
    # It can be provided to the model here by giving start values to the decision variables:
    # If this optional step is skipped, you should comment out these five lines
    if warmstart == 1:
        # Solution from Combo algorithm to be used as warm-start
        partition = pycombo.execute(Graph, weight="weight", treat_as_modularity=True)
        community_combo = convert_to_com_list(partition[0])
        for i in (Graph).nodes():
            for j in filter(lambda x: x > i, (Graph).nodes()):
                if find_in_list_of_list(community_combo, i) == find_in_list_of_list(community_combo, j):
                    x[(i, j)].start = 0
                else:
                    x[(i, j)].start = 1
    # branching priority is based on total degrees of pairs of nodes
    if branching_priotiy == 1:
        neighbors = {}
        Degree = []
        for i in range(len(Graph.nodes())):
            for j in range(i + 1, len(Graph.nodes())):
                neighbors[i] = list((Graph)[i])
                neighbors[j] = list((Graph)[j])
                Degree.append(len(neighbors[i]) + len(neighbors[j]))
        model.setAttr('BranchPriority', model.getVars()[:], Degree)
        model.update()

    start_time = time.time()
    model.optimize()
    solveTime = (time.time() - start_time)
    obj = model.getObjective()

    objective_value = np.round(obj.getValue() + sum(data['weight'] for u, v, data in Graph.edges(data=True) if u == v), 8)

    var_vals = {}
    for var in model.getVars():
        var_vals[var.varName] = var.x

    return objective_value, var_vals, model, formulation_time, solveTime


def run_ip(model, Graph, fixed_ones, fixed_zeros):
    """
    Run the IP based on model and the original Graph as input, this should only be used when there are no violating triples
    """
    for var_name in fixed_ones:
        var = model.getVarByName(var_name)
        var.setAttr("LB", 1.0)
    model.update()
    for var_name in fixed_zeros:
        var = model.getVarByName(var_name)
        var.setAttr("UB", 0.0)
    model.update()
    for var in model.getVars():
        var.setAttr(GRB.Attr.VType, GRB.BINARY)
    model.update()

    obj = model.getObjective()
    try:
        obj_val = obj.getValue()
    except AttributeError as error:
        return -1, -1, model

    objective_value = np.round(obj.getValue() + sum(data['weight'] for u, v, data in Graph.edges(data=True) if u == v), 8)

    var_vals = {}
    for var in model.getVars():
        var_vals[var.varName] = var.X
    return objective_value, var_vals, model


def run_lp(model, Graph, fixed_ones, fixed_zeros):
    """
    Run the LP based on model and the original Graph as input
    """
    for var_name in fixed_ones:
        var = model.getVarByName(var_name)
        var.setAttr("LB", 1.0)
    model.update()
    for var_name in fixed_zeros:
        var = model.getVarByName(var_name)
        var.setAttr("UB", 0.0)
    model.update()

    model.optimize()

    obj = model.getObjective()
    try:
        obj_val = obj.getValue()
    except AttributeError as error:
        return -1, -1, model

    objective_value = np.round(obj.getValue() + sum(data['weight'] for u, v, data in Graph.edges(data=True) if u == v), 8)

    var_vals = {}
    for var in model.getVars():
        var_vals[var.varName] = var.x

    return objective_value, var_vals, model


def run_combo(graph, original_graph, isolated_nodes, timeout=3):
    """
    Runs combo algorithm on input graph
    """
    global start_separate

    # Function to execute pycombo with a specified start_separate value
    def execute_pycombo(ss):
        return pycombo.execute(graph, weight='weight', treat_as_modularity=True, start_separate=ss)
    
    # first execution
    if start_separate is None:
        partition_combo = execute_pycombo(ss=False)
        communities_combo = convert_to_com_list(partition_combo[0])
        decluster_combo = decluster_communities(communities_combo, graph, isolated_nodes)
        lower_bound = calculate_objective_value(decluster_combo, original_graph)

        if lower_bound == 0:
            start_separate = True

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_pycombo, True)
            try:
                partition_combo = future.result(timeout=timeout)
                communities_combo = convert_to_com_list(partition_combo[0])
                decluster_combo = decluster_communities(communities_combo, graph, isolated_nodes)
                lower_bound = calculate_objective_value(decluster_combo, original_graph)
                start_separate = True
            except TimeoutError:
                start_separate = True if lower_bound == 0 else False

    # if first execution timed out
    elif start_separate is False:
        partition_combo = execute_pycombo(ss=False)
        communities_combo = convert_to_com_list(partition_combo[0])
        decluster_combo = decluster_communities(communities_combo, graph, isolated_nodes)
        lower_bound = calculate_objective_value(decluster_combo, original_graph)
    
    # if first execution didn't timeout or lower_bound == 0
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_pycombo, True)
            try:
                partition_combo = future.result(timeout=600)
                communities_combo = convert_to_com_list(partition_combo[0])
                decluster_combo = decluster_communities(communities_combo, graph, isolated_nodes)
                lower_bound = calculate_objective_value(decluster_combo, original_graph)
            except TimeoutError:
                print("Timeout occurred during PyCombo execution with start_separate=True")
    
    return lower_bound, decluster_combo


def reset_model_variables(model, fixed_ones, fixed_zeros):
    """
    Reset fixed variables of the LP model
    """
    for var_name in fixed_ones:
        var = model.getVarByName(var_name)
        var.setAttr("LB", 0.0)
    model.update()
    for var_name in fixed_zeros:
        var = model.getVarByName(var_name)
        var.setAttr("UB", 1.0)
    model.update()
    return model


# In[12]:

def calculate_objective_value(communities, Graph):
    """
    Computes the CP problem objective value through list of clusters, and returns it
    """
    OFV = 0
    for community in communities:
        for i in range(len(community)):
            for j in range(i, len(community)):
                if Graph.has_edge(community[i], community[j]):
                    OFV += Graph[community[i]][community[j]]['weight']    
    return np.round(OFV, 8)


def calculate_model_objective_value(var_vals, Graph):
    """
    Computes the CP problem objective value through LP model solution and returns it
    **Function currently not in use**
    """
    weights_matrix = nx.adjacency_matrix(Graph, weight='weight').astype(float)
    decision_vars_matrix = np.zeros_like(weights_matrix.toarray())
    for var_name, value in var_vals.items():
        i, j = var_name.split(',')
        i, j = int(i), int(j)
        value = 0 if value == 1 else 1
        decision_vars_matrix[i, j] = value
        decision_vars_matrix[j, i] = value
    hadamard_product = np.multiply(weights_matrix.toarray(), decision_vars_matrix)
    return np.round(np.sum(hadamard_product) / 2, 8)
    

def find_violating_triples(G, var_vals, list_of_cut_triads):
    """
    Returns a dictionary whose key is a violated constraint and value is the sum, 
    triples with high count of positive edge weights are heavily prioritized
    """
    t_3, t_2, t_1, t_0 = {}, {}, {}, {}
    for [i, j, k] in list_of_cut_triads:
        triple_sum = var_vals[str(i) + "," + str(j)] + var_vals[str(j) + "," + str(k)] + var_vals[str(i) + "," + str(k)]
        if 0 < triple_sum < 2:
            num_positive_edges = 0
            if G.has_edge(i, j) and G[i][j]['weight'] > 0:
                num_positive_edges += 1
            if G.has_edge(i, k) and G[i][k]['weight'] > 0:
                num_positive_edges += 1
            if G.has_edge(j, k) and G[j][k]['weight'] > 0:
                num_positive_edges += 1
            if num_positive_edges == 3:
                t_3[(i, j, k)] = triple_sum
            elif num_positive_edges == 2:
                t_2[(i, j, k)] = triple_sum
            elif num_positive_edges == 1:
                t_1[(i, j, k)] = triple_sum
            else:
                t_0[(i, j, k)] = triple_sum
    if len(t_3) > 0:
        return t_3
    elif len(t_2) > 0:
        return t_2
    elif len(t_1) > 0:
        return t_1
    else:
        return t_0


def get_best_triple(violated_triples_sums, node, orig_g):
    """
    Returns the constraint with the most in common with the previous nodes constraint
    """

    num_nodes = len(list(orig_g.nodes()))
    score_list = []
    violated_triples = list(violated_triples_sums.keys()) if len(violated_triples_sums) <= 100 \
        else random.sample(list(violated_triples_sums.keys()), 100)
    for triple in violated_triples:
        new_triple = [-1, -1, -1]
        value_zero = orig_g.nodes[triple[0]]['super node of'][0]
        value_one = orig_g.nodes[triple[1]]['super node of'][0]
        value_two = orig_g.nodes[triple[2]]['super node of'][0]
        for n in node.graph.nodes():
            # check if any of the values in the triples are a part of the supernode n
            if value_zero in node.graph.nodes[n]["super node of"]:
                new_triple[0] = n
            if value_one in node.graph.nodes[n]["super node of"]:
                new_triple[1] = n
            if value_two in node.graph.nodes[n]["super node of"]:
                new_triple[2] = n
            if new_triple[0] != -1 and new_triple[1] != -1 and new_triple[2] != -1:
                break
        total_score = 0
        for t in range(3):
            alpha = 0
            for i in range(triple[t] + 1, num_nodes):
                variable = str(triple[t]) + "," + str(i)
                if variable in node.get_fixed_ones():
                    alpha += 1
                if variable in node.get_fixed_zeros():
                    alpha += 1
            beta = 0
            if node.parent is not None:
                for constr in node.constraints:
                    if triple[t] in constr[:3]:
                        beta += 1
            delta = node.graph.degree(new_triple[t], weight="weight")
            score = 1 - np.exp(-alpha) + beta + abs(delta / node.graph.number_of_nodes())
            total_score += score
        score_list.append(total_score)
    sum_of_scores = np.sum(score_list)
    probability_list = [x / sum_of_scores for x in score_list]
    index = np.random.choice(len(violated_triples), p=probability_list)
    return violated_triples[index][:3]


def is_integer_solution(Graph, var_vals):
    """
    Return whether all the varaible values are integer
    """
    for i in range(len(Graph.nodes())):
        for j in range(i + 1, len(Graph.nodes())):
            if var_vals[str(i) + "," + str(j)] != 1 and var_vals[str(i) + "," + str(j)] != 0:
                return False
    return True


def reduce_triple(G, triple, orig_g):
    """
    Reduces G by creating a supernode for nodes in triple (for left branching).
    Returns the reduced graph
    """
    new_triple = [-1, -1, -1]
    value_zero = orig_g.nodes[triple[0]]['super node of'][0]
    value_one = orig_g.nodes[triple[1]]['super node of'][0]
    value_two = orig_g.nodes[triple[2]]['super node of'][0]
    for node in G.nodes():
        if value_zero in G.nodes[node]["super node of"]:
            new_triple[0] = node
        if value_one in G.nodes[node]["super node of"]:
            new_triple[1] = node
        if value_two in G.nodes[node]["super node of"]:
            new_triple[2] = node
    triple = new_triple
    Graph = G.copy()
    self_weight = 0
    for u in range(3):
        for v in range(u, 3):
            if Graph.has_edge(triple[u], triple[v]):
                if 'weight' in Graph.edges[triple[u], triple[v]]:
                    self_weight += Graph.edges[triple[u], triple[v]]["weight"]
    if Graph.has_edge(triple[0], triple[0]):
        Graph.edges[triple[0], triple[0]]["weight"] = self_weight
        curr = Graph.nodes[triple[0]]['super node of']
        new1 = Graph.nodes[triple[1]]['super node of']
        new2 = Graph.nodes[triple[2]]['super node of']
        super_list = list(set(curr + new1 + new2))
        super_list.sort()
        Graph.nodes[triple[0]]['super node of'] = super_list
    else:
        Graph.add_edge(triple[0], triple[0])
        Graph.edges[triple[0], triple[0]]["weight"] = self_weight
        Graph.edges[triple[0], triple[0]]['constrained_weight'] = False
        curr = Graph.nodes[triple[0]]['super node of']
        new1 = Graph.nodes[triple[1]]['super node of']
        new2 = Graph.nodes[triple[2]]['super node of']
        super_list = list(set(curr + new1 + new2))
        super_list.sort()
        Graph.nodes[triple[0]]['super node of'] = super_list

    if triple[0] != triple[1]:
        edge_list = list(Graph.edges(triple[1]))
        for edge in edge_list:
            if edge[1] not in [triple[0], triple[2]]:
                if Graph.has_edge(triple[0], edge[1]):
                    Graph.edges[triple[0], edge[1]]['weight'] += Graph.edges[edge]['weight']
                else:
                    Graph.add_edge(triple[0], edge[1])
                    Graph.edges[triple[0], edge[1]]['weight'] = Graph.edges[edge]['weight']
                    Graph.edges[triple[0], edge[1]]['constrained_weight'] = False
        Graph.remove_node(triple[1])

    if triple[0] != triple[2] and triple[1] != triple[2]:
        edge_list = list(Graph.edges(triple[2]))
        for edge in edge_list:
            if edge[1] not in [triple[0], triple[1]]:
                if Graph.has_edge(triple[0], edge[1]):
                    Graph.edges[triple[0], edge[1]]['weight'] += Graph.edges[edge]['weight']
                else:
                    Graph.add_edge(triple[0], edge[1])
                    Graph.edges[triple[0], edge[1]]['weight'] = Graph.edges[edge]['weight']
                    Graph.edges[triple[0], edge[1]]['constrained_weight'] = False
        Graph.remove_node(triple[2])

    Graph = nx.convert_node_labels_to_integers(Graph)
    edge_weights = [abs(orig_g[i][j]['weight']) for i, j in orig_g.edges()]
    delta = abs(np.median(edge_weights))
    for i in range(Graph.number_of_nodes()):
        for j in range(i, Graph.number_of_nodes()):
            if Graph.has_edge(i, j):
                if Graph.edges[(i, j)]['constrained_weight']:
                    Graph.edges[(i, j)]['weight'] = Graph.edges[(i, j)]['weight'] - delta
    return Graph


def alter_weight(G, triple, orig_g):
    """
    Alter the weight associated with nodes in triple by median of all edge weights (for right branching).
    Returns the reduced graph
    """
    new_triple = [-1, -1, -1]
    value_zero = orig_g.nodes[triple[0]]['super node of'][0]
    value_one = orig_g.nodes[triple[1]]['super node of'][0]
    value_two = orig_g.nodes[triple[2]]['super node of'][0]
    for node in G.nodes():
        if value_zero in G.nodes[node]["super node of"]:
            new_triple[0] = node
        if value_one in G.nodes[node]["super node of"]:
            new_triple[1] = node
        if value_two in G.nodes[node]["super node of"]:
            new_triple[2] = node
    triple = new_triple

    AdjancencyMatrix = nx.adjacency_matrix(G, weight="weight")
    edge_weights = [abs(orig_g[i][j]['weight']) for i, j in orig_g.edges()]
    delta = abs(np.median(edge_weights))

    Graph = G.copy()
    num_nodes = len(Graph.nodes())
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (i, j) in [(triple[0], triple[1]), (triple[0], triple[2]), (triple[1], triple[2]),
                            (triple[1], triple[0]), (triple[2], triple[0]), (triple[2], triple[1])] and Graph.has_edge(i,
                                                                                                                     j):
                Graph.edges[(i, j)]['weight'] = AdjancencyMatrix[i, j] - delta
                Graph.edges[(i, j)]['constrained_weight'] = True
            elif Graph.has_edge(i, j):
                Graph.edges[(i, j)]['constrained_weight'] = False
    return Graph


def reduced_cost_variable_fixing(model, var_vals, obj_value, lower_bound):
    vars_one = []
    vars_zero = []
    for key in var_vals.keys():
        var = model.getVarByName(key)
        if var_vals[key] == 1:
            if obj_value - var.getAttr(GRB.Attr.RC) < lower_bound:
                vars_one.append(key)
        elif var_vals[key] == 0:
            if obj_value + var.getAttr(GRB.Attr.RC) < lower_bound:
                vars_zero.append(key)
    return vars_one, vars_zero


class Node:
    """
    Represents one node in the troika tree
    """

    def __init__(self, constraint_list, var_vals, g, combo_comms):
        self.constraints = constraint_list
        self.var_vals = var_vals
        self.lower_bound = None
        self.upper_bound = None
        self.graph = g
        self.left = None
        self.right = None
        self.parent = None
        self.close = False
        self.is_integer = False
        self.is_infeasible = False
        self.level = -1
        self.fixed_zeros = []
        self.fixed_ones = []
        for com in combo_comms:
            com.sort()
        combo_comms.sort()
        self.combo_communities = combo_comms

    def set_bounds(self, lb, ub):
        self.lower_bound = lb
        self.upper_bound = ub

    def get_violated_triples(self, list_of_cut_triads):
        return find_violating_triples(self.graph, self.var_vals, list_of_cut_triads)

    def close_node(self):
        self.close = True

    def set_is_integer(self):
        self.is_integer = True

    def set_is_infeasible(self):
        self.is_infeasible = True

    def set_level(self, l):
        self.level = l

    def get_constraints(self):
        return self.constraints

    def set_fixed_ones(self, ones):
        self.fixed_ones = ones

    def set_fixed_zeros(self, zeros):
        self.fixed_zeros = zeros

    def get_fixed_ones(self):
        return self.fixed_ones

    def get_fixed_zeros(self):
        return self.fixed_zeros


def create_troika_edge_attributes(G):
    # 'super node of' stores all the nodes that are a part of this super node
    for edge in G.edges():
        G.edges[edge]['constrained_weight'] = False
    for node in G.nodes():
        G.nodes[node]['super node of'] = [node]
    return G


def handle_isolated_nodes(Graph):
    """
    Remove isolated nodes from graph and return new graph and a list of isolated nodes separately
    """
    isolated = []
    for x in Graph.nodes():
        if Graph.degree[x] == 0:
            isolated.append(x)
    for x in isolated:
        Graph.remove_node(x)
    Graph = nx.convert_node_labels_to_integers(Graph)
    return Graph, isolated


def convert_to_com_list(com_dict):
    """
    Turns combo algorithm output into a list of communities
    """
    out = {}
    for node, com_id in com_dict.items():
        if com_id in out.keys():
            out[com_id].append(node)
        else:
            out[com_id] = [node]

    out = [com for com in out.values()]
    return out


def output(develop, state, lower_bound, upper_bound, communities, preprocessing_time, formulation_time, solve_time):
    """
    Final output wrapper function
    """
    if develop:
        out = state, lower_bound, upper_bound, communities, preprocessing_time, formulation_time, solve_time
    else:
        if upper_bound == 0:
            gap = 0
        else:
            gap = (upper_bound - lower_bound) / upper_bound
        out = lower_bound, gap, communities, preprocessing_time + formulation_time, solve_time
    return out


def troika(G, global_threshold=0.001, time_allowed=600, lp_method=4, develop_mode=False):
    """
    Troika algorithm
    """
    # Running troika for a network with multiple connected components
    optimal_partition = []
    list_of_subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    n_sub = len(list_of_subgraphs)
    sub_results = {}
    total_gap = 0
    total_modeling_time = 0
    total_solve_time = 0
    total_preprocessing_time = 0
    total_formulation_time = 0
    threshold_sub = global_threshold / float(n_sub)
    for sub_inx, sub_graph in enumerate(list_of_subgraphs):
        if sub_graph.number_of_edges() == 0:
            optimal_partition.append([a for a in sub_graph.nodes()])
            continue

        sub_graph = nx.convert_node_labels_to_integers(sub_graph, label_attribute="original_label")
        mapping = nx.get_node_attributes(sub_graph, 'original_label')
        troika_output = alg(sub_graph, global_threshold=threshold_sub, time_allowed=time_allowed,
                             lp_method=lp_method, develop_mode=develop_mode)
        if develop_mode:
            sub_results[sub_inx] = troika_output
            optimal_partition += [[mapping[i] for i in com] for com in troika_output[3]]
            total_preprocessing_time += troika_output[4]
            total_formulation_time += troika_output[5]
            total_solve_time += troika_output[6]
        else:
            optimal_partition += [[mapping[i] for i in com] for com in troika_output[2]]
            total_gap += troika_output[1]
            total_modeling_time += troika_output[3]
            total_solve_time += troika_output[4]

    G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")
    mapping = nx.get_node_attributes(G, 'original_label')
    mapping = {val: key for key, val in mapping.items()}
    int_optimal_partition = [[mapping[i] for i in com] for com in optimal_partition]
    lower_bound = calculate_objective_value(int_optimal_partition, G)

    if develop_mode:
        for sub_inx, sub_graph in enumerate(list_of_subgraphs):
            print(f"Connected component {sub_inx}")
            print(sub_results[sub_inx])
        out = lower_bound, optimal_partition, total_preprocessing_time, total_formulation_time, total_solve_time
    else:
        out = lower_bound, total_gap, optimal_partition, total_modeling_time, total_solve_time

    return out


def alg(G, global_threshold=0.001, time_allowed=600, lp_method=4, develop_mode=False):
    """
    Run troika on input Graph while MIP gap > threshold and runtime < time_allowed (default is 600 seconds)
    """

    preprocessing_time_start = time.time()
    global start_separate
    start_separate = None
    G1 = G.copy()                           
    orig_graph = create_troika_edge_attributes(G1)   # orig_graph = troika edge attributes only
    G2 = orig_graph.copy()
    Graph, isolated_nodes = handle_isolated_nodes(G2)
    Graph = clique_filtering(Graph)             # Graph = isolated nodes handled + troika edge attributes + positive cliques and pendant nodes combined
    preprocessing_time = time.time() - preprocessing_time_start

    # get initial lower bound using Combo heuristic algorithm
    root_combo_time_start = time.time()
    obj_val_combo, communities_combo_declustered = run_combo(Graph, orig_graph, isolated_nodes)
    print('initial lower bound', obj_val_combo)
    root_combo_time = time.time() - root_combo_time_start


    # get initial upper bound estimate using best partition
    _, edge_list = deconstruct_graph(Graph, "weight")
    root_estimate_start = time.time()
    bp_upper_bound_estimate = EstimateUB.estimateUB_chains_fast(Graph.number_of_nodes(), edge_list)
    root_estimate_time = time.time() - root_estimate_start
    print("initial upper bound estimate", bp_upper_bound_estimate)
    if abs(bp_upper_bound_estimate - obj_val_combo) / bp_upper_bound_estimate < global_threshold:
        return output(develop_mode, 0, obj_val_combo, bp_upper_bound_estimate, communities_combo_declustered, preprocessing_time,
                        0, root_combo_time + root_estimate_time)


    # create initial LP formulation and run it
    cut_triads_time_start = time.time()
    list_of_cut_triads = find_list_of_cut_triads(Graph)
    cut_triads_time = time.time() - cut_triads_time_start
    obj_val_lp, var_vals, model, formulation_time, root_lp_time = lp_formulation(Graph,list_of_cut_triads, lp_method)
    formulation_time += cut_triads_time
    print('initial lp upper bound', obj_val_lp)
    if is_integer_solution(Graph, var_vals):
        obj_val_lp, var_vals = post_processing(var_vals, Graph)
        return output(develop_mode, 1, obj_val_lp, obj_val_lp,
                        decluster_communities(model_to_communities(var_vals, Graph), Graph, isolated_nodes),
                        preprocessing_time, formulation_time, root_lp_time)
    
    if abs(obj_val_lp - obj_val_combo) / obj_val_lp < global_threshold:
        return output(develop_mode, 2, obj_val_combo, obj_val_lp, communities_combo_declustered, preprocessing_time,
                        formulation_time, root_combo_time + root_lp_time + root_estimate_time)

    best_bound = min(obj_val_lp, bp_upper_bound_estimate)
    incumbent = obj_val_combo
    root = Node([], var_vals, Graph, communities_combo_declustered)
    root.set_level(0)
    root.set_bounds(obj_val_combo, obj_val_lp)
    var_fixed_ones, var_fixed_zeros = reduced_cost_variable_fixing(model, var_vals, obj_val_lp, incumbent)
    root.set_fixed_ones(var_fixed_ones)
    root.set_fixed_zeros(var_fixed_zeros)
    current_node = root
    current_level = 1
    nodes_previous_level = [root]
    best_combo = root
    best_lp = root
    root_time = root_lp_time + root_combo_time + root_estimate_time
    solve_start = time.time()
    while (incumbent < best_bound and abs(best_bound - incumbent) / best_bound > global_threshold and nodes_previous_level != []
             and time.time() - solve_start + root_time <= time_allowed):  # add time_limit as a user parameter
        print('==============================================LEVEL: ',current_level, '==============================================')
        nodes_current_level = []
        lower_bounds = []
        upper_bounds = []
        
        for node in nodes_previous_level:
            if time.time() - solve_start + root_time >= time_allowed:
                if best_combo.is_integer:
                    if best_combo.lower_bound <= best_combo.upper_bound:
                        obj_val, var_vals = post_processing(best_combo.var_vals, Graph)
                        return output(develop_mode, 3, obj_val, obj_val,
                                        decluster_communities(model_to_communities(var_vals, Graph), Graph,
                                                            isolated_nodes), preprocessing_time, formulation_time,
                                        time.time() - solve_start + root_time)
                    else:
                        return output(develop_mode, 4, best_combo.lower_bound, best_lp.upper_bound,
                                        best_combo.combo_communities, preprocessing_time, formulation_time,
                                        time.time() - solve_start + root_time)
                else:
                    return output(develop_mode, 5, best_combo.lower_bound, best_lp.upper_bound,
                                    best_combo.combo_communities, preprocessing_time, formulation_time,
                                    time.time() - solve_start + root_time)

            current_node = node
            left_node, right_node = perform_branch(node, model, incumbent, Graph, orig_graph,
                                                     isolated_nodes, list_of_cut_triads)

            if left_node.close and left_node.is_integer and incumbent <= left_node.upper_bound:
                incumbent = left_node.upper_bound
                best_combo = left_node
            if right_node.close and right_node.is_integer and incumbent <= right_node.upper_bound:
                incumbent = right_node.upper_bound
                best_combo = right_node

            current_node.left = left_node
            nodes_current_level.append(left_node)
            left_node.parent = current_node
            left_node.set_level(current_level)
            lower_bounds.append(left_node.lower_bound)
            if left_node.upper_bound >= incumbent:
                upper_bounds.append(left_node.upper_bound)
            else:
                left_node.close_node()

            current_node.right = right_node
            nodes_current_level.append(right_node)
            right_node.parent = current_node
            right_node.set_level(current_level)
            lower_bounds.append(right_node.lower_bound)
            if right_node.upper_bound >= incumbent:
                upper_bounds.append(right_node.upper_bound)
            else:
                right_node.close_node()

            incumbent = max(lower_bounds + [incumbent])
            if right_node.lower_bound == incumbent:
                best_combo = right_node
            elif left_node.lower_bound == incumbent:
                best_combo = left_node

        if len(upper_bounds) > 0:
            best_bound = max(upper_bounds)
        for n in nodes_current_level:
            if n.upper_bound == best_bound:
                best_lp = n
        current_level += 1
        nodes_p_level = []
        for a in nodes_current_level:
            if a.close == False:
                nodes_p_level.append(a)
        nodes_previous_level = nodes_p_level
    
        print("Bounds", incumbent, best_bound)

    if best_combo.is_integer:
        if best_combo.lower_bound <= best_combo.upper_bound:
            obj_val, var_vals = post_processing(best_combo.var_vals, Graph)
            return output(develop_mode, 6, obj_val, obj_val,
                            decluster_communities(model_to_communities(var_vals, Graph), Graph,
                                                isolated_nodes), preprocessing_time, formulation_time,
                            time.time() - solve_start + root_time)
        else:
            return output(develop_mode, 7, best_combo.lower_bound, best_lp.upper_bound, best_combo.combo_communities,
                            preprocessing_time, formulation_time, time.time() - solve_start + root_time)
    else:
        return output(develop_mode, 8, best_combo.lower_bound, best_lp.upper_bound, best_combo.combo_communities,
                        preprocessing_time, formulation_time, time.time() - solve_start + root_time)


def left_implied(left_fix_ones, left_fix_zeros, branch_triple):
    """
    Implied variable fixing for the left branch
    """
    ones = left_fix_ones.copy()
    zeros = left_fix_zeros.copy()
    i = branch_triple[0]
    j = branch_triple[1]
    k = branch_triple[2]
    for var in left_fix_zeros:
        first, second = var.split(",")
        if first == i:
            zeros.append(str(j) + "," + second)
            zeros.append(str(k) + "," + second)
            # FIX corresponding vars to 0 and append to left_fix_zeros
    for var in left_fix_ones:
        first, second = var.split(",")
        if first == i:
            # FIX corresponding vars to 1 and append to left_fix_ones
            ones.append(str(j) + "," + second)
            ones.append(str(k) + "," + second)
    return zeros, ones


def right_implied(right_fix_zeros, branch_triple):
    """
    Implied variable fixing for the right branch
    """
    i = branch_triple[0]
    j = branch_triple[1]
    k = branch_triple[2]
    constraints_to_be_added = []
    for var in right_fix_zeros:
        first, second = var.split(",")
        if first == i:
            constraint = [int(second), int(j), int(k)]
            constraint.sort()
            constraint_tup = (constraint[0], constraint[1], constraint[2], 2)
            constraints_to_be_added.append(constraint_tup)
    return constraints_to_be_added


def perform_branch(node, model, incumbent, Graph, original_graph, isolated_nodes, list_of_cut_triads):
    """
    Perform the left and right branch on input node
    """
    violated_triples_dict = node.get_violated_triples(list_of_cut_triads)    
    prev_fixed_ones = node.get_fixed_ones().copy()
    prev_fixed_zeros = node.get_fixed_zeros().copy()

    if len(violated_triples_dict) == 0:
        print('Solving IP')
        count = 0
        for constr in node.constraints:
            x_ij = model.getVarByName(str(constr[0]) + "," + str(constr[1]))
            x_jk = model.getVarByName(str(constr[1]) + "," + str(constr[2]))
            x_ik = model.getVarByName(str(constr[0]) + "," + str(constr[2]))
            if constr[3] == 0:
                model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_' + str(count))
            else:
                model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_' + str(count))
        upper_bound, var_vals, model = run_ip(model, Graph, prev_fixed_ones, prev_fixed_zeros)

        leaf_node = Node(node.constraints, var_vals, Graph, [])
        if upper_bound == -1 and var_vals == -1:
            leaf_node.close_node()
            leaf_node.set_is_infeasible()
            print('Infeasible IP Solution')
        else:
            if is_integer_solution(Graph, var_vals):
                print("IP solution is integer")
                leaf_node.set_is_integer()
                leaf_node.close_node()
            if upper_bound <= incumbent:
                leaf_node.close_node()
        return leaf_node, leaf_node
    
    # Select triple based on most common nodes with previous triple
    branch_triple = get_best_triple(violated_triples_dict, node, Graph)
    print("======== BRANCHING ON " + str(branch_triple) + " ========")
    x_ij = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[1]))
    x_jk = model.getVarByName(str(branch_triple[1]) + "," + str(branch_triple[2]))
    x_ik = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[2]))

    # Left branch x_ij + x_jk + x_ik = 0
    count = 0
    if node.constraints == []:
        model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_0')
        count += 1
    else:
        model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_0')
        count += 1
        for constr in node.constraints:
            x_ij = model.getVarByName(str(constr[0]) + "," + str(constr[1]))
            x_jk = model.getVarByName(str(constr[1]) + "," + str(constr[2]))
            x_ik = model.getVarByName(str(constr[0]) + "," + str(constr[2]))
            if constr[3] == 0:
                model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_' + str(count))
            else:
                model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_' + str(count))
            count += 1
    model.update()

    left_upper_bound, left_var_vals, model = run_lp(model, Graph, prev_fixed_ones, prev_fixed_zeros)
    if not (left_upper_bound == -1 and left_var_vals == -1):
        left_fix_ones, left_fix_zeros = reduced_cost_variable_fixing(model, left_var_vals, left_upper_bound, incumbent)
        model = reset_model_variables(model, prev_fixed_ones, prev_fixed_zeros)
        implied_zeros, implied_ones = left_implied(left_fix_ones, left_fix_zeros, branch_triple)

    # remove all cutting plane constraints after solving
    for i in range(count):
        model.remove(model.getConstrByName('branch_' + str(i)))
    model.update()

    left_graph = reduce_triple(node.graph, branch_triple, original_graph)
    left_lower_bound, left_decluster_combo = run_combo(left_graph, original_graph, isolated_nodes)

    left_constraints = node.constraints.copy()
    left_constraints.append(branch_triple + (0,))
    left_node = Node(left_constraints, left_var_vals, left_graph, left_decluster_combo)
    left_node.set_bounds(left_lower_bound, left_upper_bound)

    if left_upper_bound == -1 and left_var_vals == -1:
        left_node.close_node()
        left_node.set_is_infeasible()
    else:
        left_node.set_fixed_ones(prev_fixed_ones + left_fix_ones + implied_ones)
        left_node.set_fixed_zeros(prev_fixed_zeros + left_fix_zeros + implied_zeros)
        if is_integer_solution(left_graph, left_var_vals):
            print("LP solution is integer")
            print('left_upper_bound', left_upper_bound)
            left_node.set_is_integer()
            left_node.close_node()
        if left_upper_bound <= incumbent:
            left_node.close_node()

    # Right branch x_ij + x_jk + x_ik >= 2
    x_ij = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[1]))
    x_jk = model.getVarByName(str(branch_triple[1]) + "," + str(branch_triple[2]))
    x_ik = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[2]))
    count = 0
    if node.constraints == []:
        model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_0')
        count += 1
    else:
        model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_0')
        count += 1
        for constr in node.constraints:
            x_ij = model.getVarByName(str(constr[0]) + "," + str(constr[1]))
            x_jk = model.getVarByName(str(constr[1]) + "," + str(constr[2]))
            x_ik = model.getVarByName(str(constr[0]) + "," + str(constr[2]))
            if constr[3] == 0:
                model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_' + str(count))
            else:
                model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_' + str(count))
            count += 1
    model.update()

    right_upper_bound, right_var_vals, model = run_lp(model, Graph, prev_fixed_ones, prev_fixed_zeros)

    implied_constraints = []
    if not (right_upper_bound == -1 and right_var_vals == -1):
        right_fix_ones, right_fix_zeros = reduced_cost_variable_fixing(model, right_var_vals, right_upper_bound,
                                                                       incumbent)
        model = reset_model_variables(model, prev_fixed_ones, prev_fixed_zeros)
        implied_constraints = right_implied(right_fix_zeros, branch_triple)

    for i in range(count):
        model.remove(model.getConstrByName('branch_' + str(i)))
    model.update()

    right_graph = alter_weight(node.graph, branch_triple, original_graph)
    right_lower_bound, right_decluster_combo = run_combo(right_graph, original_graph, isolated_nodes)

    right_constraints = node.constraints.copy()
    right_constraints.append(branch_triple + (2,))
    right_constraints += implied_constraints
    right_node = Node(right_constraints, right_var_vals, right_graph, right_decluster_combo)
    right_node.set_bounds(right_lower_bound, right_upper_bound)

    if right_upper_bound == -1 and right_var_vals == -1:
        right_node.close_node()
        right_node.set_is_infeasible()
    else:
        right_node.set_fixed_ones(prev_fixed_ones + right_fix_ones)
        right_node.set_fixed_zeros(prev_fixed_zeros + right_fix_zeros)
        if is_integer_solution(right_graph, right_var_vals):
            print("LP solution is integer")
            print('right_upper_bound', right_upper_bound)
            right_node.set_is_integer()
            right_node.close_node()
        if right_upper_bound <= incumbent:
            right_node.close_node()

    return left_node, right_node
