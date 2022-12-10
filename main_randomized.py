#### Mariana Rosa,  98390 ####

import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from math import dist
from itertools import combinations, product
from numpy import save
from pyparsing import Or
from regex import P
import time
import pandas as pd

import random
'''  '''
random.seed(98390)

v = ["A", [9, 6]], ["B", [12, 14]], [
    "C", [15, 19]], ["D", [10, 19]], ["E", [19, 17]]
ed = [["C", "E"], ["A", "B"], ["C", "A"], ["C", "B"]]

def adjacency_list(edges):
    # {"vertices": [["A", [7, 6]], ["B", [3, 1]], ["C", [1, 13]], ["D", [12, 12]]],
    # "edges": [["C", "D"], ["A", "B"]]}

    adjacency_lst = {}
    for v1, v2 in edges:
        if v1 in adjacency_lst:
            adjacency_lst[v1].append(v2)
        else:
            adjacency_lst[v1] = [v2]
        if v2 in adjacency_lst:
            adjacency_lst[v2].append(v1)
        else:
            adjacency_lst[v2] = [v1]

    #Exemplo de uma lista de adjacencias {'C': ['D'], 'D': ['C'], 'A': ['B'], 'B': ['A']}
    return adjacency_lst


def check_if_there_are_isolated_vertex(vertex,edges): #If there is any check isolated_vertex the problem is not solvable
    adj_list = adjacency_list(edges)
    vertex_in_adj_list = list(adj_list.keys())
    graph_vertex = [v for v,p in vertex]
    graph_vertex.sort()
    vertex_in_adj_list.sort()

    if graph_vertex == vertex_in_adj_list:
        return False #if there are none isolated vertex return False
    else:
        return True  #if there are isolated vertex return True


def calc_weight_edges(vertices, edges):  # Calculate the weight of the edges
    dic_edges_count = {} #Example of dic_edges_count {('A', 'B'): 1, ('C', 'D'): 5}
  
    for e1, e2 in edges:
      

        for v in vertices:
            if e1 == v[0]:
                point1 = v[1]
            if e2 == v[0]:
                point2 = v[1]

        dic_edges_count[e1, e2] = round(dist(point1, point2))

    return dic_edges_count

def calculate_weight_cut (subsetS, subsetT, weight_list): #Calculate the weight of the cut with the subsets S and T
    possible_edges = list(product(subsetS, subsetT))
    weight_sum = 0
    for edge in weight_list.keys():
        if edge in possible_edges or edge[::-1] in possible_edges:
            weight_sum += weight_list[edge]   
    return weight_sum


def find_cut_randomized_simple(vertex, edges):
    ### FIRST VERSION ###
    #Randomized algorithm
    #{"Subset S": cut_value}
    weights = calc_weight_edges(vertex, edges)

    dic_solution_cut = {}
    dic_iteration_max_cut = {}

    iterations = 0.2 * 2**len(edges)
    print("ITERATIONS", iterations)
    iteration = 0 

    for i in range(int(iterations)):
        subsetS = []
        subsetT = []
        iteration +=  1
        for v in vertex:
            result = random.choice(["Heads","Tails"])
            if result == "Heads":
                subsetS.append(v[0])
            else:
                subsetT.append(v[0])
            tuple_subset_s = tuple(subsetS)
            if  subsetS != [] and tuple_subset_s not in dic_solution_cut.keys():
                dic_solution_cut[tuple_subset_s] = calculate_weight_cut(subsetS, subsetT, weights)


        best_solution = max(dic_solution_cut.values())
        dic_iteration_max_cut[iteration] = best_solution




def find_cut_randomized_algorithm(vertex,edges,weights): #1ª Função
    # Selecionar aleatóriamente 2 vertices do grafo, 1/2 dos vértices existentes 
    # Escolher o que tem aresta mais pesada para o subset S, por E1 = [V1,V2]. V1 vai para S, V2 para T
    # Fazer isto x vezes, sendo x 1/2*len(vertex)
    subset_s = []
    subset_t = []
    iterations = 0

    for i in range (int(len(edges)/2)):
        iterations += 1
        e1 = random.choice(edges)
        e2 = random.choice(edges) 
        
        if e1!=e2:
            tuple_e1 = tuple(e1)
            tuple_e2 = tuple(e2)
            weights_e1 = weights[tuple_e1]
            weights_e2 = weights[tuple_e2]

            if weights_e1 > weights_e2:
                if e1[0] not in subset_s and e1[1] not in subset_t and e1[0] not in subset_t and e1[1] not in subset_s:
                    subset_s.append(e1[0])
                    subset_t.append(e1[1])
                if e1[1] not in subset_s and e1[0] not in subset_t and e1[0] not in subset_s and e1[1] not in subset_t:
                    subset_s.append(e1[1])
                    subset_t.append(e1[0])

            elif weights_e1 < weights_e2:

                if e2[0] not in subset_s and e2[1] not in subset_t and e2[0] not in subset_t and e2[1] not in subset_s:
                    subset_s.append(e2[0])
                    subset_t.append(e2[1])
                if e2[1] not in subset_s and e2[0] not in subset_t and e2[0] not in subset_s and e2[1] not in subset_t:
                    subset_s.append(e2[1])
                    subset_t.append(e2[0])


            if weights_e1 == weights_e2:
                result = random.choice(["Heads","Tails"])
                if result == "Heads":
                    if e1[0] not in subset_s and e1[1] not in subset_t and e1[0] not in subset_t and e1[1] not in subset_s:
                        subset_s.append(e1[0])
                        subset_t.append(e1[1])
                else:
                    if e2[0] not in subset_s and e2[1] not in subset_t and e2[0] not in subset_t and e2[1] not in subset_s:
                        subset_s.append(e2[0])
                        subset_t.append(e2[1])
    for v in vertex:
        iterations += 1
        if v[0] not in subset_s and v[0] not in subset_t:
            subset_t.append(v[0])

    
    max_cut = calculate_weight_cut(subset_s, subset_t, weights)
    return subset_s, subset_t, max_cut, iterations

def perform_randomized_algorithm(vertex, edges,weights): #2ª função, comparar com brute force e ver qual é melhor
    iteration = 0 

    dic_subsetS_maxcut = {}

    number_iterations = int(0.2*2**len(edges))

    for i in range(number_iterations):
        iteration += 1
        result = find_cut_randomized_algorithm(vertex, edges,weights)
    
        subset_s = result[0]
        subset_t = result[1]
        max_cut = result[2]
        number_iterations = result[3]
        tuple_subset_s = tuple(subset_s)
        if tuple_subset_s not in dic_subsetS_maxcut.keys():
            dic_subsetS_maxcut[tuple_subset_s] = (max_cut, subset_t, number_iterations)

    best_solution = max(dic_subsetS_maxcut.values())
    subset_s = list(dic_subsetS_maxcut.keys())[list(dic_subsetS_maxcut.values()).index(best_solution)]
    subset_t = best_solution[1]
    max_cut = best_solution[0]
    number_iterations = best_solution[2]

    print("BEST SOLUTION: ", best_solution)

    return subset_s, subset_t, max_cut, iteration
        



def find_max_cut_brute_force(vertex, edges,weights):
    #Brute force algorithm
    #We are going to find all the possible combinations of subsets S and T and then we are going to calculate the weight of the cut for each one of them
    #The combination with the highest weight will be the one we are going to return

    adj_list = adjacency_list(edges)
    possible_cuts = []
    dic_cut_weight = {}
    iterations = 0
    graph_vertex = list(adj_list.keys())
    graph_vertex_number = len(list(adj_list.keys()))

    for i in range(1, int(graph_vertex_number/2)+1): #We are going to find all the possible combinations of subsets S 
        combs = combinations(graph_vertex, i)
        if possible_cuts == []:
            possible_cuts = [','.join(comb) for comb in combs]
        else:
            possible_cuts = [','.join(comb) for comb in combs] + possible_cuts

    for cut in possible_cuts:
        iterations += 1
        chosen_cut = cut.split(',')
        subset_s = chosen_cut
        subset_t = [v for v in graph_vertex if v not in chosen_cut] #Subset T is going to be all the vertices that are not in subset S
        weights_temp = weights.copy() #this copy is unnecessary, but I did it to make sure that the weights were not being changed
        cut_weight = calculate_weight_cut(subset_s, subset_t, weights_temp) #Calculate the weight of the cut (S->T)
        dic_cut_weight[cut] = cut_weight
        

    max_cut_vertices = max(dic_cut_weight, key=dic_cut_weight.get)
    subset_s = list(max_cut_vertices.split(','))
    subset_t = list(set(graph_vertex) - set(subset_s))
    max_cut_value = dic_cut_weight[max_cut_vertices]

    print("MAX CUT VALUE: ", max_cut_value)

    return max_cut_value, subset_s, subset_t, iterations


    
def find_max_cut_greedy(vertex, edges, weights): #Greedy algorithm

    sorted_weights = {edge: cost for edge, cost in sorted(weights.items(), key=lambda item: item[1], reverse=True)}

    #The heuristic used in these case was to find first the edges with the highest weight and put one vertex in subset S and another one in T

    subset_s = [list(sorted_weights.keys())[0][0]]
    subset_t = [list(sorted_weights.keys())[0][1]]
    iterations = 0
    # if there were only were one edge in the graph, the algorithm would not work, so we need to check if there is only one edge
    if len(sorted_weights) == 1:
        return  calculate_weight_cut(subset_s, subset_t, weights), subset_s, subset_t,0
    else:
        #then we will analyse the rest of the edges
        next_edges = list(sorted_weights.keys())[1:][0]
        #Example - next_edges = ('A', 'B')
        #If in the next edge, the vertex A is in subset S, we are going to put B in subset T, so they can be in different subsets to make the count of the cut
        if next_edges[0] in subset_s and next_edges[1] not in subset_t:
            subset_t.append(next_edges[1])

        #We can not only make this condition to evaluate this case, we need 3 more to make sure we evaluate all the cases

        elif next_edges[0] in subset_t and next_edges[1] not in subset_s:
            subset_s.append(next_edges[1])
        

        if next_edges[1] in subset_s and next_edges[0] not in subset_t:
            subset_t.append(next_edges[0])

        elif next_edges[1] in subset_t and next_edges[0] not in subset_s:
            subset_s.append(next_edges[0])
      

        final_vertices = list(sorted_weights.keys())[2:] # The rest of the vertices will go to subset T
        for v1,v2 in final_vertices:
            iterations +=1
            if v1 not in subset_t and not v1 in subset_s:
                subset_t.append(v1)
            if v2 not in subset_t and not v2 in subset_s:
                subset_t.append(v2)

        cut_weight = calculate_weight_cut(subset_s, subset_t, weights)
        return cut_weight, subset_s, subset_t,iterations



def save_solution(vertex, edges, filepath):     #Save the solution in a txt file
    
    time_start_brute = time.time()
    name_file = "max_cut_"+str(len(vertex))+"_vertex_" + \
        str(len(edges))+"_edges"+".txt"
    adj_list = adjacency_list(edges)
    weights = calc_weight_edges(vertex, edges)
    if check_if_there_are_isolated_vertex(vertex, edges) == True:
        print("The problem is not solvable")
        with open(os.path.join(filepath, name_file), "w") as f:
            f.write("GRAPH WITH "+str(len(vertex))+" NODES \n\n")
            f.write(str(len(vertex)) + " vertices: " + str(vertex)+"\n")
            f.write(str(len(edges)) + " edges : " + str(edges)+"\n\n")
            f.write("Adjacency list: "+str(adj_list)+"\n")
            f.write("This problem is not solvable, it has isolated vertices.")
            f.close()
    else:

        #Brute force
        max_cut_brute = find_max_cut_brute_force(vertex, edges, weights)
        time_end = time.time()
        execution_time_brute = str(time_end - time_start_brute)

        #Greedy
        time_start_greedy = time.time()
        max_cut_greedy = find_max_cut_greedy(vertex, edges, weights)
        time_end_greedy = time.time()
        execution_time_greedy = str(time_end_greedy - time_start_greedy)
        
        #Randomized
        time_start_randomized = time.time()
        max_cut_randomized = perform_randomized_algorithm(vertex, edges, weights)
        time_end_randomized = time.time()
        execution_time_randomized = str(time_end_randomized - time_start_randomized)


        with open(os.path.join(filepath, name_file), "w") as f:

            f.write("GRAPH WITH "+str(len(vertex))+" NODES \n\n")
            f.write(str(len(vertex)) + " vertices: "+str(vertex)+"\n")
            f.write(str(len(edges)) + " edges : "+str(edges)+"\n\n")
            f.write("Adjacency list: "+str(adj_list)+"\n")
            f.write("Weight list: "+str(weights)+"\n")

            f.write("WITH THE BRUTE FORCE ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_brute[0]) + "\n")
            f.write("Subset S: " + str(max_cut_brute[1]) + "\n")
            f.write("Subset T: " + str(max_cut_brute[2]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_brute[3])+"\n\n")
            f.write("TOTAL EXECUTION TIME: "+str(execution_time_brute)+"s \n\n")

            f.write("WITH THE GREEDY ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_greedy[0]) + "\n")
            f.write("Subset S: " + str(max_cut_greedy[1]) + "\n")
            f.write("Subset T: " + str(max_cut_greedy[2]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_greedy[3])+"\n\n")
            f.write("TOTAL EXECUTION TIME: "+execution_time_greedy+"s\n\n")

            f.write("WITH THE RANDOMIZED ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_randomized[2]) + "\n")
            f.write("Subset S: " + str(max_cut_randomized[0]) + "\n")
            f.write("Subset T: " + str(max_cut_randomized[1]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_randomized[3])+"\n\n")
            f.write("TOTAL EXECUTION TIME: "+execution_time_randomized+"s\n\n")


            f.close()


def plot_cut(vertices, edges, cuts,filepath,algorithm):
    #Plot the graph with the cut being the vertices in red the ones belonging to subset S and the ones in blue to subset T

    n_vertex = len(vertices)
    n_edges = len(edges)

    weight_edges = calc_weight_edges(vertices, edges)
    G = nx.Graph()

    color_map = []

    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=weight_edges[(edge[0], edge[1])])
    for node in range(len(vertices)):
        
        if vertices[node][0] in cuts:
            color_map.append('red')
            G.add_node(vertices[node][0], pos=vertices[node][1], color='red')
        else:
            color_map.append('blue')
            G.add_node(vertices[node][0], pos=vertices[node][1])

    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edge_labels(G, pos, labels)
    nx.draw(G, pos, with_labels=True, node_color=color_map)
    if algorithm == "brute_force":
        plt.savefig(f"{filepath}/{n_vertex}vertices_{n_edges}_edges_brute.png", format="PNG")
        plt.close()

    elif algorithm == "greedy":
        plt.savefig(f"{filepath}/{n_vertex}vertices_{n_edges}_edges_greedy.png", format="PNG")
        plt.close()

    elif algorithm == "randomized":
        plt.savefig(f"{filepath}/{n_vertex}vertices_{n_edges}_edges_randomized.png", format="PNG")
        plt.close()

def analyse_file(file_name):
    n_vertex = 0
    n_edges = 0
    vertex = []
    edges = []

    with open(file_name) as f:
        lines = f.readlines()
        n_vertex = lines[2]
        n_edges = lines[3]
        
        for line in lines[4:]:
            edges.append(line.split())
        for i in range(int(n_vertex)):
            pos = random.randint(0, 20), random.randint(0, 20)
            vertex.append([str(i), pos])
    return vertex, edges

graph1 = analyse_file("SW_ALGUNS_GRAFOS/SWtinyG.txt")
graph2 = analyse_file("SW_ALGUNS_GRAFOS/SWmediumG.txt")
#graph3 = analyse_file("SW_ALGUNS_GRAFOS/SWlargeG.txt")

print("Graph 0",graph1[0])
print("Graph 1",graph1[1])
save_solution(graph1[0], graph1[1], "solutions")
save_solution(graph2[0], graph2[1], "solutions")
#save_solution(graph3[0], graph3[1], "solutions")


#perform_multiple_times_algorithms(graph1[0], graph1[1])