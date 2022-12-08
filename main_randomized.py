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
            if e1 == v[0][0]:
                point1 = v[1]
            if e2 == v[0][0]:
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
    print("WEIGHTS", weights)

    dic_solution_cut = {}
    dic_iteration_max_cut = {}

    iterations = 0.2 * 2**len(edges)
    print("ITERATIONS", iterations)
    iteration = 0 

    for i in range(int(iterations)):
        subsetS = []
        subsetT = []
        iteration +=  1
        print("ITERATION", iteration)
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

    for k,v in dic_solution_cut.items():
        print(k, v)

    print("DIC SOLUTION CUT")
    for k,v in dic_iteration_max_cut.items():
        print("ITERATION: ",k,"Max_cut_value: ", v)




def find_max_cut_randomized_algorithm(vertex,edges): #1ª Função
    # Selecionar aleatóriamente 2 vertices do grafo, 1/2 dos vértices existentes 
    # Escolher o que tem aresta mais pesada para o subset S, por E1 = [V1,V2]. V1 vai para S, V2 para T
    # Fazer isto x vezes, sendo x 1/2*len(vertex)
    weights = calc_weight_edges(vertex, edges)
    print("WEIGHTS", weights)
    subset_s = []
    subset_t = []

    for i in range (int(len(edges)/2)):
        e1 = random.choice(edges)
        e2 = random.choice(edges) 
        
        if e1!=e2:
            tuple_e1 = tuple(e1)
            tuple_e2 = tuple(e2)
            weights_e1 = weights[tuple_e1]
            weights_e2 = weights[tuple_e2]
            if weights_e1 > weights_e2:
                subset_s.append(e1[0])
                subset_t.append(e1[1])
            elif weights_e1 < weights_e2:
                subset_s.append(e2[0])
                subset_t.append(e2[1])
            if weights_e1 == weights_e2:
                result = random.choice(["Heads","Tails"])
                if result == "Heads":
                    subset_s.append(e1[0])
                    subset_t.append(e1[1])
                else:
                    subset_s.append(e2[0])
                    subset_t.append(e2[1])

    max_cut = calculate_weight_cut(subset_s, subset_t, weights)

    return subset_s, subset_t, max_cut

def perform_randomized_algorithm(vertex, edges): #2ª função
    iteration = 0 

    dic_subsetS_maxcut = {}

    number_iterations = int(0.2*2**len(edges))

    for i in range(number_iterations):
        iteration += 1
        result = find_max_cut_randomized_algorithm(vertex, edges)
        subset_s = result[0]
        max_cut = result[2]
        tuple_subset_s = tuple(subset_s)
        if tuple_subset_s not in dic_subsetS_maxcut.keys():
            dic_subsetS_maxcut[tuple_subset_s] = max_cut
    
    for k,v in dic_subsetS_maxcut.items():
        print("SUBSET S: ", k, "MAX_CUT: " , v)
    
    return dic_subsetS_maxcut
        
def find_top_5(dic_subsetS_maxcut): ##3ª função
    top_5 = {}
    for i in range(5):
        best_solution = max(dic_subsetS_maxcut, key=dic_subsetS_maxcut.get)
        top_5[best_solution] = dic_subsetS_maxcut[best_solution]
        del dic_subsetS_maxcut[best_solution]
    for k,v in top_5.items():
        print("TOP 5: ", k, "MAX_CUT: " , v)




def find_max_cut_brute_force(vertex, edges):
    #Brute force algorithm
    #We are going to find all the possible combinations of subsets S and T and then we are going to calculate the weight of the cut for each one of them
    #The combination with the highest weight will be the one we are going to return

    adj_list = adjacency_list(edges)
    weights = calc_weight_edges(vertex, edges)
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


def analyse_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        n_vertex = lines[2]
        n_edges = lines[3]
        vertex = []
        edges = []

        for line in lines[4:]:
            edges.append(line.split())
        for i in range(int(n_vertex)):
            pos = random.randint(0, 20), random.randint(0, 20)
            vertex.append([str(i), pos])
    return vertex, edges

graph1 = analyse_file("AA_RandomizedMaxWeightCut/SW_ALGUNS_GRAFOS/SWtinyG.txt")
max_brute_force = find_max_cut_brute_force(graph1[0], graph1[1])
print("ANSWER BRUTE", max_brute_force)
randoms = perform_randomized_algorithm(graph1[0], graph1[1])
top = find_top_5(randoms)

res_rand = find_max_cut_randomized_algorithm(graph1[0], graph1[1])
print("ANSWER", res_rand)
