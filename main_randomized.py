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
from decimal import Decimal  
import random
'''  '''
random.seed(98390)

def adjacency_list(edges): #1º trabalho 
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


def check_if_is_possible(vertex,edges): #If there is any check isolated_vertex the problem is not solvable  #1º trabalho 
    adj_list = adjacency_list(edges)
    vertex_in_adj_list = list(adj_list.keys())
    graph_vertex = [v for v,p in vertex]
    graph_vertex.sort()
    vertex_in_adj_list.sort()

    if graph_vertex == vertex_in_adj_list:
        return False #if there are none isolated vertex return False
    else:
        return True  #if there are isolated vertex return True


def calc_weight_edges(vertices, edges):  # Calculate the weight of the edges #1º trabalho 
    dic_edges_count = {} #Example of dic_edges_count {('A', 'B'): 1, ('C', 'D'): 5}
  
    for e1, e2 in edges:
      

        for v in vertices:
            if e1 == v[0]:
                point1 = v[1]
            if e2 == v[0]:
                point2 = v[1]

        dic_edges_count[e1, e2] = round(dist(point1, point2))

    return dic_edges_count

def calculate_weight_cut (subsetS, subsetT, weight_list): #Calculate the weight of the cut with the subsets S and T #1º trabalho 
    possible_edges = list(product(subsetS, subsetT))
    weight_sum = 0
    for edge in weight_list.keys():
        if edge in possible_edges or edge[::-1] in possible_edges:
            weight_sum += weight_list[edge]   
    return weight_sum


def find_cut_randomized_simple(vertex, edges): #1º trabalho 
    ### FIRST VERSION ###
    #Randomized algorithm
    weights = calc_weight_edges(vertex, edges)
    iteration = 0
    subsetS = []
    subsetT = []
    for v in vertex:
        iteration += 1
        result = random.choice(["Heads","Tails"]) #Randomly choose between Heads or Tails
        if result == "Heads":
            subsetS.append(v[0])
        else:
            subsetT.append(v[0])
        
    cut_weight = calculate_weight_cut(subsetS, subsetT, weights)

    return cut_weight, subsetS, subsetT, iteration 




def find_cut_randomized_algorithm(vertex,edges,weights): #1ª Função

    subset_s = []
    subset_t = []
    iterations = 0

    for i in range (int(len(edges)/2)): #Durante i vezes, onde i = metade dos edges, escolher aleatoriamente dois edges e comparar os seus pesos
        iterations += 1
        e1 = random.choice(edges)
        e2 = random.choice(edges) 
        
        if e1!=e2:
            tuple_e1 = tuple(e1)
            tuple_e2 = tuple(e2)
            weights_e1 = weights[tuple_e1]
            weights_e2 = weights[tuple_e2]

            if weights_e1 > weights_e2: #Se o peso do edge 1 for maior que o peso do edge 2, adicionar os vertices do edge 1 ao subset s e t
                if e1[0] not in subset_s and e1[1] not in subset_t and e1[0] not in subset_t and e1[1] not in subset_s:
                    subset_s.append(e1[0])
                    subset_t.append(e1[1])
                

            elif weights_e1 < weights_e2: #Se o peso do edge 2 for maior que o peso do edge 1, adicionar os vertices do edge 2 ao subset s e t

                if e2[0] not in subset_s and e2[1] not in subset_t and e2[0] not in subset_t and e2[1] not in subset_s:
                    subset_s.append(e2[0])
                    subset_t.append(e2[1])

            if weights_e1 == weights_e2: #Se o peso do edge 1 for igual ao peso do edge 2, escolher aleatoriamente entre adicionar os vertices do edge 1 ou do edge 2 ao subset s e t
                result = random.choice(["Heads","Tails"])
                if result == "Heads":
                    if e1[0] not in subset_s and e1[1] not in subset_t and e1[0] not in subset_t and e1[1] not in subset_s:
                        subset_s.append(e1[0])
                        subset_t.append(e1[1])
                else:
                    if e2[0] not in subset_s and e2[1] not in subset_t and e2[0] not in subset_t and e2[1] not in subset_s:
                        subset_s.append(e2[0])
                        subset_t.append(e2[1])
    for v in vertex: #Adicionar os vertices que não foram adicionados ao subset t
        iterations += 1
        if v[0] not in subset_s and v[0] not in subset_t:
            subset_t.append(v[0])

    
    cut_value = calculate_weight_cut(subset_s, subset_t, weights) #Calcular o valor do cut
    return subset_s, subset_t, cut_value, iterations 

def perform_randomized_algorithm(vertex, edges,weights): #2ª Função
    time_start = time.time()

    iteration = 0 

    dic_subsetS_maxcut = {}
    #Fazer limites para o número de iterações consoante as arestas
    if len(edges)<= 15:
        number_iterations = 0.2 * 2**len(edges)

    elif len(edges)> 16 and len(edges)<25:
        print("hi")
        number_iterations = 0.1 * 2**len(edges)
    else:
        number_iterations = 1000

    for i in range(int(number_iterations)):
        
        result = find_cut_randomized_algorithm(vertex, edges,weights) #Chamar a função find_cut_randomized_algorithm para calcular o cut
        iterations_before =  result[3]
        if iteration == 0: #Se for a primeira iteração, calcular o número de iterações com base no número de iterações da função find_cut_randomized_algorithm
            iteration = iterations_before + 1 #Adicionar um

        else:
            iteration +=1 #Se não for a primeira iteração, adicionar 1 ao número de iterações

        subset_s = result[0]
        subset_t = result[1]
        max_cut = result[2]
        tuple_subset_s = tuple(subset_s)

        if tuple_subset_s not in dic_subsetS_maxcut.keys(): #Se o subset s não estiver no dicionário, adicionar o subset s e o seu valor de cut para encontrar sempre o máximo
            dic_subsetS_maxcut[tuple_subset_s] = (max_cut, subset_t)

    best_solution = max(dic_subsetS_maxcut.values())
    subset_s = list(dic_subsetS_maxcut.keys())[list(dic_subsetS_maxcut.values()).index(best_solution)]
    subset_t = best_solution[1]
    max_cut = best_solution[0]
    print("BEST SOLUTION: ", best_solution)
    print("TIME", time.time() - time_start, "seconds")


    return subset_s, subset_t, max_cut, iteration
        



def find_max_cut_brute_force(vertex, edges,weights): #1º trabalho 
    #Brute force algorithm
    #We are going to find all the possible combinations of subsets S and T and then we are going to calculate the weight of the cut for each one of them
    #The combination with the highest weight will be the one we are going to return
    time_start = time.time()
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
    time_end = time.time()
    print("TIME: ", time_end - time_start)

    print("MAX CUT VALUE: ", max_cut_value)

    return max_cut_value, subset_s, subset_t, iterations


    
def find_max_cut_greedy(vertex, edges, weights): #Greedy algorithm 1º trabalho

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



def save_solution(vertex, edges, filepath):     #Save the solution in a txt file 1º trabalho, com upgrades para este
    print("LEN VERTEX: ", len(vertex))
    print("LEN EDGES: ", len(edges))
    
    time_start_brute = time.time()
    name_file = "max_cut_randomized_"+str(len(vertex))+"_vertex_" + \
        str(len(edges))+"_edges"+".txt"
    adj_list = adjacency_list(edges)
    weights = calc_weight_edges(vertex, edges)
    if check_if_is_possible(vertex, edges) == True or len(vertex) <= 2 or len(edges) < 3:
        print("The problem is not solvable")
        with open(os.path.join(filepath, name_file), "w") as f:
            f.write("GRAPH WITH "+str(len(vertex))+" NODES \n\n")
            f.write(str(len(vertex)) + " vertices: " + str(vertex)+"\n")
            f.write(str(len(edges)) + " edges : " + str(edges)+"\n\n")
            f.write("Adjacency list: "+str(adj_list)+"\n")
            f.write("This problem is not solvable, it has isolated vertices or it doens't require the conditions to execute the algorithm.")
            f.close()
    else:
        """
        RESULTADOS JÁ ANALISADOS NO PRIMEIRO TRABALHO
        #Brute force
        max_cut_brute = find_max_cut_brute_force(vertex, edges, weights)
        time_end = time.time()
        execution_time_brute = str(time_end - time_start_brute)
        
        #Greedy
        time_start_greedy = time.time()
        max_cut_greedy = find_max_cut_greedy(vertex, edges, weights)
        time_end_greedy = time.time()
        execution_time_greedy = str(time_end_greedy - time_start_greedy)
        """
        #Simple randomized
        time_start_simple_rand = time.time() 
        max_cut_randomized_simple = find_cut_randomized_simple(vertex, edges)
        time_end_simple_rand = time.time()
        execution_time_simple_rand = str(time_end_simple_rand - time_start_simple_rand)


        #Randomized
        time_start_randomized = time.time()
        print("RANDOMIZED ALGORITHM")
        max_cut_randomized = perform_randomized_algorithm(vertex, edges, weights)
        print("RANDOMIZEDMAX CUT VALUE: ", max_cut_randomized[0])
        time_end_randomized = time.time()
        execution_time_randomized = str(time_end_randomized - time_start_randomized)
        print("TIME RANDOMIZED: ", execution_time_randomized)
        print("PLOTTING")
        plot_cut(vertex, edges, max_cut_randomized[0],"solutions","randomized")
    
        with open(os.path.join(filepath, name_file), "w") as f:

            f.write("GRAPH WITH "+str(len(vertex))+" NODES \n")
            f.write(str(len(vertex)) + " vertices: "+str(vertex)+"\n")
            f.write(str(len(edges)) + " edges : "+str(edges)+"\n\n")
            f.write("Adjacency list: "+str(adj_list)+"\n")
            f.write("Weight list: "+str(weights)+"\n\n")

            """
            f.write("WITH THE BRUTE FORCE ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_brute[0]) + "\n")
            f.write("Subset S: " + str(max_cut_brute[1]) + "\n")
            f.write("Subset T: " + str(max_cut_brute[2]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_brute[3])+"\n")
            f.write("TOTAL EXECUTION TIME: "+str(execution_time_brute)+"s \n\n")
            f.write("WITH THE GREEDY ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_greedy[0]) + "\n")
            f.write("Subset S: " + str(max_cut_greedy[1]) + "\n")
            f.write("Subset T: " + str(max_cut_greedy[2]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_greedy[3])+"\n")
            f.write("TOTAL EXECUTION TIME: "+execution_time_greedy+"s\n\n")
            """


            f.write("WITH THE SIMPLE RANDOMIZED ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_randomized_simple[0]) + "\n")
            f.write("Subset S: " + str(max_cut_randomized_simple[1]) + "\n")
            f.write("Subset T: " + str(max_cut_randomized_simple[2]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_randomized_simple[3])+"\n")
            f.write("TOTAL EXECUTION TIME: "+execution_time_simple_rand+"s\n\n")


            f.write("WITH THE COMPLEX RANDOMIZED ALGORITHM: \n")
            f.write("Maximum weight cut: " + str(max_cut_randomized[2]) + "\n")
            f.write("Subset S: " + str(max_cut_randomized[0]) + "\n")
            f.write("Subset T: " + str(max_cut_randomized[1]) + "\n")
            f.write("NUMBER OF ITERATIONS: "+str(max_cut_randomized[3])+"\n")
            f.write("TOTAL EXECUTION TIME: "+execution_time_randomized+"s\n\n")


            f.close()



def analyse_file(file_name): #funcao desenvolvida para ler os grafos fornecidos pelos docentes e guardar os dados num array
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


def plot_solutions_all():  #Correr o algoritmo para todos os grafos gerados no primeiro trabalho 

    percentages = [12.5, 25, 50, 75]

    for p in percentages:
        path_percentage = "Percentage_" + str(p)
        path_solution = "solutions/"+path_percentage

        if os.path.exists(path_solution) == False:
            os.mkdir(path_solution)
        json_files = [pos_json for pos_json in os.listdir(
            "generated_graphs_first_project/"+path_percentage) if pos_json.endswith('.json')]
        jsons_data = pd.DataFrame(columns=['vertices', 'edges'])
        print(json_files)
        for index, js in enumerate(json_files):
            with open(os.path.join("generated_graphs_first_project/"+path_percentage, js)) as json_file:
                json_text = json.load(json_file)
                vertex = json_text["vertices"]
                edges = json_text["edges"]
                save_solution(vertex, edges, path_solution)
                jsons_data.loc[index] = [vertex, edges]



graph1 = analyse_file("SW_ALGUNS_GRAFOS/SWtinyG.txt")
weights_graph1 = calc_weight_edges(graph1[0], graph1[1])  #calcular os pesos do grafo 1
#save_solution(graph1[0], graph1[1], "solutions") testar o algoritmo para o grafo 1

graph2 = analyse_file("SW_ALGUNS_GRAFOS/SWmediumG.txt") #nao consegue resolver
graph3 = analyse_file("SW_ALGUNS_GRAFOS/SWlargeG.txt") #nao consegue resolver
