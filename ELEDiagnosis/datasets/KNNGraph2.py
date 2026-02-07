from datasets.Generator import Gen_graph

def KNNGraph2(interval,data,label,task,mat_len,graph_num):
    a, b = mat_len*graph_num,mat_len*graph_num+interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = Gen_graph('KNNGraph',graph_list, label,task)
    return graphset


