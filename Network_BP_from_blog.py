import random

# Node class
# data: Node indexes, up stream, down stream
# compute: output value, delta
class Node():
    def __init__(self, layer_index, node_index):
        '''construct the node, with two indexes'''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0
    
    def set_output(self, output):
        '''if this layer is the input layer, use this func to set the values'''
        self.output=output
    
    def append_downstream_connection(self, conn):
        '''append a downstream Connection'''
        self.downstream.append(conn)
    
    def append_upstream_connection(self, conn):
        '''append upstream connection'''
        self.upstream.append(conn)
    
    def calc_output(self, active_func):
        '''calculate the output of the current node'''
        sum=sum(_upstream.upstream_node.ouput * _upstream.weight for _upstream in self.upstream)
        self.output= active_func(sum)
    
    def calc_hidden_layer_delta(self):
        '''compute hidden layer delta'''
        a_i=self.output
        sum_downstream_weight_delta=sum([ds.weight*ds.downstream_node.delta for ds in self.downstream])
        self.delta=a_i*(1-a_i)*sum_downstream_weight_delta
    
    def calc_output_layer_delta(self, label):
        '''compute output layer delta'''
        self.delta=self.output*(1-self.output)*(label-self.output)
    
    def __str__(self):
        '''print node info'''
        node_info='Node index:\tlayer {}\tnode {}\nNode output:\t{}\nNode Delta:\t{}'.format(self.layer_index,self.node_index,self.output,self.delta)
        node_upstream_info=str([_.upsteam_node, _.weight for _ in self.upstream])
        node_downstream_info=str([_.downstream_node, _.weight for _ in self.downstream])
        return node_info+'\n\tupstream\n'+node_upstream_info+'\n\tdownstream\n'+node_downsteram_info


# ConstNode class
# data: node indexes, down stream
# compute: constant output of 1
class ConstNode():
    def __init__(self, layer_index, node_index):
        '''construct the node, with two indexes'''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.output=1
        self.delta=0
    def append_downstream_connection(self, conn):
        '''append a downstream Connection'''
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        '''compute hidden layer delta'''
        a_i=self.output
        sum_downstream_weight_delta=sum([ds.weight*ds.downstream_node.delta for ds in self.downstream])
        self.delta=a_i*(1-a_i)*sum_downstream_weight_delta
    def __str__(self):
        '''print node info'''
        node_info='Node index:\tlayer {}\tnode {}\nNode output:\t{}\nNode Delta:\t{}'.format(self.layer_index,self.node_index,self.output,self.delta)
        node_downstream_info=str([_.downstream_node, _.weight for _ in self.downstream])
        return node_info+'\n\tdownstream\n'+node_downsteram_info

# Layer class
# data: layer index, nodes in the layer
# compute: layer output
class Layer(object):
    def __init__(self, layer_index,node_count):
        '''construct a layer'''
        self.layer_index=layer_index
        self.nodes=[]
        for _ in range(node_count):
            self.nodes.append(Node(layer_index,_))
        self.nodes.append(ConstNode(layer_index,node_count))
    
    def set_output(self,data):
        '''set output value if this layer is input layer'''
        for _ in range(len(data)):
            self.nodes[_]=data[i]
    
    def calc_output(self):
        '''calculate the output of nodes in this layer'''
        for node in self.nodes[:-1]:
            node.calc_output()
    
    def dump(self):
        '''print layer info'''
        for node in self.nodes:
            print(node)


# Connection class
# data: up stream node, down stream node, weight
# compute: gradiant
class Connection():
    def __init__(self, upstream_node,downstream_node):
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-0.1,0.1)
        self.gradiant=0.0
    def calc_gradient(self):
        '''calculate gradient'''
        self.gradiant=self.downstream_node.delta*self.upstream_node.output
    def get_gradient(self):
        '''get current gradient'''
        return self.gradiant
    def update_weight(self, rate):
        '''update weight according to gradient'''
        self.calc_gradient()
        self.weight+=rate*self.gradiant
    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)


# Connections class
# data: many Connection instances
class Connections(object):
    def __init__(self):
        self.connections=[]
    def add_connection(self, connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)
    
# Network class
# data: connections, layers, num_layers, num_nodes
# compute: train(labels,dataset, rate, iteration), predict(sample)
class Network(object):
    def __init__(self, layers):
        '''init a full connected Network
        layers: 2d array, describe number of neurons on each layers'''
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers)
        node_count=0
        for _ in range(layer_count):
            self.layers.append(Layer(_, layers[_]))
        for layer in range(layer_count-1):
            connections=[Connection(upstream_node,downstream_node) for upsteam_node in self.layers[layer].nodes for downstream_node in self.layers[layer+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    
    def train(self, labels, data_set,rate,iteration):
        '''train the Network
        labels: [] label of dataset'''
        for _ in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(lables[d],dataset[d],rate)
    
    def train_one_sample(self, label,sample,rate):
        '''train the network with one data'''
        self.predict(sample)
        self.calc_delta(label)
        self.upgrade_weight(rate)

    def calc_delta(self, label):
        '''calculate delta of every neuron'''
        output_nodes=self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layer[-2::-1]:
            for node in layer.nodes:
                node.cal_hidden_layer_delta()
    def upgrade_weight(self, rate):
        '''update weight of each cennection'''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)
    def calc_gradient(self):
        '''compute gradient of each connection'''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
    def get_gradient(self, label, sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample: 数组，样本的特征，也就是网络的输入向量
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])
    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()    