class Perceptron():

    def __init__(self, input_num, activator):
        '''初始化感知器，设置输入参数个数以及激活函数
        激活函数类型为double->double
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]  # 初始化权重
        self.bias = 0.0  # 初始化bias

    def __str__(self):
        '''打印学习到的权重和bias'''
        return '权重：\n{}\nbias：\n{}\n'.format(self.weights, self.bias)

    def predict(self, input_vec):
        '''输入向量，输入感知器计算结果'''

        return self.activator(sum([w * v for w, v in zip(self.weights, input_vec)]) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''输入训练数据：向量，label，训练次数，学习率'''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            print(self)

    def _one_iteration(self, input_vecs, labels, rate):
        '''一次迭代'''
        for input_vec, label in zip(input_vecs, labels):
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''感知器权重更新'''
        d = label - output
        self.weights = [weight + rate * d * vec_dem for vec_dem,
                        weight in zip(input_vec, self.weights)]
        self.bias += d * rate


def f(x):
    '''激活函数'''
    return 1 if x > 0 else 0


def get_training_data():
    '''输入数据'''
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_data()
    p.train(input_vecs, labels, 10, 0.1)
    return p
if __name__=='__main__':
    and_perceptron = train_perceptron()

    print(and_perceptron)
    and_perceptron.predict([0, 1])
