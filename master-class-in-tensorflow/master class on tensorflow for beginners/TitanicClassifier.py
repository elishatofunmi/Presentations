# This is a neural network classifier built using tensorflow
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


class TitanicClassifier:
    
    def __init__(self,X, y,n_epochs, batch_size, learning_rate,validation_data):
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.x = X
        self.y = y
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(712/batch_size))
    
        # accumulation of the scores
        self.scores = {}
        
        # parameters to evaluate
        #optimizers
        self.optimizers = ['rmsprop', 'gradientdescent', 'adam','timedelta']
        #layers
        self.layers = {'h1':[50,100,200,300,400], 'h2':[50,100,200,300]}
        
        b1, b2,b3 = self.evaluate(n_epochs, batch_size, learning_rate, validation_data)
        print('best h1: '+str(b1))
        print('best h2: '+ str(b2))
        print('best optimizer: '+str(b3))
        
        return
    
    
    def compute(self,n,f ,i,j,k,validation_data):
        
        learning_rate = 0.01
        
        if k == 'gradientdescent':
            opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
            score = self.NeuralNet(n,f, i,j,opt, validation_data, path_save = False)
            name = str(i)+'_'+str(j)+'_'+str(k)
            self.scores[name] = score
        elif k == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
            score = self.NeuralNet(n, f, i,j,opt,validation_data, path_save = False)
            name = str(i)+'_'+str(j)+'_'+str(k)
            self.scores[name] = score

        elif k == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate = learning_rate)
            score= self.NeuralNet(n, f,i,j,opt,validation_data, path_save = False)
            name = str(i)+'_'+str(j)+'_'+str(k)
            self.scores[name] = score

        elif k == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate = learning_rate)
            score = self.NeuralNet(n,f,i,j,opt,validation_data,path_save = False)
            name = str(i)+'_'+str(j)+'_'+str(k)
            self.scores[name] = score

        else:
            # default optimizer
            opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
            score = self.NeuralNet(n,f,i,j,opt,validation_data,path_save = False)
            name = str(i)+'_'+str(j)+'_'+str(k)
            self.scores[name] = score
            
        return True
        
    
    def evaluate(self,n,f, learning_rate,validation_data):
        self.scores = {}
        # compute
        
        computing = [self.compute(n,f,i,j,k,validation_data) for i in self.layers['h1'] for j in self.layers['h2'] for k in self.optimizers]
        # best argument
        index = np.argmin([np.abs(self.scores[m][0]-self.scores[m][1]) for m in self.scores.keys()])
        out = [x for i,x in enumerate(self.scores.keys()) if i == index][0]
        output_names = out.split('_')
        b1, b2,b3 = output_names
        return b1, b2,b3
    
    
    
    def fetch_batch(batch_index):
        if batch_index < self.n_batches-1:
            start = batch_index * self.batch_size
            stop = batch_size + start
            x_batch = self.x[start:stop]
            y_batch = self.y[start:stop]

        else:
            start = batch_index* batch_size
            x_batch = x_train[start:]
            y_batch = y_train[start:]
        return x_batch, y_batch
    
    def to_categorical(self,y):
        feed = {'y':y}
        frame = pd.DataFrame(feed)
        return pd.get_dummies(frame, columns = ['y'])
    
    def neuron_layer(self, X, n_neurons, name, activation = None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2/ np.sqrt(n_inputs)
            inita = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
            W = tf.Variable(inita, name = 'weights')
            b = tf.Variable(tf.zeros([n_neurons]), name = 'biases')
            z = tf.matmul(X, W) + b

            if activation == 'relu':
                return tf.nn.relu(z)

            else:
                return z
    
    
    def NeuralNet(self,n_epochs, batch_size, h_update_one, h_update_two, optimizer,validation_data, path_save = False):
        X = tf.placeholder(tf.float32, shape = (None, 9), name = 'x')
        y = tf.placeholder(tf.float32, shape = (None,2), name = 'y')

        
        # defined inputs, hidden layers and output
        n_inputs = 9
        h1 = h_update_one
        h2 = h_update_two
        n_output = 2
        
        with tf.name_scope('NeuralNet'):
            
            hidden1 = self.neuron_layer(X, h1, 'hidden1', activation = 'relu')
            hidden2 = self.neuron_layer(hidden1, h2, 'hidden2', activation = 'relu')
            logits = self.neuron_layer(hidden2, n_output, 'outputs')
         
        
        with tf.name_scope('loss'):
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = logits)
            loss = tf.reduce_mean(xentropy, name = 'loss')
    
        # optimizer
        with tf.name_scope('train'):
            training_op = optimizer.minimize(loss)
        
        # Metric
        with tf.name_scope('precision'):
            #fn, fp, tn, tp

            prediction = tf.argmax(logits, 1)
            actual = tf.argmax(y,1)
            #predicted = tf.cast(prediction, tf.float32)
            #actual_predicton = tf.cast(actual, tf.float32)

            TP = tf.math.count_nonzero(prediction * actual)
            TN = tf.math.count_nonzero((prediction - 1) * (actual - 1))
            FP = tf.math.count_nonzero(prediction * (actual - 1))
            FN = tf.math.count_nonzero((prediction - 1) * actual)

            accuracy = (TP+TN)/(TP+TN+FP+FN)
            precision = TP/(TP+FP)
            Recall = TP/(TP+FN)
            F1_Score = 2*(Recall * precision) / (Recall + precision)
            
            
        # train the data
        #desired metric is accuracy
        placeholders = (X,y)
        score = self.train(n_epochs, batch_size, training_op, loss, validation_data,accuracy,placeholders,path_save = False, metric = 'accuracy')
        return score
    
    
    def one_hot(self, y):
        frame = pd.DataFrame({'k':y.astype(np.float32)})
        return pd.get_dummies(frame, columns = ['k'])
    
    
    
    def train(self, n_epochs, batch_size, training_op, loss,validation_data,accuracy, placeholders, path_save = False, metric = 'accuracy'):
        with tf.Session() as sess:
            self.init = tf.global_variables_initializer()
            self.init.run()
            X,y = placeholders
            x_test, y_test = validation_data
            #self.init.run()
            max_acc=0
            acc_going_down=0
            for epoch in range(n_epochs):
                batch_step=0
                avg_loss = 0
                total_loss= 0
                total_batch = int(self.x.shape[0]/self.batch_size)

                for batch_index in range(total_batch):
                    x_batch, y_batch = self.fetch_batch(batch_index)
                    _,l=sess.run([training_op,loss],feed_dict={X:x_batch, y:self.one_hot(y_batch)})
                    batch_step+=1
                    total_loss += l
                if((epoch)%10==0):
                    avg_loss = total_loss/self.batch_size
                    
             # metric computation
            if metric == 'accuracy':
                v1 = accuracy.eval({X: self.x, y: self.one_hot(self.y)})
                v2 = accuracy.eval({X: x_test, y: self.one_hot(y_test)})
            elif metric == 'f1_score':
                pass
            elif metric == 'precision':
                pass
            elif metric =='roc_auc':
                pass
            else:
                pass
            
        if path_save == True:
            save_path = self.saver(sess, './my_model.ckpt')
        else:
            pass
        
        return v1,v2

    
