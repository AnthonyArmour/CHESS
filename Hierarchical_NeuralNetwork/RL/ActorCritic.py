# import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, add
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import L2
# from tensorflow.keras.metrics import categorical_crossentropy
# import random
# import pickle
import numpy as np

class ActorCritic():

    def __init__(self, action_space, lr=0.0001, convL=4, fcL=(2, 2048), filters=[32, 64, 64, 64],
                 activation="sigmoid", kernel_size=[(7, 7), (5, 5), (3, 3), (3, 3)]):
        self.action_space = action_space
        self.lr = lr
        self.convL = convL
        self.fcL = fcL
        self.filters = filters
        self.activation = activation
        self.kernel_size = kernel_size
        self.Actor, self.Critic = self.get_ActorCritic_ResConvNet()
        self.states, self.actions, self.rewards = None, [], []
        self.scores, self.episodes, self.average = [], [], []


    def get_ActorCritic_ResConvNet(self):
        Network = {
            "convL": 4,
            "fcL": (2, 4096),
            "filters": [32, 64, 64, 64],
            "kernel_size": [(7, 7), (5, 5), (3, 3), (3, 3)],
            "activation": "sigmoid"
        }
        initializer = k.initializers.HeNormal()
        input = k.Input(shape=(8, 8, 1))
        conv_add = BatchNormalization()(input)
        # conv_add = None

        for layer in range(self.convL):
            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same"
                )(conv_add)
            conv = BatchNormalization()(conv)
            convA = Activation(self.activation)(conv)

            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same"
                )(convA)
            conv = BatchNormalization()(conv)
            convB = Activation(self.activation)(conv)

            conv_add = add([convA, convB])


        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(conv_add)
        conv = BatchNormalization()(conv)
        conv = Activation(self.activation)(conv)

        fc_add = Flatten()(conv)

        for layer in range(int(self.fcL[0]/2)):
            fc = Dense(units=self.fcL[1], kernel_initializer=initializer)(fc_add)
            fc = BatchNormalization()(fc)
            fcA = Activation(self.activation)(fc)

            fc = Dense(units=self.fcL[1], kernel_initializer=initializer)(fcA)
            fc = BatchNormalization()(fc)
            fcB = Activation(self.activation)(fc)

            fc_add = add([fcA, fcB])



        action = Dense(units=self.action_space, activation="softmax", kernel_initializer=initializer)(fc_add)
        value = Dense(1, kernel_initializer=initializer)(fc_add)

        Actor = Model(input, action)
        Actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

        Critic = Model(input, value)
        Critic.compile(loss='mse', optimizer=Adam(lr=self.lr))

        return Actor, Critic

    def memory(self, state, action, reward):
        # self.states.append(state)
        if self.states is None:
            self.states = state
        else:
            self.states = np.concatenate((self.states, state), axis=0)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def getAction(self, state):

        prediction = self.Actor.predict(state)[0]
        return np.random.choice(self.action_space, p=prediction)

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99
        running_add = 0
        #TD targets
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r
 
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        # if str(episode)[-2:] == "00":# much faster than episode % 100
        #     pylab.plot(self.episodes, self.scores, 'b')
        #     pylab.plot(self.episodes, self.average, 'r')
        #     pylab.ylabel('Score', fontsize=18)
        #     pylab.xlabel('Steps', fontsize=18)
        #     try:
        #         pylab.savefig(self.path+".png")
        #     except OSError:
        #         pass

        return self.average[-1]