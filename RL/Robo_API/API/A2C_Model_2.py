import tensorflow.keras as k
import tensorflow.keras.backend as backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, BatchNormalization, Conv2D, Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class ActorCritic():

    def __init__(self, action_space=1968, lr=0.0000025, beta=0.00005, convL=4, fcL=(2, 2048), filters=[32, 64, 64, 64],
                 activation="tanh", kernel_size=[(7, 7), (5, 5), (3, 3), (3, 3)]):
        self.action_space = action_space
        self.lr = lr
        self.convL = convL
        self.fcL = fcL
        self.filters = filters
        self.beta = beta
        self.activation = activation
        self.kernel_size = kernel_size
        # self.Actor, self.Critic = self.GetModel()
        # self.states, self.actions, self.rewards = None, [], []


    def ConvResNet(self):

        inp = Input(shape=(8, 8, 1))
        initializer = k.initializers.HeNormal()
        conv_add = BatchNormalization()(inp)

        for layer in range(self.convL):
            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same", use_bias=True, bias_initializer='ones'
                )(conv_add)
            conv = BatchNormalization()(conv)
            convA = Activation(self.activation)(conv)

            conv = Conv2D(
                filters=self.filters[layer], kernel_size=self.kernel_size[layer], padding="same", use_bias=True, bias_initializer='ones'
                )(convA)
            conv = BatchNormalization()(conv)
            convB = Activation(self.activation)(conv)

            conv_add = add([convA, convB])


        conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=True, bias_initializer='ones')(conv_add)
        conv = BatchNormalization()(conv)
        conv = Activation(self.activation)(conv)

        fc_add = Flatten()(conv)

        for layer in range(int(self.fcL[0]/2)):
            fc = Dense(units=self.fcL[1], kernel_initializer=initializer, use_bias=True, bias_initializer='ones')(fc_add)
            fc = BatchNormalization()(fc)
            fcA = Activation(self.activation)(fc)

            fc = Dense(units=self.fcL[1], kernel_initializer=initializer, use_bias=True, bias_initializer='ones')(fcA)
            fc = BatchNormalization()(fc)
            fcB = Activation(self.activation)(fc)

            fc_add = add([fcA, fcB])
        
        return fc_add, initializer, inp



    def get_ActorCritic_ResConvNet(self, paths=None):

        fc, init, inp = self.ConvResNet()
        fc_coach, _, inpC = self.ConvResNet()

        action = Dense(units=self.action_space, activation="softmax", kernel_initializer=init, kernel_regularizer='l2')(fc)
        value = Dense(1, kernel_initializer=init, kernel_regularizer='l2')(fc)
        coach_action = Dense(units=self.action_space, activation="softmax", kernel_initializer=init)(fc_coach)

        # def custom_loss(y_true, y_pred):
        #     out = backend.clip(y_pred, 1e-8, 1-1e-8)
        #     log_lik = y_true*backend.log(out)

        #     return backend.sum(-log_lik*delta)


        Actor = Model(inp, action)
        Actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

        Critic = Model(inp, value)
        Critic.compile(loss='mse', optimizer=Adam(lr=self.beta))

        # Policy  = Model(inp, action)
        Coach = Model(inpC, coach_action)
        if paths:
            Actor.load_weights(paths["actor"]) #, custom_objects={"custom_loss": custom_loss}
            Critic.load_weights(paths["critic"])
            # Policy.load_weights(paths["policy"])
            Coach.load_weights(paths["coach"])

        return Actor, Critic, Coach

