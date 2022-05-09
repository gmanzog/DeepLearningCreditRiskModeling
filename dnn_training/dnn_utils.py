"""
Copyright:
    Deep Learning Credit Risk Modeling
    Gerardo Manzo and Xiao Qiao
    The Journal of Fixed Income Fall 2021, jfi.2021.1.121; DOI: https://doi.org/10.3905/jfi.2021.1.121
Disclaimer:
     THIS SOFTWARE IS PROVIDED "AS IS".  YOU ARE USING THIS SOFTWARE AT YOUR OWN
     RISK.  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
     THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
     PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT,
     INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
     TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
     PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
     LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
     NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
     THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Description:
    Utils to train DNN
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, least_squares
from keras import backend as K
import keras

class NumpyNN(object):
    def __init__(self, model_pars, NumLayers, NNParameters):
        """
        Numpy DNN implemented for speed
        :param model_pars: parameters
        :param NumLayers: number of layers
        :param NNParameters: DNN trained parameters
        """
        self.NumLayers = NumLayers
        self.NNParameters = NNParameters
        self.Npars = len(model_pars)


    def myscale(self, x):
        """
        scaler
        :param x:
        :return:
        """
        res = np.zeros(len(x))
        for i in range(len(x)):
            res[i] = (x[i] - (self.ub[i] + self.lb[i]) * 0.5) * 2 / (self.ub[i] - self.lb[i])
        return res

    def elu(self, x):
        # ELU activation func
        ind = (x < 0)
        x[ind] = np.exp(x[ind]) - 1
        return x

    def relu(self, x):
        # RELU activation func
        x = np.maximum(x, 0)
        return x

    def eluPrime(self, y):
        # we make a deep copy of input x
        x = np.copy(y)
        ind = (x < 0)
        x[ind] = np.exp(x[ind])
        x[~ind] = 1
        return x

    def reluPrime(self, y):
        # we make a deep copy of input x
        x = np.copy(y)
        x = np.maximum(x, 0)
        return x

    def NeuralNetwork(self, x):
        input1 = x

        for i in range(self.NumLayers):
            input1 = np.dot(input1, self.NNParameters[i][0]) + self.NNParameters[i][1]
            # Elu activation
            input1 = self.relu(input1)
        # The output layer is linnear
        i += 1
        return np.dot(input1, self.NNParameters[i][0]) + self.NNParameters[i][1]

    def NeuralNetworkGradient(self, x):
        input1 = x

        # Identity Matrix represents Jacobian with respect to initial parameters
        grad = np.eye(len(input1))
        # Propagate the gradient via chain rule
        for i in range(self.NumLayers):
            input1 = (np.dot(input1, self.NNParameters[i][0]) + self.NNParameters[i][1])
            grad = (np.einsum('ij,jk->ik', grad, self.NNParameters[i][0]))
            # Elu activation
            grad *= self.reluPrime(input1)
            input1 = self.relu(input1)

        grad = np.einsum('ij,jk->ik', grad, self.NNParameters[i + 1][0])
        # grad stores all intermediate Jacobians, however only the last one is used here as output
        return grad[:5, :]

class ScaleNormalize(object):
    def __init__(self, y, x, lb, ub, test_size=0.05, random_state=42):
        """
        Scaler for pricing model's parameters to be fed into DNN
        :param y: target
        :param x: features
        :param lb: params lower bounds
        :param ub: params upper bounds
        :param test_size:
        :param random_state:
        """
        self.y = y
        self.x = x
        self.ub = ub
        self.lb = lb
        self.Npars = len(ub)
        self.test_size = test_size
        self.random_state = random_state

        self.train_split()

        self.scaler()

    def train_split(self):
        """
        train split data
        :return:
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y,
                             test_size=self.test_size,
                             random_state=self.random_state)

    def scaler(self):
        """
        scaler
        :return:
        """
        self.scale = StandardScaler()
        self.scale2 = StandardScaler()
        self.y_train_transform = self.scale.fit_transform(self.y_train)
        self.y_test_transform = self.scale.transform(self.y_test)
        self.x_train_transform = self.scale2.fit_transform(self.X_train)
        self.x_test_transform = self.scale2.transform(self.X_test)

    def xinversetransform(self, x):
        return self.scale2.inverse_transform(x)

    def myscale_old(self, x):
        res = np.zeros(self.Npars)
        for i in range(self.Npars):
            res[i] = (x[i] - (self.ub[i] + self.lb[i]) * 0.5) * 2 / (self.ub[i] - self.lb[i])
        return res

    def myscale(self, x):
        rb = (self.ub + self.lb) * 0.5
        rm = self.ub - self.lb
        res = (x - rb.values) * 2 / rm.values
        return res

class Optimization(object):
    def __init__(self, paramlb, paramub, model_pars, NumLayers, NNParameters, sample_ind=500, method='Levenberg-Marquardt',
                 x_test_transform=None):
        self.x_test_transform = x_test_transform
        self.sample_ind = sample_ind
        self.method = method
        self.init = np.zeros(len(paramlb))
        self.lb = paramlb
        self.ub = paramub
        self.model_pars = model_pars
        self.NumLayers = NumLayers
        self.NNParameters = NNParameters

        self.NN = NumpyNN(model_pars, NumLayers, NNParameters)

    def optimize(self):
        solution = []
        for i in range(5000):
            out = self.opt_method(i)
            print([i, out.x])
            solution.append(self.myinverse(out.x))
        return solution

    def opt_method(self, i):
        if self.method == "L-BFGS-B":
            opt = minimize(self.CostFunc, x0=self.init, args=i, method='L-BFGS-B', jac=self.Jacobian, tol=1E-10,
                         options={"maxiter": 5000})
        elif self.method == 'SLSQP':
            opt = minimize(self.CostFunc, x0=self.init, args=i, method='SLSQP', jac=self.Jacobian, tol=1E-10,
                         options={"maxiter": 5000})
        elif self.method == 'BFGS':
            opt = minimize(self.CostFunc, x0=self.init, args=i, method='BFGS', jac=self.Jacobian, tol=1E-10,
                         options={"maxiter": 5000})
        elif self.method == 'Levenberg-Marquardt':
            opt = least_squares(self.CostFuncLS, self.init, self.JacobianLS, args=(i,), gtol=1E-10)
        else:
            raise Warning('Methods in "L-BFGS-B ", "SLSQP", "BFGS", "Levenberg-Marquardt"')
        return opt

    def CostFunc(self, x, sample_ind):
        return np.sum(np.power((self.NN.NeuralNetwork(x) - self.x_test_transform[sample_ind]), 2))

    def Jacobian(self, x, sample_ind):
        return 2 * np.sum((self.NN.NeuralNetwork(x) - self.x_test_transform[sample_ind]) * self.NeuralNetworkGradient(x), axis=1)

    def CostFuncLS(self, x, sample_ind):
        return (self.NN.NeuralNetwork(x) - self.x_test_transform[sample_ind])

    def JacobianLS(self, x, sample_ind):
        return self.NN.NeuralNetworkGradient(x).T

    def myinverse(self, x):
        res = np.zeros(len(x))
        for i in range(len(x)):
            res[i] = x[i] * (self.ub[i] - self.lb[i]) * 0.5 + (self.ub[i] + self.lb[i]) * 0.5
        return res

class LoadNeuralNetsModels(object):

    def __init__(self, params, Nmat, model, nNodes=100, dropout=None, summary=True):
        """
        Loads the architecture of DNN
        :param params: trained paramters
        :param Nmat: int, number of maturities
        :param model: str, models
        :param nNodes: int, number of nodes
        :param dropout: DNN dropout
        :param summary: True prints the DNN architecture
        """
        self.model = model
        self.params = params
        self.Npars = len(params)
        self.Nmat = Nmat
        self.nNodes = nNodes
        self.dropout = dropout

        keras.backend.set_floatx('float64')

        self.modelGEN = self.load_model()

        if summary:
            self.modelGEN.summary()

        self.modelGEN.compile(loss=self.RMSE, optimizer="adam")

    def load_model(self):
        input1 = keras.layers.Input(shape=(self.Npars,))

        if self.model in ['Merton74basic', 'Merton76jump', 'KouJump']:
            x1 = keras.layers.Dense(self.nNodes, activation='relu')(input1)
            x2 = keras.layers.Dense(self.nNodes, activation='relu')(x1)
            out_layer = keras.layers.Dense(self.Nmat, activation='linear')(x2)

        elif self.model in ['Heston', 'HestonJump', '1SV1SJ', '2SV1SJ',
                            '0SV1SJ', '2SV0SJ', 'PanSingleton2008']:

            x1 = keras.layers.Dense(self.nNodes, activation='relu')(input1)
            x2 = keras.layers.Dense(self.nNodes, activation='relu')(x1)
            x3 = keras.layers.Dense(self.nNodes, activation='relu')(x2)
            out_layer = keras.layers.Dense(self.Nmat, activation='linear')(x3)
        else:
            raise Warning('Model is not specified correctly')

        modelGEN = keras.models.Model(inputs=input1, outputs=out_layer)
        return modelGEN

    def RMSE(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))



