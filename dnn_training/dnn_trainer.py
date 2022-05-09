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
    DNN trainer
"""
import pandas as pd
import numpy as np
from dnn_training.dnn_utils import ScaleNormalize, LoadNeuralNetsModels
from environment import FOLDERS
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

folderSims = FOLDERS['sims_models']
folderTrainedDNN = FOLDERS['trained_models']

class DNNTrainer(object):
     def __init__(self, model, maturities):
        """
        Train DNN model
        :param model: str, one of the simulated models
        :param maturities: list of int, meturities
        """
        self.model = model
        self.mats = maturities

        # load simulated data
        self.data = self.load_simulated_data()
        # prepare data to feed into DNN
        self.proc_data = self.data_processing(data=self.data)
        # load DNN model
        param_labels = self.data['param_labels']
        self.dnn_generator = self.load_dnn_model(model=model, param_labels=param_labels)

     def load_simulated_data(self):
        """
        Load simulated data
        :return: dict of simulated data
        """
        sim_dat = pd.read_pickle(folderSims + self.model + "_sims.plk")
        if 'PanSingleton' in self.model:
            sim_dat.drop('leverage', inplace=True, axis=1)
        spread_cols = [i for i in sim_dat.columns if 'mat:' in i and int(i.split(':')[1]) in self.mats]
        spreads = sim_dat[spread_cols]

        pars_cols = [ i for i in sim_dat.columns if i not in spread_cols + ['maturity']]
        pars = sim_dat[pars_cols].values
        lb = sim_dat[pars_cols].min()
        ub = sim_dat[pars_cols].max()
        data = {'params': pars, 'lb': lb, 'ub': ub, 'spreads': spreads, 'param_labels': pars_cols}
        return data

     def data_processing(self, data):
        """
        Prepare data for training DNN
        :param data: dict of simulated data
        :return:
        """
        spreads = data['spreads']
        pars = data['params']
        lowerb = data['lb']
        upperb = data['ub']
        scaler = ScaleNormalize(y=spreads, x=pars, lb=lowerb, ub=upperb)
        # pars_train_transform = np.array([scaler.myscale(x) for x in scaler.X_train])
        pars_train_transform = scaler.myscale(scaler.X_train)
        # pars_test_transform = np.array([scaler.myscale(x) for x in scaler.X_test])
        pars_test_transform = scaler.myscale(scaler.X_test)
        proc_data = {'scaler': scaler,
                     'train_transf': pars_train_transform,
                     'test_transf': pars_test_transform}
        return proc_data

     def load_dnn_model(self, model, param_labels):
         """
         load DNN model architecture
         :param model: str
         :return:
         """
         Nmat = len(self.mats)
         loadNN = LoadNeuralNetsModels(params=param_labels,
                                       Nmat=Nmat,
                                       model=model)
         modelGEN = loadNN.modelGEN
         return modelGEN

     def fit(self, verbose=False):
         """
         Train and save the DNN model
         :return:
         """
         scaler = self.proc_data['scaler']
         pars_train_transform = self.proc_data['train_transf']
         pars_test_transform = self.proc_data['test_transf']
         self.dnn_generator.fit(x=pars_train_transform, y=scaler.y_train_transform,
                                batch_size=1024,
                                validation_data=(pars_test_transform, scaler.y_test_transform),
                                epochs=500,
                                verbose=verbose, shuffle=1)
         self.dnn_generator.save_weights(folderTrainedDNN + '%s_DNNWeights.h5' % self.model)
         return print('Model %s has been trained' % self.model)

     def load_trained_weights(self):
        """
        Load trained params
        :return:
        """
        try:
            self.dnn_generator.load_weights(folderTrainedDNN + '%s_DNNWeights.h5' % self.model)
        except:
            raise ValueError('Model %s han not been trained yet' % self.model)

        NNParameters=[]
        for i in range(1, len(self.dnn_generator.layers)):
           NNParameters.append(self.dnn_generator.layers[i].get_weights())
        return NNParameters

     def predict(self):
         """
         Prediction using trained model
         :return:
         """
         # load trained model
         scaler = self.proc_data['scaler']
         pars_test_transform = self.proc_data['test_transf']

         NNParameters = self.load_trained_weights()

         prediction = {}
         for k in range(scaler.X_test.shape[0]):
             pred = self.dnn_generator.predict(pars_test_transform[k][:, None].T)[0]
             prediction[k] = scaler.scale.inverse_transform(pred)

         cds_cols = self.data['spreads'].columns
         prediction = pd.DataFrame(prediction, index=cds_cols).T

         input_data = scaler.y_test.reset_index(drop=True)
         err2sum = np.sum((input_data - prediction) ** 2)
         y2sum = np.sum((input_data - input_data.mean()) ** 2)
         self.r2 = 1 - err2sum / y2sum
         print('Model r2: ')
         print(self.r2)
         return prediction



