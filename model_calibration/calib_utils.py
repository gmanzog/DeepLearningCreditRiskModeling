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
    Model calibration to real data utils
"""
import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, least_squares
from dnn_training.dnn_utils import NumpyNN
from joblib import Parallel, delayed
import multiprocessing as mp
from uk_dnn_filter import KalmanUKF

class UnscentedKFLogLlk(object):
    def __init__(self, spreads, lev, mat, rf, dt, model, params, ub, lb, NNpars):
        """
        PArallel calibration of a DNN trained model to real data using Unscented Kalman Filter
        :param spreads: spread data
        :param lev: leverage data
        :param mat: maturities
        :param rf: risk free
        :param dt: time fraction
        :param model: str, model
        :param param: model parameters
        :param ub: params upper bounds
        :param lb: params lower bounds
        :param NNpars: DNN trained parameters
        """
        self.spreads = spreads
        self.model = model
        self.mat = mat
        self.lev = lev
        self.ub = ub
        self.lb = lb
        self.dt = dt
        self.rf = rf
        self.NNpars = NNpars
        self.x0, self.bounds, self.par_labels = self.set_param_bounds()
        self.params = params

        self.lb_se = [i[0] for i in self.bounds]
        self.ub_se = [i[1] for i in self.bounds]

        self.num_cores = mp.cpu_count()
        # num_cores = 12
        print(['num cores: ', self.num_cores])



    def calib_output(self, kf_t):
        """
        Extract output from parallelization and copmute performance metrics
        :param kf_t:
        :return:
        """
        T = len(kf_t)
        N = self.spreads.shape[1]
        llk = 0
        states = []
        fit_data = {}
        ss_res = 0
        for i in range(T):
            llk += kf_t[i].llk
            factors = kf_t[i].x_state[0]
            states.append(factors)
            ss_res += kf_t[i].err2
            fit_data[i] = kf_t[i].fit_data

        fit_data = DataFrame(fit_data, index=self.spreads.columns).T
        fit_data.index = self.spreads.index

        states = np.asarray(states)
        if states.ndim == 1:
            states = Series(states, index=self.spreads.index)
        else:
            states = DataFrame(states, index=self.spreads.index)

        r2, r2mat = self.r2(y=self.spreads, y_hat=fit_data, ss_res=ss_res)
        results = {'states': states, 'fittedvals': fit_data, 'sq_errs': ss_res,
                   'neg_llk': -llk / (T * N), 'r2': r2, 'r2byMat': r2mat}
        return results

    def r2(self, y, y_hat, ss_res):
        """
        R-squared
        :param y:
        :param y_hat:
        :param ss_res:
        :return:
        """
        err2sum = (y - y_hat).pow(2).sum()
        ss_tot = y.sub(y.mean()).pow(2).sum()
        r2mat = 1 - err2sum / ss_tot
        ss_tot = np.sum(ss_tot)
        r2 = 1 - ss_res / ss_tot
        return r2, r2mat

    def set_param_bounds(self):
        """
        Set models bounds for optimization. Add here new models to calibrate
        :return:
        """
        if self.model == 'Heston':
            x0 = np.array([1.34103376e-01, 1.03998288e-03, 1.49012263e-01, -9.36743934e-01,
                           1.60319430e-05])
            bounds = ((0.001, 5), (0.001, 5), (0.001, 2), (-0.98, -0.05), (0.000001, 5))
            par_labels = ['vol', 'kappa', 'theta', 'rho', 'sigma_eps']

        elif self.model == 'PanSingleton2008':
            x0 = np.array([0.39267124, 0.03060086, -0.37510181, 0.00090748])
            bounds = ((0.001, 5), (0.001, 2), (-7, 1), (0.000001, 5))
            par_labels = ['vol', 'kappa', 'theta', 'sigma_eps']
        else:
            raise ValueError('Model %s has not been implemented' % self.model)
        return x0, bounds, par_labels

    def negllk(self, params=None):
        params = params if params is not None else self.params
        params = np.maximum(np.minimum(params, self.ub_se), self.lb_se)
        scaler = StandardScaler()
        scaler.fit_transform(self.spreads)

        T, N = self.spreads.shape

        with Parallel(n_jobs=self.num_cores) as parallel:
            kf_t = parallel(delayed(KalmanUKF)(idx=_t, CDS=self.spreads.iloc[_t], lev=self.lev.iloc[_t],
                                               mat=self.mat, rf=self.rf, dt=self.dt, model=self.model,
                                               params=params, NNpars=self.NNpars, scaler=scaler,
                                               ub=self.ub, lb=self.lb)
                            for _t in range(T))

        results = self.calib_output(kf_t=kf_t)
        # self.fittedvals = results['fittedvals']
        # self.states = results['states']
        neg_llk = results['neg_llk'].flatten()[0]
        # self.ss_res = results['sq_errs']
        # self.r2mat = results['r2byMat']
        r2 = results['r2']
        print({'r2': r2, 'negllk': neg_llk, 'params': params.tolist()})
        return neg_llk, results



class ModelCalibrator(object):
    def __init__(self, obsData, paramlb, paramub, model_pars, NumLayers, NNParameters, sample_ind=500, method='Levenberg-Marquardt',
                 x_test_transform=None, obsParams=None):
        """
        Calibration of a model to real data
        :param obsData: Series of observed data
        :param paramlb: parameters lower bounds
        :param paramub: parameters upper bounds
        :param model_pars:
        :param NumLayers: int, number of DNN layers
        :param NNParameters: trained DNN parameters
        :param sample_ind:
        :param method:
        :param x_test_transform:
        :param obsParams:
        """

        self.scale = StandardScaler()
        self.scaled_data = DataFrame(self.scale.fit_transform(obsData))

        self.x_test_transform = x_test_transform
        self.sample_ind = sample_ind
        self.method = method
        self.lb = paramlb.values
        self.ub = paramub.values
        self.model_pars = model_pars
        self.NumLayers = NumLayers
        self.NNParameters = NNParameters
        self.obsData = obsData
        self.dates = obsData.index
        self.obsParams = obsParams
        Npars = len(paramlb)
        if obsParams is not None:
            Npars -= obsParams.shape[1]

        self.NN = NumpyNN(model_pars, NumLayers, NNParameters)

    def optimize(self):
        x0 = np.array([0.02, 0.2, 0.06, 0.23, -.5])
        solution = {}
        data_fit = {}
        window = 52
        for t in range(window, len(self.dates)):
            scale = StandardScaler()
            obs_data = self.obsData.loc[:t]
            scaled_data = scale.fit_transform(obs_data)
            # optimize Heston params
            out = self.opt_method(t, x0, obs_data=scaled_data[-1, :], obsparam=self.obsParams.loc[t])

            # scale the observed inputs
            scaled_obs_inputs = self.myscale(self.obsParams.loc[t], ub=self.ub[-1], lb=self.lb[-1])
            # store optimal params (that are scaled) and the (scaled) observed inputs
            opt_inputs = np.insert(out.x, len(out.x), scaled_obs_inputs)
            solution[t] = self.myinverse(opt_inputs)
            print([t, solution[t]])
            _fit = self.NN.NeuralNetwork(opt_inputs)
            data_fit[t] = scale.inverse_transform(_fit)
            # data_fit[t] = _fit * self.scale.scale_ + self.scale.mean_
            print(np.corrcoef(data_fit[t], self.obsData.loc[t])[0, 1])
            # solution.append(out.x)
        return DataFrame(solution).T, DataFrame(data_fit).T


    def opt_method(self, i, x0, obs_data, obsparam):
        """
        'Methods in "L-BFGS-B ", "SLSQP", "BFGS", "Levenberg-Marquardt"'
        :param i:
        :param x0:
        :param obs_data:
        :param obsparam:
        :return:
        """
        if self.method == 'Levenberg-Marquardt':
            lb = [-0.99] * 5
            ub = [0.99] * 5
            opt = least_squares(self.CostFuncLS, x0, self.JacobianLS, args=(i, obs_data, obsparam),
                                gtol=1E-10, bounds=(lb, ub))
        else:
            opt = minimize(self.CostFunc, x0=x0, args=(i, obs_data, obsparam), method=self.method, jac=self.Jacobian,
                           tol=1E-10,
                           options={"maxiter": 5000}, bounds=((-.99, .99), )*5)

        return opt

    def CostFunc(self, x, idx, obs_data, obsparam):
        """
        Cost function: MSE
        :param x:
        :param idx:
        :param obs_data:
        :param obsparam:
        :return:
        """
        x = np.insert(x, len(x), obsparam)
        err = self.NN.NeuralNetwork(x) - obs_data
        return np.sum(err ** 2)

    def Jacobian(self, x, idx, obs_data, obsparam):
        """
        DNN Jacobian
        :param x:
        :param idx:
        :param obs_data:
        :param obsparam:
        :return:
        """
        x = np.insert(x, len(x), obsparam)

        err = self.NN.NeuralNetwork(x) - obs_data
        grad = self.NN.NeuralNetworkGradient(x)
        return 2 * np.sum(err * grad, axis=1)

    def CostFuncLS(self, x, idx, obs_data, obsparam):
        x = np.insert(x, len(x), obsparam)
        return (self.NN.NeuralNetwork(x) - obs_data)

    def JacobianLS(self, x, idx, obs_data, obsparam):
        x = np.insert(x, len(x), obsparam)
        return self.NN.NeuralNetworkGradient(x).T

    def myinverse(self, x):
        res = np.zeros(len(x))
        for i in range(len(x)):
            res[i] = x[i] * (self.ub[i] - self.lb[i]) * 0.5 + (self.ub[i] + self.lb[i]) * 0.5
        return res

    def myscale(self, x, ub, lb):
        res = (2 * x - (ub + lb)) / (ub - lb)
        return res
