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
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import pandas as pd
import numpy as np


pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

def nearPSD(A, epsilon=0):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return out

class SpreadNN(object):
    def __init__(self, Npars, NumLayers, NNParameters):
        self.NumLayers = NumLayers
        self.NNParameters = NNParameters
        self.Npars = Npars

    def elu(self, x):
        # Careful function ovewrites x
        ind = (x < 0)
        x[ind] = np.exp(x[ind]) - 1
        return x

    def relu(self, x):
        # Careful function ovewrites x
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

class KalmanUKF(object):
    def __init__(self, idx, CDS, lev, mat, rf, dt, model, params, NNpars, scaler, ub, lb):
        # print(idx)
        self.idx = idx
        self.lev = lev
        self.mat = mat
        self.rf = rf
        self.model = model
        self.params = params
        self.scaler = scaler
        self.ub = ub
        self.lb = lb

        self.dt = dt

        # Initiliaze Trained Neural Network
        Npars = len(params) - 1  # -1 because model params + measurement error
        NumLayers = NNpars['NumLayers']
        NNParameters = NNpars['NNParameters']
        self.NNpricing = SpreadNN(Npars=Npars, NumLayers=NumLayers, NNParameters=NNParameters)

        # scaled_data = scaler.fit_transform(CDS)
        self.CDS = CDS#.iloc[-1]

        self.get_params()

        self.llk, self.x_state, self.err2, self.fit_data = self.UKFfit()

    def myinverse(self, x, ub, lb):
        res = x * (ub - lb) * .5 + (ub + lb) * .5
        return res

    def myscale(self, x, ub, lb):
        res = (2 * x - (ub + lb)) / (ub - lb)
        return res

    def measur_eq(self, state_sigmas, params):
        # scaled_lev = self.myscale(x=self.lev, ub=0.01, lb=0.9)
        if self.model == 'Heston':
            parsNN = np.concatenate([state_sigmas, params * np.ones((3, 1)), self.lev.values * np.ones((3, 1))], axis=1)
        elif self.model == 'PanSingleton2008':
            parsNN = np.concatenate([state_sigmas, params * np.ones((3, 1))], axis=1)

        scaled_parsNN = self.myscale(parsNN, self.ub.values, self.lb.values)
        NN_fit = self.NNpricing.NeuralNetwork(x=scaled_parsNN * np.ones((3, 1)))
        sigmas_measur = self.scaler.inverse_transform(NN_fit)
        return sigmas_measur


    def UKFfit(self):
        n_pars_pricing = len(self.params) - 1
        n_factors = len(self.factors)
        nCDS = len(self.CDS)
        # state equation: mean and variance
        if self.model == 'Heston':
            F = np.exp(-self.kappa * self.dt)
            F2 = np.exp(-2 * self.kappa * self.dt)
            state_eq = lambda ff0: self.theta * (1 - F) + np.maximum(ff0, 0.00001) * F
            var_state_eq = lambda ff0: ff0 * self.vol ** 2 / self.kappa * (F - F2) + \
                                       self.theta * self.vol ** 2 / (2 * self.kappa) * (1 - F) ** 2
        elif self.model == 'PanSingleton2008':
            F2 = np.exp(-2 * self.kappa * self.dt)
            F = np.exp(-self.kappa * self.dt)
            state_eq = lambda ff0: np.exp(np.log(ff0) * np.exp(-self.kappa * self.dt) + self.theta / self.kappa * (1 - F))
            var_state_eq = lambda ff0: self.vol ** 2 / (2 * self.kappa) * (1 - F2)

        # initial mean and cov of the state: assume homoskedasticity across states
        x_state = self.factors
        P_state = self.vol ** 2

        R = np.eye(nCDS) * self.sigma_err

        points = MerweScaledSigmaPoints(n=n_factors, alpha=1e-3, beta=2., kappa=3)
        Wm, Wc = points.Wm, points.Wc

        err_iter = []
        state_iter = []
        cds_fit_iter = []
        ukf_cov_cds_iter = []
        err_sum_iter = []
        for _tt in range(5):
            # print(_tt)
            cds_tt = self.CDS
            r_t = self.rf
            x_state = state_eq(x_state)
            P_state = F ** 2 * P_state + var_state_eq(x_state)
            P_state = np.maximum(P_state, 0.0001)

            try:
                state_sigmas = points.sigma_points(x_state, P_state)
            except:
                print('PROBLEM here')
                P_state = nearPSD(P_state)
                state_sigmas = points.sigma_points(x_state, P_state)
            # number of sigma points is 2n+1
            self.n_sigma_point = 2 * n_factors + 1

            # use unscented transform to get new mean and covariance
            sigmas_measur = self.measur_eq(state_sigmas, params=self.params[:n_pars_pricing])


            # use unscented transform to get new mean and covariance
            ukf_mean_cds, ukf_cov_cds = unscented_transform(sigmas_measur, Wm, Wc, noise_cov=R)
            cds_fit_iter.append(ukf_mean_cds)
            ukf_cov_cds_iter.append(ukf_cov_cds)

            # Innovations
            err = (cds_tt - ukf_mean_cds)
            # print([_tt, np.sum(err ** 2)])
            err_sum_iter.append(np.sum(err ** 2))
            # print(np.sum(err ** 2))
            err_iter.append(err)

            # compute cross variance of the state and the measurements
            Pxz = np.zeros((n_factors, nCDS))
            for i in range(self.n_sigma_point):
                Pxz += Wc[i] * np.outer(state_sigmas[i] - x_state,
                                        sigmas_measur[i] - ukf_mean_cds)
            # Kalman gain
            inv_ukf_cov_cds = np.linalg.inv(ukf_cov_cds)
            K = np.dot(Pxz, inv_ukf_cov_cds)

            x_state = np.maximum(x_state + np.dot(K, err), 0.00001)
            state_iter.append(x_state)

            P_state = P_state - np.dot(K, ukf_cov_cds).dot(K.T)

        min_vals = np.argmin(err_sum_iter)
        min_err = err_iter[min_vals][:, None]
        min_ukf_cov_y = ukf_cov_cds_iter[min_vals]
        fit_data = cds_fit_iter[min_vals]
        inv_ukf_cov_cds = np.linalg.inv(min_ukf_cov_y)
        det = np.linalg.det(min_ukf_cov_y)
        err2_ = np.dot(np.dot(min_err.T, inv_ukf_cov_cds), min_err)
        llk = -.5 * nCDS * np.log(2 * np.pi) -.5 * np.log(det) - .5 * err2_
        x_state = state_iter[min_vals]
        ukf_mean_cds = cds_fit_iter[min_vals]
        minErr = err_sum_iter[min_vals]
        return llk, x_state, minErr, fit_data

    def get_params(self):
        if self.model == 'Heston':
            pars = ['vol', 'kappa', 'theta', 'rho']

        elif self.model == 'PanSingleton2008':
            pars = ['vol', 'kappa', 'theta']
        pars += ['sigma_err']
        for name, val, in zip(pars, self.params):
            setattr(self, name, val)
        f0 = np.exp(self.theta) if self.model == 'PanSingleton2008' else self.theta
        self.factors = np.array([f0])






