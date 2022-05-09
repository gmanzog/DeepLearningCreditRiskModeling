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
    Pricing function
"""
import numpy as np
from pandas import DataFrame, Series
import scipy.stats as si
from smt.sampling_methods import LHS
import math
import charafun_pricing as mychara
import scipy.linalg as la

class ModelSelection(object):
    def __init__(self, modelType, nSamples):
        """
        Select pricing model and its parameter sets
        :param modelType: str, see self.model_params for implemented models
        :param nSamples: int, samples to simulate, e.g. 1mm
        """
        sim_params, common_inputs, par_labels_order = self.LHSsampling(modelType=modelType, nSamples=nSamples)

        mod_pars_labels = [i for i in sim_params.columns if i not in common_inputs.keys()]

        self.model_pars = sim_params[mod_pars_labels]
        self.common_inputs = sim_params[common_inputs.keys()]
        self.param_labels = par_labels_order

    def LHSsampling(self, modelType, nSamples):
        """
        Latin hypercube sampling (LHS)
        :param modelType:
        :param nSamples:
        :return:
        """
        par_labels_order, all_ranges = self.model_params(modelType)

        # Add common inputs: leverage, risk free, maturity
        common_inputs = self.common_model_inputs()
        [(all_ranges.append(v), par_labels_order.append(i)) for i, v in common_inputs.items()]

        np.random.seed(12345)
        all_ranges = np.asarray(all_ranges)
        sampling = LHS(xlimits=all_ranges)
        sim_params = sampling(nSamples)
        sim_params = DataFrame(sim_params, columns=par_labels_order)
        return sim_params, common_inputs, par_labels_order

    def common_model_inputs(self):
        """
        set input for leverage for structural models
        :return:
        """
        inputs = {'leverage': [0.01, 0.99]}
        return inputs

    def fix_feller_condition(self, sim_params):
        """
        Enforce Feller condition for mean-reverting models (2 * k * Theta > eps^2)
        with k = reversion speed, Theta = reversion mean, eps: vol of vol
        :param sim_params:
        :return:
        """
        for pname in sim_params.columns:
            if 'kappa' in pname:
                state = pname.split('_')[1]
                drift = 2 * sim_params['kappa' + '_' + state] * sim_params['theta' + '_' + state]
                var = sim_params['sigma' + '_' + state] ** 2
                feller_false = drift < var
                sim_params.loc[feller_false, 'sigma' + '_' + state] = np.sqrt(drift.loc[feller_false])
        return sim_params

    def model_params(self, modelType):
        """
        Define parameter range for each model.
        :return: range of values for each parameter for each model
        """
        param_ranges = {'Merton74basic': {'sigma': [0.001, 2], 'mu': [0.01, 2]},
                        'Merton76jump': {'lambd': [0.001, 1], 'sigma': [0.001, 2],
                                         'muJ': [-2, 0], 'sigmaJ': [0.001, 1],
                                         'mu': [-0.5, .5]},
                        'Heston': {'v0': [0.001, 1], 'sigma_var': [0.001, 1],
                                   'kappa_var': [0.001, 2], 'theta_var': [0.001, 1],
                                   'rho': [-.99, -.05]},
                        'HestonJump': {'v0': [0.001, 1], 'sigma_var': [0.001, 2],
                                       'kappa_var': [0.001, 2], 'theta_var': [0.001, 1],
                                       'rho': [-.99, -.05],
                                       'lambd': [0.0001, 1],
                                       'muJ': [-2, 0],
                                       'sigmaJ': [0.001, 1]},
                        'KouJump': {'lambd': [0.001, 1], 'sigma': [0.001, 2],
                                    'pUp': [0.001, 1], 'mDown': [0.001, 1],
                                    'mUp': [0.001, 1]},
                        '1SV1SJ': {'v01': [0.001, 1], 'sigma_var1': [0.001, 2], 'kappa_var1': [0.001, 2],
                                   'theta_var1': [0.001, 1], 'rho1': [-.99, -.05],
                                   'z0': [0.001, 1], 'sigma_z': [0.001, 2], 'kappa_z': [0.001, 2],
                                   'theta_z': [0.001, 1], 'muJ': [-2, 0], 'sigmaJ': [0.001, 1],
                                   'a_Var': [0.001, 1]},
                        '2SV1SJ': {'v01': [0.001, 1], 'sigma_var1': [0.001, 2], 'kappa_var1': [0.001, 2],
                                   'theta_var1': [0.001, 1], 'rho1': [-.99, -.05],
                                   'v02': [0.001, 1], 'sigma_var2': [0.001, 2], 'kappa_var2': [0.001, 2],
                                   'theta_var2': [0.001, 1], 'rho2': [-.99, -.05],
                                   'z0': [0.001, 1], 'sigma_z': [0.001, 2], 'kappa_z': [0.001, 2],
                                   'theta_z': [0.001, 1], 'muJ': [-2, 0], 'sigmaJ': [0.001, 1],
                                   'a_Var': [0.001, 1]},
                        '0SV1SJ': {'z0': [0.001, 0.1], 'sigma_z': [0.001, 2], 'kappa_z': [0.001, 2],
                                   'theta_z': [0.001, 1], 'muJ': [-2, 0], 'sigmaJ': [0.001, 1]},
                        '2SV0SJ': {'v01': [0.001, 1], 'sigma_var1': [0.001, 2], 'kappa_var1': [0.001, 2],
                                   'theta_var1': [0.001, 1], 'rho1': [-.99, -.05],
                                   'v02': [0.001, 1], 'sigma_var2': [0.001, 2], 'kappa_var2': [0.001, 2],
                                   'theta_var2': [0.001, 1], 'rho2': [-.99, -.05]},
                        'PanSingleton2008': {'lmbd0': [0.001, 1], 'sigma_lmbd': [0.001, 2], 'kappa_lmbd': [0.001, 2],
                                             'theta_lmbd': [-7, 1]}}

        param_ranges = param_ranges[modelType]
        par_labels_order = []
        all_ranges = []
        for par_label, par_range in param_ranges.items():
            par_labels_order.append(par_label)
            all_ranges.append(par_range)
        return par_labels_order, all_ranges



class CreditModelPricer(object):

    def __init__(self, param, leverage, maturity, risk_free, div_yield, model_type, LGD=0.55,
                 states=None, nTrial=1e4, model=None):
        """
        Model pricer
        :param param: dict of params for each model, refer to SelectModels for implemented models
        :param leverage: int
        :param maturity:
        :param risk_free:
        :param div_yield:
        :param model_type: str, model to price, refer to SelectModels for implemented models
        :param states:
        :param nTrial:
        :param model:
        """
        self.param = param
        self.states = states
        self.debt = leverage
        self.rf = risk_free
        self.q = div_yield
        self.LGD = LGD
        self.nTrial = nTrial
        self.model = model
        self.assets = 1
        self.mat = maturity
        self.model_type = model_type

        if model_type == 'PanSingleton2008':
            self.spread = self.reducedform_model()
        else:
            # Compute Implied Put
            self.putPrice = self.implied_put()
            # Compute Spread
            self.spread = self.put2spread()

    def reducedform_model(self):
        """
        price the reduced form model of pan and singleton
        :return:
        """
        lmbd0 = self.param['lmbd0']
        DiscFun = DataFrame(np.ones((1, 100)))
        spread = {}
        for i in range(len(self.mat)):
            param = self.param.loc[i]
            obj = PanSingletonSims(lmbd0=lmbd0[i], param=param, LGD=self.LGD, DiscFun=DiscFun,
                                   mat=[self.mat[i]])
            spread[i] = obj.CDS[self.mat[i]]
            print([i, spread[i]])
        spread = Series(spread)
        return spread

    def call2putprice(self, callPrice):
        """
        Put-Call parity to retrieve put price
        :param callPrice:
        :return:
        """
        discFun = np.exp(-self.rf * self.mat)
        divFun = np.exp(-self.q * self.mat)
        putPrice = np.maximum(callPrice - self.assets * divFun + discFun * self.debt, 0)
        return putPrice

    def put2spread(self):
        """
        Convert put price into spreads
        Add dividend yield here if needed
        :return:
        """
        L = self.debt * np.exp(-self.rf * self.mat)
        CS = - np.log(1 - (self.putPrice.astype('float') / L)) / self.mat
        spread = np.maximum(CS, 0)
        return spread

    def bs_call_price(self, mu=None, rf=None, sigma=None):
        """
        Black-Scholes-Merton call option price
        :return:
        """
        # S: spot price
        # K: strike price
        # T: time to maturity
        # r: interest rate
        # sigma: volatility of underlying asset

        rf = rf if rf is not None else self.rf
        sigma = sigma if sigma is not None else self.param['sigma']
        mu = mu if mu is not None else self.param.get('mu', mu)

        # inverse of leverage
        Linv = self.assets * np.exp((mu - self.q) * self.mat) / self.debt
        sT = sigma * np.sqrt(self.mat)
        d1 = np.log(Linv) / sT + 0.5 * sT
        d2 = d1 - sT
        # Normal CDFs
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        # call price
        callPrice = (self.assets * np.exp(-self.q * self.mat) * Nd1 -
                     self.debt * np.exp(-rf * self.mat) * Nd2)
        return callPrice

    def bs_call_price_with_jump(self):
        """
        Blac-Scholes-Merton call price with jumps
        Merton (1974)
        :return:
        """
        try:
            lmbd = self.param['lambd']
            sigma = self.param['sigma']
            muJump = self.param['muJ']
            sigmaJump = self.param['sigmaJ']
        except:
            lmbd, sigma, muJump, sigmaJump = self.param

        k = np.exp(muJump + .5 * sigmaJump ** 2) - 1
        lmbd_hat = lmbd * (1 + k)

        callPrice = 0
        N = 100
        for i in range(N):
            sigma_ii = np.sqrt(sigma ** 2 + i * sigmaJump ** 2 / self.mat)
            r_ii = self.rf - k * lmbd + (i * np.log(1 + k) / self.mat)
            # jump-free call price
            bs_prices = self.bs_call_price(mu=r_ii, rf=r_ii, sigma=sigma_ii)
            # jump factor
            jump_factor = np.exp(-lmbd_hat * self.mat) * (lmbd_hat * self.mat) ** i
            jump_factor /= float(math.factorial(i))
            # final call price
            callPrice += jump_factor * bs_prices

        return callPrice

    def implied_put(self):
        """
        Model-implied put price
        :return:
        """
        if self.model_type == 'Merton74basic':
            callPrice = self.bs_call_price()
        elif self.model_type == 'Merton76jump':
            callPrice = self.bs_call_price_with_jump()
        elif self.model_type not in ['Merton74basic', 'Merton76jump']:
            callPrice = mychara.CharaBasedPrice(param=self.param, stock=self.assets, strike=self.debt,
                                                maturity=self.mat, risk_free=self.rf, div_yield=self.q,
                                                model_type=self.model_type, state=self.states).callPrice
        else:
            raise Warning('ModelType not properly specified or not existent')
        self.callPrice = callPrice
        putPrice = self.call2putprice(callPrice)
        return putPrice




class PanSingletonSims(object):
    def __init__(self, lmbd0, param, LGD, DiscFun, mat, pay_freq=4, FullyImplicitMethod=True):
        """
        Panl and Singleton (2008) pricing model
        :param lmbd0: initial value for log-normal process
        :param param: dict of params,  {'k', 'theta', 'sigma'}
        :param LGD: loss-given default
        :param DiscFun: dicount function
        :param mat: maturity, int
        :param pay_freq: premium frequency of CDs contract
        :param FullyImplicitMethod: numerical approximation of default/survival probability
        """
        self.LGD = LGD
        self.DiscFun = DiscFun
        self.mat = mat
        self.param = param
        self.lmbd0 = lmbd0
        self.freq = pay_freq
        self.payments = pay_freq * mat
        self.kappa = param['kappa_lmbd']
        self.theta = param['theta_lmbd']
        self.sig = param['sigma_lmbd']
        if FullyImplicitMethod:
            self.CDS = self.CrankNickPricing()
        else:
            self.CDS = self.sim_spread()

    def sim_spread(self, Nsims=1000, dt=1./360, burnout=200):
        """
        Simulate spreads
        :param Nsims: int, number of simulations
        :param dt: float, time fraction
        :param burnout: int, number of initial simulations to discard
        :return:
        """

        T = int(self.mat / dt)
        eps = np.random.normal(size=(T + burnout, Nsims))
        ln_lmbdt = np.zeros((T + burnout, Nsims))
        ln_lmbdt[0, :] = np.log(self.lmbd0)
        for t in range(1, T + burnout):
            ln_lmbdt[t, :] = ln_lmbdt[t - 1, :] + self.kappa * (self.theta - ln_lmbdt[t - 1, :]) * dt + self.sig * eps[t - 1, :] * np.sqrt(dt)

        lmbdt = np.exp(ln_lmbdt[burnout:, :])
        pay_dates = [(i + 1) * 90 for i in range(self.payments)]
        surv_prob = 0
        for q in range(self.payments):
            lq = lmbdt[:pay_dates[q], :]
            surv_prob += np.mean(np.exp(-np.sum(lq * dt, axis=0)))

        def_prob = 0
        for tq in range(T):
            def_prob += np.mean(lmbdt[tq, :] * np.exp(-np.sum(lmbdt[:tq, :] * dt, axis=0))) * dt

        CDS = def_prob * self.LGD / surv_prob
        return CDS

    def TDMASolve(self, a, b, c, d):
        """
        TDMA solver, a b c d can be NumPy array type or Python list type.
        refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
        :return:
        """
        nf = len(d)  # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
        for it in range(1, nf):
            mc = ac[it - 1] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]

        xc = bc
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        return xc

    def CDSgrid(self, T, DiscFun, SurvProb, DefProb):
        """
        Create the grid of CDS spread
        :param T: maturity of CDS spread
        :param DiscFun: discount function for that date
        :param SurvProb: grid of survival probability
        :param DefProb: grid of default probability
        :return: grid of CDS spreads
        """
        freq = 4
        pay_dates = [i * 3 for i in range(freq + 1)]
        prem_leg = {}
        def_leg = {}
        for q in range(freq):
            surv_prob = SurvProb[1:, pay_dates[q]:pay_dates[q + 1]]
            def_prob = DefProb[1:, pay_dates[q]:pay_dates[q + 1]]
            disc_fun = DiscFun.values[0, pay_dates[q]:pay_dates[q + 1]]
            prem_leg[q] = np.dot(surv_prob, disc_fun)
            def_leg[q] = np.dot(def_prob, disc_fun)

        # CDS spread
        cds_grid = self.LGD * DataFrame(def_leg).sum(axis=1) / DataFrame(prem_leg).sum(axis=1)
        return cds_grid

    def intSprd(self, CDSfitGridT, lmbdGrid):
        """
        Routing to interpolate spreads. This runs in parallel
        :param date: keep track of date in the parallelization
        :param obsCDS: observed spread at date t
        :param DiscFun: discount function at date t
        :param survProb: grid of survival probability
        :param defProb: grid of default probability
        :param ProbMeasure: probability measure 'Q' or 'P'
        :return: dict called results
        """
        expLmbdGrid = np.exp(lmbdGrid)
        idx_min = abs(expLmbdGrid - self.lmbd0).argmin()
        state_i1, state_i = expLmbdGrid[idx_min - 1], expLmbdGrid[idx_min]
        w = (self.lmbd0 - state_i1) / (state_i - state_i1)
        if idx_min >= self.M - 1:
            idx_min -= 1
        CDSfit = {T: (1. - w) * CDSfitGridT[T][idx_min - 1] + w * CDSfitGridT[T][idx_min] for T in self.mat}
        return CDSfit

    def FD_method(self, prob_type='SurvProd'):
        """
        Implicit Finite Difference method to construct survival and default probability
        :param prob_type: 'SurvProd' or 'DefProd'
        :param probMeasure: 'P' or 'Q'
        :return: grif od probabilities F and default intensity vecX
        """
        # number of grid points: intensity
        dX = (np.log(self.Xmax) - np.log(self.Xmin)) / self.M
        # intensity points
        points = np.arange(self.M)
        vecX = np.log(self.Xmin) + dX * points

        F = np.zeros((self.M, self.N))
        # drift of the log-intensity of default
        driftV = self.kappa * (self.theta - vecX) * self.dt

        # boundary values
        F[:, 0] = 1 if prob_type == 'SurvProb' else np.exp(vecX)

        C1 = self.sig ** 2. / (2. * dX ** 2.) + driftV / (2. * dX)
        C3 = self.sig ** 2. / (2. * dX ** 2.) - driftV / (2. * dX)
        C2 = -self.sig ** 2. / dX ** 2. - np.exp(vecX) - 1. / self.dt

        Ab = np.zeros((3, self.M))
        Ab[0, 1:] = C1[:-1]
        Ab[1, :] = C2
        Ab[2, :-1] = C3[1:]

        for nn in range(1, self.N):
            A = -F[:, nn - 1] / self.dt
            A[-1] = 0
            ss = la.solve_banded((1, 1), ab=Ab, b=A)
            F[:, nn] = ss.flatten()
        return F, vecX

    def set_finite_diff_pars(self):
        """
        Set parameters for Finite Difference Method
        :return:
        """
        self.Xmin = 0.000001
        self.Xmax = 1
        self.dt = 1 / 12
        self.M = 200
        self.N = 120

    def CrankNickPricing(self):
        """
        Crank-Nicholson routine to copmute probabilities numerically
        :return:
        """
        self.set_finite_diff_pars()

        # Compute Survivial probabilities using finite implicit method
        SurvProbGrid, StateGrid = self.FD_method(prob_type='SurvProb')

        # Compute Default probabilities using finite implicit method
        DefProbGrid, _ = self.FD_method(prob_type='DefProb')

        CDSfitGridT = {T: self.CDSgrid(T, self.DiscFun, SurvProbGrid, DefProbGrid) for T in self.mat}

        cds = self.intSprd(CDSfitGridT=CDSfitGridT, lmbdGrid=StateGrid)
        return cds



if __name__ == '__main__':
    import time
    param = [0.61, 0.4, .3]
    LGD = 0.55
    DiscFun = DataFrame(np.ones((50, 390)))
    mat = [1, 3, 5, 7, 10]
    mat = [1]
    lmbd0 = .0599
    pay_freq = 4

    # PanSingleton(param, LGD, DiscFun, mat)
    time0 = time.time()
    out = PanSingletonSims(lmbd0, param, LGD, DiscFun, mat, pay_freq, FullyImplicitMethod=True)
    time1 = time.time()
    print([out.CDS, 'CrankNicholson time: ', time1 - time0])

