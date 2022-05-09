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
    Characteristic function
"""
import numpy as np
from pandas import DataFrame
from scipy import optimize
from scipy.interpolate import interp1d
import numpy.matlib
import multiprocessing as mp


class CharFun(object):
    def __init__(self, u, A0, param, rf, q, tau, model_type, state0=None):
        """
        Characteristic functions for different model_type
        :param u: points of the integrand
        :param A0: asset value at 0, scalar
        :param param: dict of parameters. E.g. {'sigma': .2, 'lambd': .5}. See each charac for param names
        :param rf: annualized risk free rate
        :param q: annualized dividend yield
        :param tau: time to maturity
        :param model_type: one of Heston, HestonJump, KouJump, BlackScholes, BlackScholesJump
        """
        self.param = param
        self.tau = tau
        self.rf = rf
        self.q = q
        self.u = u
        self.model_type = model_type
        self.x0 = A0
        self.state0 = state0

        self.cf = self.charfun()

    def heston(self):
        """
        Heston basic and Heston with Jump
        'Heston'
            v0 = spot variance level
            kappa = mean reversion speed
            theta = mean interest rate level
            sigma = volatility of variance
            rho = correlation between innovations in variance and spot process

        'HestonJump': Heston with jumps. Requires 'Heston' and the following:
            muJ = expected jump size
            sigmaJ = jump volatility
            lambd = jump intensity
        """
        try:
            kappa = self.param['kappa_var']
            theta = self.param['theta_var']
            sigma = self.param['sigma_var']
            rho = self.param['rho']
            v0 = self.param['v0']
        except:
            kappa, theta, sigma, rho = self.param
            v0 = self.state0

        u = self.u

        a = -.5 * u * 1j * (1 - u * 1j)
        b = rho * sigma * u * 1j - kappa
        c = .5 * sigma ** 2
        m = -np.sqrt(b ** 2 - 4 * a * c)

        emt = np.exp(m * self.tau)
        beta2 = (m - b) / sigma ** 2 * (1 - emt) / (1 - (b - m) / (b + m) * emt)
        alpha = (self.rf - self.q) * u * 1j * self.tau + \
                kappa * theta * (m - b) / sigma ** 2 * self.tau + \
                kappa * theta / c * np.log((2 * m) / ((m - b) * emt + b + m))
        phi = alpha + u * 1j * np.log(self.x0) + beta2 * v0
        return np.exp(phi)

    def hestonJump(self):
        """
        Heston basic and Heston with Jump
        'Heston'
            v0 = spot variance level
            kappa = mean reversion speed
            theta = mean interest rate level
            sigma = volatility of variance
            rho = correlation between innovations in variance and spot process

        'HestonJump': Heston with jumps. Requires 'Heston' and the following:
            muJ = expected jump size
            sigmaJ = jump volatility
            lambd = jump intensity
        """
        u = self.u

        lambd = self.param['lambd']
        muJ = self.param['muJ']
        sigmaJ = self.param['sigmaJ']

        compensator = np.exp(muJ + .5 * sigmaJ ** 2) - 1
        compensator_u1j = np.exp(1j * u * muJ - u ** 2 * .5 * sigmaJ ** 2) - 1

        phi_jump = lambd * (compensator_u1j - u * 1j * compensator)

        phi_Heston = self.heston()
        phi = phi_Heston * np.exp(phi_jump * self.tau)
        return phi

    def heston_stochastic_jump(self):
        """
        Heston basic and Heston with Jump
        'Heston'
            v0 = spot variance level
            kappa = mean reversion speed
            theta = mean interest rate level
            sigma = volatility of variance
            rho = correlation between innovations in variance and spot process

        'HestonJump': Heston with jumps. Requires 'Heston' and the following:
            muJ = expected jump size
            sigmaJ = jump volatility
            lambd = jump intensity
        """
        # param for stochastic vol
        v0 = self.param['v0']
        eta_var = self.param['sigma_var']
        k_var = self.param['kappa_var']
        theta_var = self.param['theta_var']
        rho = self.param['rho']

        # param for stochastic jump
        j0 = self.param['j0']
        eta_jump = self.param['sigma_jump']
        k_jump = self.param['kappa_jump']
        theta_jump = self.param['theta_jump']
        muJ = self.param['muJ']
        sigmaJ = self.param['sigmaJ']

        u = self.u
        xi_jump = np.exp(muJ + .5 * sigmaJ ** 2) - 1
        phi_jump = 1 - np.exp(1j * u * muJ - .5 * u ** 2 * sigmaJ ** 2)

        # vol_1 factor
        cV1 = k_var - 1j * u * rho * eta_var

        dV1 = np.sqrt(cV1 ** 2 + (1j * u + u ** 2) * eta_var ** 2)
        fV1 = np.divide(cV1 - dV1, cV1 + dV1)

        ge = np.exp(-dV1 * self.tau)
        gf = np.multiply(fV1, ge)

        CV1 = np.multiply((cV1 - dV1) / eta_var ** 2, np.divide(1 - ge, 1 - gf))
        BV1 = (k_var * theta_var) / (eta_var ** 2) * (
                (cV1 - dV1) * self.tau - 2 * np.log(np.divide(1 - gf, 1 - fV1)))

        # jump factor
        d_jump = np.sqrt(k_jump ** 2 + (1j * u * xi_jump + phi_jump) * 2 * eta_jump ** 2)
        f_jump = np.divide(k_jump - d_jump, k_jump + d_jump)

        ge_j = np.exp(-d_jump * self.tau)
        gf_j = np.multiply(f_jump, ge_j)

        C_jump = (k_jump - d_jump) / (eta_jump ** 2) * np.divide(1 - ge_j, 1 - gf_j)
        B_jump = (k_jump * theta_jump) / (eta_jump ** 2) * ((k_jump - d_jump) * self.tau - 2 * np.log(
            np.divide(1 - gf_j, 1 - f_jump)))

        drift_psi = 1j * u * (self.rf - self.q)

        phi = drift_psi * self.tau + BV1 + CV1 * v0 + B_jump + C_jump * j0 + u * 1j * np.log(self.x0)
        return np.exp(phi)

    def most_complex_model(self):
        """
        Heston basic and Heston with Jump
        'Heston'
            v0 = spot variance level
            kappa = mean reversion speed
            theta = mean interest rate level
            sigma = volatility of variance
            rho = correlation between innovations in variance and spot process

        'HestonJump': Heston with jumps. Requires 'Heston' and the following:
            muJ = expected jump size
            sigmaJ = jump volatility
            lambd = jump intensity
        """
        skipV1 = skipV2 = skipZ = False
        # param for stochastic vol 1
        try:
            v01 = self.param['v01']
            eta_var1 = self.param['sigma_var1']
            k_var1 = self.param['kappa_var1']
            theta_var1 = self.param['theta_var1']
            rho1 = self.param['rho1']
        except:
            skipV1 = True
            v01 = eta_var1 = k_var1 = theta_var1 = rho1 = 0

        # param for stochastic vol 2
        try:
            v02 = self.param['v02']
            eta_var2 = self.param['sigma_var2']
            k_var2 = self.param['kappa_var2']
            theta_var2 = self.param['theta_var2']
            rho2 = self.param['rho2']
        except:
            skipV2 = True
            v02 = eta_var2 = k_var2 = theta_var2 = rho2 = 0

        # param for stochastic jump
        try:
            z0 = self.param['z0']
            eta_z = self.param['sigma_z']
            k_z = self.param['kappa_z']
            theta_z = self.param['theta_z']
            muJ = self.param['muJ']
            sigmaJ = self.param['sigmaJ']
        except:
            skipZ = True
            z0 = eta_z = k_z = theta_z = muJ = sigmaJ = 0

        try:
            a_Var = self.param['a_Var']
        except:
            a_Var = 0

        # j0 = a * (v01 + v02) + z0

        u = self.u
        xi_jump = np.exp(muJ + .5 * sigmaJ ** 2) - 1
        phi_jump = 1 - np.exp(1j * u * muJ - .5 * u ** 2 * sigmaJ ** 2)

        # vol_1 factor
        if not skipV1:
            cV1 = k_var1 - 1j * u * rho1 * eta_var1

            # dV1 = np.sqrt(cV1 ** 2 + (1j * u + u ** 2) * eta_var1 ** 2)
            dV1 = np.sqrt(cV1 ** 2 + (1j * u + u ** 2) * eta_var1 ** 2 +
                          (1j * u * xi_jump + phi_jump) * a_Var * 2 * eta_var1 ** 2)
            fV1 = np.divide(cV1 - dV1, cV1 + dV1)

            ge1 = np.exp(-dV1 * self.tau)
            gf1 = np.multiply(fV1, ge1)

            CV1 = np.multiply((cV1 - dV1) / eta_var1 ** 2, np.divide(1 - ge1, 1 - gf1))
            BV1 = (k_var1 * theta_var1) / (eta_var1 ** 2) * (
                    (cV1 - dV1) * self.tau - 2 * np.log(np.divide(1 - gf1, 1 - fV1)))
        else:
            CV1 = BV1 = 0

        # vol_2 factor
        if not skipV2:
            cV2 = k_var2 - 1j * u * rho2 * eta_var2

            # dV2 = np.sqrt(cV2 ** 2 + (1j * u + u ** 2) * eta_var2 ** 2)
            dV2 = np.sqrt(cV2 ** 2 + (1j * u + u ** 2) * eta_var2 ** 2 +
                          (1j * u * xi_jump + phi_jump) * a_Var * 2 * eta_var2 ** 2)
            fV2 = np.divide(cV2 - dV2, cV2 + dV2)

            ge2 = np.exp(-dV2 * self.tau)
            gf2 = np.multiply(fV2, ge2)

            CV2 = np.multiply((cV2 - dV2) / eta_var2 ** 2, np.divide(1 - ge2, 1 - gf2))
            BV2 = (k_var2 * theta_var2) / (eta_var2 ** 2) * (
                    (cV2 - dV2) * self.tau - 2 * np.log(np.divide(1 - gf2, 1 - fV2)))
        else:
            CV2 = BV2 = 0

        # z factor
        if not skipZ:
            d_z = np.sqrt(k_z ** 2 + (1j * u * xi_jump + phi_jump) * 2 * eta_z ** 2)
            f_z = np.divide(k_z - d_z, k_z + d_z)

            ge_z = np.exp(-d_z * self.tau)
            gf_z = np.multiply(f_z, ge_z)

            C_z = (k_z - d_z) / (eta_z ** 2) * np.divide(1 - ge_z, 1 - gf_z)
            B_z = (k_z * theta_z) / (eta_z ** 2) * ((k_z - d_z) * self.tau - 2 * np.log(
                np.divide(1 - gf_z, 1 - f_z)))
        else:
            C_z = B_z = 0

        drift_psi = 1j * u * (self.rf - self.q) * self.tau + u * 1j * np.log(self.x0)

        phi = drift_psi + \
              BV1 + CV1 * v01 + \
              BV2 + CV2 * v02 + \
              B_z + C_z * z0
        return np.exp(phi)

    def kou_jump(self):
        """
        Kou's model with asymmetric double exponential jump distribution
        sigma = diffusive volatility of spot process
        lambd = jump intensity
        pUp = probability of upward jump
        mUp = mean upward jump (set 0 < mUp < 1 for finite expectation)
        mDown = mean downward jump
        """
        sigma = self.param['sigma']
        lambd = self.param['lambd']
        pUp = self.param['pUp']
        mDown = 1 / self.param['mDown']
        mUp = 1 / self.param['mUp']
        u = self.u
        comp = lambda x: pUp * mUp / (mUp - x) + (1 - pUp) * mDown / (mDown + x) - 1
        alpha = -self.rf * self.tau + u * 1j * (self.rf - self.q - 0.5 * sigma ** 2 - lambd * comp(1)) * self.tau -\
                0.5 * u ** 2 * sigma ** 2 * self.tau + self.tau * lambd * comp(u * 1j)
        phi = alpha + u * 1j * np.log(self.x0)
        return np.exp(phi)

    def blackscholes_jump(self):
        """
        Black and Scholes with Jump characteristic function
        sigma = spot volatility
        muJ = expected jump size
        sigmaJ = jump volatility
        lambda = jump intensity
        """
        sigma = self.param['sigma']
        muJ = self.param['muJ']
        sigmaJ = self.param['sigmaJ']
        lambd = self.param['lambd']
        u = self.u
        m = np.exp(muJ+1 / 2 * sigmaJ ^ 2) - 1

        alpha = -self.rf * self.tau + u * 1j * self.tau * (self.rf - self.q - 0.5 * sigma ** 2 - lambd * m) - \
                0.5 * u ** 2 * sigma ** 2 * self.tau +\
                lambd * self.tau * (np.exp(muJ * u * 1j - 0.5 * u ** 2 * sigmaJ ** 2) - 1)

        phi = alpha + u * 1j * np.log(self.x0)
        phi = np.exp(phi)
        return phi

    def blackscholes(self):
        """
        Black and Scholes characteristic function
        sigma = spot volatility
        """
        sigma = self.param['sigma']
        u = self.u
        phi = -self.rf * self.tau + u * 1j * np.log(self.x0) + u * 1j * self.tau * (self.rf - self.q - 0.5 * sigma ** 2) - \
              0.5 * u ** 2 * sigma ** 2 * self.tau
        phi = np.exp(phi)
        return phi

    def charfun(self):
        if self.model_type == 'Heston':
            cf = self.heston()
        elif self.model_type == 'HestonJump':
            cf = self.hestonJump()
        elif self.model_type == 'HestonStochasticJump':
            cf = self.heston_stochastic_jump()
        elif self.model_type == 'KouJump':
            cf = self.kou_jump()
        elif self.model_type == 'BlackScholesJump':
            cf = self.blackscholes_jump()
        elif self.model_type == 'BlackScholes':
            cf = self.blackscholes()
        elif self.model_type in ['1SV1SJ', '2SV1SJ', '0SV1SJ', '2SV0SJ']:
            cf = self.most_complex_model()
        else:
            raise Warning('The model you are requesting does not exist')
        return cf

class CarrMadanPricing(object):
    def __init__(self, char_fun, price, strike, maturity, risk_free,
                 optimize_alpha=False):
        """
        Carr-Madan pricing function for call options
        :param char_fun: characteristic function
        :param price: price level
        :param strike: strike price
        :param maturity: maturity
        :param risk_free: risk free rate
        :param optimize_alpha: Carr-Madan numerical alpha
        """
        self.strike = strike
        self.price = price
        self.rf = risk_free
        self.T = maturity
        self.N = 1  # numer of contracts to price
        self.char_fun = char_fun
        self.optimize_alpha = optimize_alpha  # if True, we get more 0 values

        self.callPrice = self.price_call_option()

    def LordKahlFindAlpha(self, alph, v=0):
        CF = self.char_fun(u=v - (alph + 1) * 1j)
        CF = np.divide(CF, alph ** 2 + alph - v ** 2 + 1j * (2 * alph + 1) * v).T
        y = - alph * np.log(self.strike) + .5 * np.log(np.real(CF) ** 2)
        return y

    def optimal_alpha(self):
        """
        optimal Carr-Madan alpha
        :return:
        """
        obj = lambda x: self.LordKahlFindAlpha(alph=x)
        opt_alphas = optimize.fminbound(obj, 0.1, 1, disp=False)
        return opt_alphas

    def price_call_option(self):
        """
        numerical routine of the pricing function
        :return:
        """
        n = 10  # Limit of integration
        k = np.log(self.strike)  # log - Strike

        D = np.exp(-self.rf * self.T)  # Discount Function

        N = 2 ** n  # the integration is performed over N summations which are usually a power of 2
        eta = .05  # spacing of the integrand

        lmbd = (2 * np.pi) / (N * eta)  # spacing size of the log-Strike
        b = (N * lmbd) / 2  # the log-Strike will be constrained between -b and +b

        u = np.arange(1, N + 1)[None, :]
        kk = -b + lmbd * (u - 1)  # log - Strike levels
        jj = np.arange(1, N + 1)
        vj = (jj - 1) * eta

        if self.optimize_alpha:
            opt_alpha = self.optimal_alpha()
        else:
            opt_alpha = 0.2

        CF = self.char_fun(u=vj - (opt_alpha + 1) * 1j)
        CF = np.divide(CF,
                       np.kron(opt_alpha ** 2 + opt_alpha - vj ** 2 + 1j * (2 * opt_alpha + 1) * vj,
                               np.ones((self.N, 1)))).T
        calls = D * np.multiply(CF, np.exp(np.matlib.repmat(1j * vj * b, self.N, 1)).T) * (eta / 3)

        SimpsonW = 3 + (-1) ** jj - ((jj - 1) == 0)

        calls = np.multiply(calls, np.matlib.repmat(SimpsonW, self.N, 1).T)

        # price interpolation
        kkvec = kk[0]
        callsVector = (np.multiply(np.exp(-opt_alpha * kkvec) / np.pi, np.fft.fft(calls.T))).real
        callprice = np.diag(interp1d(kkvec, callsVector)(k))
        return callprice

class CharaBasedPrice(object):
    def __init__(self, param, stock, strike, maturity, risk_free, div_yield, model_type, state=None):
        """
        Class to price call options using the closed-form characteristic function defined in CharFun
        :param param: set of paramters. DataFrame nSims x nParam with columns named as params.
                    See each model in CharaFun for the proper names:
        :param stock: this is set to 1 (asset value), thus D is leverage
        :param strike: leverage, usually in (0, 1), DataFrame nSims x 1
        :param maturity: Maturity, DataFrame nSims x 1
        :param risk_free: risk-free, DataFrame nSims x 1
        """
        self.Nsims = len(param)
        self.T = maturity
        self.rf = risk_free
        self.q = div_yield
        self.stock = stock
        self.strike = strike
        self.param = param
        self.model_type = model_type
        self.state0 = state

        callPrice = []
        for i in range(len(param)):
            call_tmp = self.single_call_price(i, self.param.iloc[i], self.T.iloc[i], 0, 0,
                                              self.strike.iloc[i])
            callPrice.append(call_tmp)
        self.callPrice = DataFrame(callPrice).set_index(0).sort_index()[1]

    def single_call_price(self, idx, param, T, rf, q, strike):
        # print(idx)
        char_fun = lambda u: CharFun(u, A0=self.stock, param=param,
                                     rf=rf, q=q, tau=T, model_type=self.model_type, state0=self.state0).cf
        try:
            call_price = CarrMadanPricing(char_fun=char_fun, price=self.stock, strike=strike,
                                          maturity=T, risk_free=rf).callPrice[0][0]
        except:
            # this try-except is very ugly but its a quick inexpensive trick for now
            call_price = CarrMadanPricing(char_fun=char_fun, price=self.stock, strike=strike,
                                          maturity=T, risk_free=rf).callPrice[0]
        return [idx, call_price]

    def apply_async_with_callback(self):
        result_list = []

        def log_result(result):
            # This is called whenever foo_pool(i) returns a result.
            # result_list is modified only by the main process, not the pool workers.
            result_list.append(result)

        pool = mp.Pool()
        __spec__ = None
        for i in range(self.Nsims):
            pool.apply_async(self.single_call_price,
                             args=(i, self.param.iloc[i], self.T.iloc[i], self.rf.iloc[i], self.q.iloc[i],
                                   self.strike.iloc[i]),
                             callback=log_result)
        pool.close()
        pool.join()
        return result_list


