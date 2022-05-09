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
    Model calibration to real data
"""
import pandas as pd
from environment import FOLDERS
from calib_utils import UnscentedKFLogLlk
from dnn_training.dnn_trainer import DNNTrainer
from scipy.optimize import minimize
import pickle

# Set folders
folderSims = FOLDERS['sims_models']
folderTrainedPars = FOLDERS['trained_models']
folderMktData = FOLDERS['market_data']
folderCalib = FOLDERS['calibrated_models']

Models = ['PanSingleton2008', 'Heston', 'Heston_mktLev']
maturities = [1, 3, 5, 7, 10]

# Load market data and resample weekly
data = pd.read_excel(folderMktData + "market_data.xlsx").dropna()
data = data.set_index('Unnamed: 0').resample('W').last()
ycols = [i for i in data.columns if 'spread' in i]
spreads = data[ycols]

method = 'SLSQP'
options = {'eps': 1e-12}

for model in Models:
    # load leverage
    lev = data[['mkt_leverage']] if 'mktLev' in model else data[['leverage']]

    print(model)
    dnnObj = DNNTrainer(model=model, maturities=maturities)

    simdata = dnnObj.data
    # collect min/max for each param
    lb = simdata['lb']
    ub = simdata['ub']

    # load trained params
    NNParameters = dnnObj.load_trained_weights()

    NumLayers = 3
    dt = 1 / 52
    NNpars = {'NumLayers': NumLayers, 'NNParameters': NNParameters}
    llkObj = UnscentedKFLogLlk(spreads=spreads, lev=lev, mat=maturities,
                               rf=0, dt=dt, model=model,
                               params=None, ub=ub, lb=lb, NNpars=NNpars)

    obj = lambda x: llkObj.negllk(params=x)[0]
    x0 = llkObj.x0
    bounds = llkObj.bounds

    opt = minimize(obj, x0=x0, method=method, tol=1e-8, bounds=bounds, options=options)
    opt_pars = opt.x
    print(opt.x)

    print('------------Optimal Pars')
    print(opt_pars)
    optObj = UnscentedKFLogLlk(spreads=spreads, lev=lev, mat=maturities,
                               rf=0, dt=dt, model=model,
                               params=None, ub=ub, lb=lb, NNpars=NNpars)
    llk, results = optObj.negllk(params=opt_pars)
    f = open(folderCalib + "%s_calib_output.pkl" % model, "wb")
    pickle.dump(results, f)
    f.close()
