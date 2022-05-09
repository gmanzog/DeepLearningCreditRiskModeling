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
    Paper replication
"""
import pandas as pd
from pandas import DataFrame
import pricing_models as myprc
import time
from environment import FOLDERS

# Set folder to save simulated data
folderData = FOLDERS['sims_models']

# list of models to simulate
Models = ['Merton74basic', 'Merton76jump', 'Heston', 'HestonJump',
          'KouJump', '1SV1SJ', '2SV1SJ', '0SV1SJ', '2SV0SJ',
          'PanSingleton2008']

# sample of data per model
nSamples = 1000000
mat = [1, 3, 5, 7, 10]

if __name__ == '__main__':
    SpreadDist = {}
    for model in Models:
        print(model)
        # select model and simulate combination of parameters
        sim_pars = myprc.ModelSelection(modelType=model, nSamples=nSamples)
        model_pars = sim_pars.model_pars
        common_inputs = sim_pars.common_inputs
        # simulate model
        t = time.time()
        spreads = {}
        for m in mat:
            common_inputs['maturity'] = m
            out = myprc.CreditModelPricer(param=model_pars,
                                          leverage=common_inputs['leverage'],
                                          maturity=common_inputs['maturity'],
                                          risk_free=0,
                                          div_yield=0,
                                          model_type=model)
            spreads['mat:' + str(m)] = out.spread

        spreads = DataFrame(spreads)
        sims = pd.concat([model_pars, common_inputs, spreads], axis=1)
        sims.to_pickle(folderData + '%s_sims.plk' % model)
        time_Taken = time.time() - t
        print(time_Taken)
        print(time_Taken / 60 / 60)
