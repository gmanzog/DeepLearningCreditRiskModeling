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
    DNN model training
"""
from dnn_trainer import DNNTrainer

maturities = [1, 3, 5, 7, 10]

Models = ['Merton74basic', 'Merton76jump', 'Heston', 'HestonJump',
          'KouJump', '1SV1SJ', '2SV1SJ', '0SV1SJ', '2SV0SJ',
          'PanSingleton2008']
Models = ['Merton74basic']
NNfits = {}
for model in Models:
    print(model)
    dnnObj = DNNTrainer(model=model, maturities=maturities)
    # train model
    dnnObj.fit(verbose=True)
    # predictions
    preds = dnnObj.predict()
    # store accuracy
    NNfits[model] = dnnObj.r2



