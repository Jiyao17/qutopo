# install NetSquid with the following command
# replace ***** with the password
# pip3 install --extra-index-url https://jiyao:*****@pypi.netsquid.org netsquid

from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.models.qerrormodels import DephaseNoiseModel

# Create a new delay model
fiber_delay_model = FibreDelayModel()
space_delay_model = FibreDelayModel(c=3e5)
print("Speed of light in fibre: %.2f km/s" % fiber_delay_model.c)
print("Speed of light in space: %.2f km/s" % space_delay_model.c)

from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.qchannel import QuantumChannel
loss_model = FibreLossModel(p_loss_init=0.83, p_loss_length=0.2)
qchannel = QuantumChannel("MyQChannel", length=20, models={'quantum_loss_model': loss_model})

