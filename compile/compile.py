from model import Model_U
import torch
import numpy as np
import os
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
device = torch.device("cpu")


if __name__ == '__main__':
    model_u = Model_U()
    input_data = torch.Tensor(1, 64, 261).unsqueeze(-1)
    # scripted_model = torch.jit.trace(torch_model, calib_data).eval()
    quantizer = torch_quantizer('calib', model_u, input_data, device=device)
    
    samples = os.listdir('./calib')
    quant_model = quantizer.quant_model
    for s in samples: 	
        data = torch.load('./calib/'+s)
        quant_model(data.unsqueeze(-1))
    quantizer.export_quant_config()
    
    #calib
    quantizer = torch_quantizer('test', model_u, input_data, device=device)
    compile_model = quantizer.quant_model
    for s in samples: 	
        data = torch.load('./calib/'+s)
        compile_model(data.unsqueeze(-1))

    quantizer.export_xmodel(deploy_check=True)

# vai_c_xir -x ./quantize_result/Model_U_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -n vad_model_u

# [UNILOG][FATAL][XCOM_DATA_OUTRANGE][Data value is out of range!]
