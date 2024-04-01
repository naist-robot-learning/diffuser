import torch
from torch.func import vmap, jacrev, functional_call
from functorch.compile import aot_function
import time

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module


def square_function(inputs):
    res = (inputs ** 2)
    return res
#aot_print_fn = aot_function(square_function, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
# Generate dummy inputs
for i in range(0, 6):
    batch_size = 32
    inputs = torch.randn(batch_size, 6, requires_grad=True).to('cuda:0')
    jacobian = torch.empty(batch_size,6,6, dtype=torch.float32).to('cuda:0')
    # Compute the outputs
    #outputs = square_function(inputs)
    # Compute the Jacobian
    
    #start = time.time()
    #jac = torch.autograd.functional.jacobian(aot_print_fn, inputs, vectorize=True)
    #end = time.time()
    #print("time to compute jacobian: ", end-start)
    start = time.time()
    jac = torch.autograd.functional.jacobian(square_function, inputs, vectorize=True)
    end = time.time()
    print("normal jacobian computation: 0.080 s")
    print("vectorized jacobian computation: 0.020 s")
    print("time to compute jacobian: ", end-start)
    for i in range(0, batch_size):
        jacobian[i,:,:] = jac[i,:,i,:]
    
    import ipdb; ipdb.set_trace()