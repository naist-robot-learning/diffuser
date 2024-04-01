import torch
from torch.func import vmap, jacrev, functional_call
from functorch.compile import aot_function
import time
from UR5kinematicsAndDynamics_vectorized import fkine

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module


def square_function(inputs):
    outputs = torch.empty(inputs.shape, dtype=torch.float32).to('cuda:0')
    for i in range(0, inputs.shape[0]):
        outputs[i,0] = inputs[i,0]**2
    res = (outputs ** 2)
    return res
def ts_compile(fx_g, inps):
    print("compiling")
    f = torch.jit.script(fx_g)
    #import ipdb; ipdb.set_trace()
    #f = torch.jit.freeze(f.eval())
    return f

def ts_compiler(f):
    return aot_function(f, ts_compile, ts_compile)

def functorch_jacobian(inputs):
    def _func_single(inputs):
        return fkine(inputs).sum(axis=0)
    return jacrev(_func_single)(inputs)

#aot_print_fn = aot_function(funct, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
functorch_jacobian3 = ts_compiler(lambda points: functorch_jacobian(points)) 
#f = torch.jit.script(jacrev(fkine)(inputs))

# Generate dummy inputs
for i in range(0, 6):
    batch_size = 32
    inputs = torch.randn(6, batch_size, requires_grad=True).to('cuda:0')
    jacobian = torch.empty(batch_size,6,6, dtype=torch.float32).to('cuda:0')
    # Compute the outputs
    #outputs = square_function(inputs)
    # Compute the Jacobian
    
    #start = time.time()
    #jac = torch.autograd.functional.jacobian(aot_print_fn, inputs, vectorize=True)
    #end = time.time()
    #print("time to compute jacobian: ", end-start)
    start = time.time()
    #jac = torch.autograd.functional.jacobian(fkine, inputs, vectorize=True)
    f = functorch_jacobian3(inputs)
    #jac = functorch_jacobian3(inputs)
    end = time.time()
    #start = time.time()
    jac = jacrev(fkine)(inputs)
    #end = time.time()
    print("normal jacobian computation: 0.080 s")
    print("vectorized jacobian computation: 0.020 s")
    print("time to compute jacobian: ", end-start)
    for i in range(0, batch_size):
        jacobian[i,:,:] = jac[i,:,:,i]
    
    import ipdb; ipdb.set_trace()