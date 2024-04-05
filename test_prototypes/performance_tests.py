import torch
import torch.nn as nn
import pdb
import numpy as np
import einops
import time
from pytictac import Timer, CpuTimer


def run_cost_function_exec_time_test():
    """Run a test to measure the execution time of the cost function"""
    ## torch tensor shape: [ batch x horizon x transition ] [ 64 x 32 x 26 ]
    for i in range(0, 2):
        torch.manual_seed(42)
        print("Running naive cost function execution time test... ")
        x = torch.randn(64, 32, 26).to("cuda").requires_grad_()

        ###### NAIVE IMPLEMENTATION, BASICALLY USELESS ########
        from UR5kinematicsAndDynamics import compute_reflected_mass

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        x = einops.rearrange(x, "b h t -> b t h")
        cost = torch.empty((batch_size, horizon)).to("cuda")
        u = torch.empty(3).to("cuda")
        u[0] = 1
        u[1] = 0
        u[2] = 0
        for i in range(0, batch_size):
            for j in range(0, horizon):
                cost[i, j] = compute_reflected_mass(x[i, 6:12, j], u)
        final_cost = cost.sum(axis=1).sum(axis=0)
        end.record()
        torch.cuda.synchronize()
        print("final_cost", final_cost)
        print("Cost computation time: ", start.elapsed_time(end))
        start.record()
        grad = torch.autograd.grad([final_cost.sum()], [x])[0]
        end.record()
        torch.cuda.synchronize()
        print("Gradient computation time: ", start.elapsed_time(end))


def run_cost_function_exec_time_test_vectorized():
    """Run a test to measure the execution time of the cost function"""
    ## torch tensor shape: [ batch x horizon x transition ] [ 64 x 32 x 26 ]
    print("\n")
    print("Running vectorized cost function execution time test... ")
    for i in range(0, 2):
        torch.manual_seed(42)
        x = torch.randn(64, 32, 26).to("cuda").requires_grad_()
        ####### (TODO) VECTORISED IMPLEMENTATION ########
        from UR5kinematicsAndDynamics_vectorized import compute_reflected_mass

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        x = einops.rearrange(x, "b h t -> b t h")
        cost = torch.empty(batch_size, horizon).to("cuda")
        u = torch.empty((horizon, 3, 1), dtype=torch.float32).to("cuda")
        u[:, 0] = 1
        u[:, 1] = 0
        u[:, 2] = 0
        for i in range(0, batch_size):
            cost[i, :] = compute_reflected_mass(x[i, 6:12, :], u)

        final_cost = cost.sum(axis=1).sum(axis=0)
        end.record()
        torch.cuda.synchronize()
        print("final_cost", final_cost)
        print("Cost computation time: ", start.elapsed_time(end))
        start.record()

        grad = torch.autograd.grad([final_cost.sum()], [x])[0]
        end.record()
        torch.cuda.synchronize()
        print("Gradient computation time: ", start.elapsed_time(end))

    return final_cost


def run_cost_function_exec_time_test_vectorized_precompiled():
    """Run a test to measure the execution time of the cost function"""
    ## torch tensor shape: [ batch x horizon x transition ] [ 64 x 32 x 26 ]
    print("\n")
    print("Running precompiled vectorized cost function execution time test... ")
    torch.manual_seed(42)
    x = torch.randn(64, 65, 26).to("cuda").requires_grad_()
    ####### (TODO) VECTORISED IMPLEMENTATION ########
    model = UR5Model()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    batch_size = x.shape[0]
    horizon = x.shape[1]
    transition_dim = x.shape[2]
    x = einops.rearrange(x, "b h t -> b t h")
    cost = torch.empty(batch_size, horizon).to("cuda")
    u = torch.empty((horizon, 3, 1), dtype=torch.float32).to("cuda")
    u[:, 0] = 1
    u[:, 1] = 0
    u[:, 2] = 0
    for i in range(0, batch_size):
        cost[i, :] = model.compute_precompiled_reflected_mass_with_precompiled_jacobian(
            x[i, 6:12, :], u
        )

    final_cost = cost.sum(axis=1).sum(axis=0)
    end.record()
    torch.cuda.synchronize()
    print("final_cost", final_cost)
    print("Cost computation time: ", start.elapsed_time(end))
    start.record()

    grad = torch.autograd.grad([final_cost.sum()], [x])[0]
    end.record()
    torch.cuda.synchronize()
    print("Gradient computation time: ", start.elapsed_time(end))
    return final_cost


def run_cost_function_exec_time_test_vectorized_all_batched():
    """Run a test to measure the execution time of the cost function"""
    ## torch tensor shape: [ batch x horizon x transition ] [ 64 x 32 x 26 ]
    print("\n")
    print("Running vectorized cost function batched execution time test... ")
    for i in range(0, 2):
        torch.manual_seed(42)
        x = torch.randn(64, 32, 26).to("cuda").requires_grad_()
        ####### (TODO) VECTORISED IMPLEMENTATION ########
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        from UR5kinematicsAndDynamics_vectorized_v21 import compute_reflected_mass

        start.record()
        batch_size = x.shape[0]
        horizon = x.shape[1]
        transition_dim = x.shape[2]
        x = einops.rearrange(x, "b h t -> b t h")
        # x_batch = einops.rearrange(x, 'b h t -> t (b h)')
        # import ipdb; ipdb.set_trace()
        cost = torch.empty(batch_size, horizon).to("cuda")
        u = torch.empty((batch_size * horizon, 3, 1), dtype=torch.float32).to("cuda")
        u[:, 0] = 1
        u[:, 1] = 0
        u[:, 2] = 0
        cost = compute_reflected_mass(x[:, 6:12, :], u)
        final_cost = cost.sum(axis=1).sum(axis=0)
        end.record()
        torch.cuda.synchronize()
        print("final_cost", final_cost)
        print("Cost computation time: ", start.elapsed_time(end))
        start.record()
        grad = torch.autograd.grad([final_cost.sum()], [x])[0]
        end.record()
        torch.cuda.synchronize()
        print("Gradient computation time: ", start.elapsed_time(end))
    return final_cost


if __name__ == "__main__":

    run_cost_function_exec_time_test()
    run_cost_function_exec_time_test_vectorized()
    # run_cost_function_exec_time_test_vectorized_precompiled()
    run_cost_function_exec_time_test_vectorized_all_batched()
