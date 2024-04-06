import torch
from functorch.compile import aot_function
import numpy as np
import time 

# UR5 Standard DH parameters (a, alpha, d, theta)
DH_params = torch.tensor([
    (0.0, torch.pi/2, 0.08946),   # link 1 parameters
    (-0.425, 0.0, 0.0),           # link 2 parameters
    (-0.39225, 0.0, 0.0),         # link 3 parameters
    (0.0, torch.pi/2, 0.10915),   # link 4 parameters
    (0.0, -torch.pi/2, 0.09465),  # link 5 parameters
    (0.0, 0.0, 0.0823)            # link 6 parameters
]).to('cuda')

def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

class UR5Model:
    def __init__(self):
        self.DH_params = DH_params
        self.aot_fkine = aot_function(
            self._fkine, 
            fw_compiler=compiler_fn, bw_compiler=compiler_fn
            )
        
        self.aot_compute_reflected_mass = aot_function(
            self.compute_reflected_mass, 
            fw_compiler=compiler_fn, bw_compiler=compiler_fn
            )
        
        self.aot_compute_reflected_mass_with_precompiled_jacobian = aot_function(
            self.compute_reflected_mass_with_precompiled_jacobian, 
            fw_compiler=compiler_fn, bw_compiler=compiler_fn
            )
        
    def _dh_to_transformation(self, a, alpha, d, theta):
        """ Convert DH parameters to a homogenous transformation matrix
    
        Args:
            a (float): link length
            alpha (float): link twist
            d (float): link offset
            theta (float): joint angle
    
        Returns:
            T (4x4 torch tensor): homogenous transformation matrix
            
        """
        #import ipdb; ipdb.set_trace()
        horizon = theta.shape[0]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
        T = torch.eye(4, dtype=torch.float32, device='cuda').unsqueeze(0).expand(horizon, -1, -1).clone()
        T[:,0,0] = cos_theta
        T[:,0,1] = -sin_theta * cos_alpha
        T[:,0,2] = sin_theta * sin_alpha
        T[:,0,3] = a * cos_theta
        T[:,1,0] = sin_theta
        T[:,1,1] = cos_theta * cos_alpha
        T[:,1,2] = -cos_theta * sin_alpha
        T[:,1,3] = a * sin_theta
        T[:,2,1] = sin_alpha
        T[:,2,2] = cos_alpha
        T[:,2,3] = d
        return T
        
    def _compute_fk(self, q):
        """ Computes the forward kinematics of the UR5 manipulator
    
        Args:
            q (6x1 torch tensor): joint angles
            
        Returns:
            T (4x4 torch tensor): homogenous transformation matrix
            
        """
        #import ipdb; ipdb.set_trace()   
        horizon = q.shape[1]
        assert len(q) == 6, "There should be 6 joint angles"
        T = torch.eye(4, dtype=torch.float32, device='cuda').unsqueeze(0).expand(horizon, -1, -1)
        for i, (a, alpha, d) in enumerate(DH_params):
            T = torch.matmul(T, self._dh_to_transformation(a, alpha, d, q[i,:]))
        return T
    
    def _compute_rpy_xyz(self, H):
        ''' Compute the roll, pitch, yaw angles of a x-y-z rotation sequence from an 
        homogenous transformation matrix. The code was extracted and adapted to pytorch
        from Peter Corke's robotics toolbox for Python.
        Link: https://petercorke.github.io/robotics-toolbox-python
        
        Args:
            H: 4x4 torch tensor
        
        Returns:
            rpy: 3x1 torch tensor
               
        '''
        horizon = H.shape[0]
        rpy = torch.empty((horizon,3), dtype=torch.float32).to('cuda')
        tol = 20
        eps = 1e-6
                
        # Check for singularity
        abs_r13_minus_1 = torch.abs(torch.abs(H[:, 0, 2]) - 1)
        singularity_mask = abs_r13_minus_1 < tol * eps
        
        # Compute roll (rpy[:, 0])
        rpy[:, 0] = torch.where(singularity_mask, torch.tensor(0, dtype=H.dtype, device=H.device),
                                -torch.atan2(H[:, 0, 1], H[:, 0, 0]))
        # Compute yaw (rpy[:, 2])
        #rpy[:, 2] = -torch.atan2(H[:, 1, 2], H[:, 2, 2])
        mask2 = H[:, 0, 2] > 0
        rpy[:, 2] = torch.where(singularity_mask, 
                                torch.where(mask2, torch.atan2(H[:, 2, 1], H[:, 1, 1]),
                                            -torch.atan2(H[:, 1, 0], H[:, 2, 0])), 
                                -torch.atan2(H[:, 1, 2], H[:, 2, 2]))
        
        # Compute pitch (rpy[:, 1])
        rpy[:, 1] = torch.where(singularity_mask, torch.asin(torch.clip(H[:, 0, 2], -1.0, 1.0)),
                                 -9999999)
        
        # Handling cases where k is not 0 or 1 (k == 2 or k == 3)
        k = torch.argmax(torch.abs(torch.stack((H[:, 0, 0], H[:, 0, 1], H[:, 1, 2], H[:, 2, 2]), dim=1)), dim=1)
        cr = torch.cos(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
        sr = torch.sin(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
        sr2 = torch.sin(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))
        cr2 = torch.cos(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))
        
        rpy[:, 1] = torch.where(k == 0, torch.atan(H[:, 0, 2] * cr / H[:, 0, 0]), rpy[:, 1])
        rpy[:, 1] = torch.where(k == 1, -torch.atan(H[:, 0, 2] * sr / H[:, 0, 1]), rpy[:, 1])
        rpy[:, 1] = torch.where(k == 2, -torch.atan(H[:, 0, 2] * sr2 / H[:, 1, 2]), rpy[:, 1])
        rpy[:, 1] = torch.where(k == 3, torch.atan(H[:, 0, 2] * cr2 / H[:, 2, 2]), rpy[:, 1])
    
        return rpy[:,0], rpy[:,1], rpy[:,2]
        
    def _hom_to_pose(self, H):
        """Converts a 4x4 homogenous transformation matrix to a 6x1 compact pose
          
        Args:
            H (4x4 torch tensor): homogenous transformation matrix
    
        Returns:
            pose (6x1 torch tensor): [x, y, z, roll, pitch, yaw]
    
        """
        horizon = H.shape[0]
        pose = torch.empty((horizon, 6), dtype=torch.float32).to('cuda')
        pose[:,0] = H[:,0, 3]
        pose[:,1] = H[:,1, 3]
        pose[:,2] = H[:,2, 3]
        pose[:,3], pose[:,4], pose[:,5] = self._compute_rpy_xyz(H)
    
        return pose
    
    def _fkine(self, q):
        """ Computes the forward kinematics of the UR5 manipulator
        in compact representation [x, y, z, roll, pitch, yaw]
    
        Args:
            q (6x1 torch tensor): joint angles
    
        Returns:
            compact (6x1 torch tensor): [x, y, z, roll, pitch, yaw]
            
        """
        H = self._compute_fk(q)
        compact = self._hom_to_pose(H)
        return compact
    
    def compute_analytical_jacobian(self, q):     
        horizon = q.shape[1]
        timer = False
        if timer:
            start_time = time.time()
            outputs = self._fkine(q)
            end_time = time.time()
            print("Forward kinematics time: ", end_time - start_time)
            start_time = time.time()
            jac = torch.autograd.functional.jacobian(self._fkine, q, create_graph=False, vectorize=True).to('cuda')
            end_time = time.time()
            print("Jacobian computation time: ", end_time - start_time)
            start_time = time.time()
            J = torch.empty((horizon,6, 6), dtype=torch.float32).to('cuda')
            for i in range(0, horizon):
                J[i,:,:] = jac[i,:,:,i]
            end_time = time.time()
            print("Jacobian rearrange time: ", end_time - start_time)
        else:
            jac = torch.autograd.functional.jacobian(self._fkine, q, create_graph=False, vectorize=True).to('cuda')
            J = torch.empty((horizon,6, 6), dtype=torch.float32).to('cuda')
            for i in range(0, horizon):
                J[i,:,:] = jac[i,:,:,i]
        return J
    
    def compute_precompiled_analytical_jacobian(self, q):     
        horizon = q.shape[1]
        jac = torch.autograd.functional.jacobian(self.aot_fkine, q, create_graph=False, vectorize=True).to('cuda')
        J = torch.empty((horizon,6, 6), dtype=torch.float32).to('cuda')
        for i in range(0, horizon):
            J[i,:,:] = jac[i,:,:,i]
        return J
    
    
    def compute_inertia_matrix(self, q):
        """ Computes the inertia matrix of the UR5 manipulator based on 
        https://github.com/kkufieta/ur5_modeling_force_estimate and 
        adapted to pytorch.
        
        Matrix structure
        
        Mq = [m11 m12 m13 m14 m15 m16;
             m21 m22 m23 m24 m25 m26;
             m31 m32 m33 m34 m35 m36;
             m41 m42 m43 m44 m45 m46;
             m51 m52 m53 m54 m55 m56;
             m61 m62 m63 m64 m65 m66];  
                    
        Args:
            q (6xBatch torch tensor): joint angles
            
        Returns:
            Mq (Batchx6x6 torch tensor): inertia matrix
        """
        ## Returns the dynamics of the UR5 manipulator
        horizon = q.shape[1]
        Mq = torch.empty(horizon,6, 6, dtype=torch.float32).to('cuda')
        q1 = q[0,:]
        q2 = q[1,:]
        q3 = q[2,:]
        q4 = q[3,:]
        q5 = q[4,:]
        q6 = q[5,:]
        
        cos_q234_2q5 = torch.cos(q2 + q3 + q4 + 2.0*q5)
        cos_2q5_neg342 = torch.cos(2.0*q5 - q3 - q4 - q2)
        cos_q5neg3neg4neg2 = torch.cos(q5 - q3 - q4 - q2)
        cos_q2345 = torch.cos(q2 + q3 + q4 + q5)
        cos_q234 = torch.cos(q2 + q3 + q4)
        sin_q234 = torch.sin(q2 + q3 + q4)
        sin_q235 = torch.sin(q2 + q3 + q5)
        sin_q23neg5 = torch.sin(q2 + q3 - q5)
        sin_q45 = torch.sin(q4 + q5)
        sin_q25 = torch.sin(q2 + q5)
        sin_q2_neg5 = torch.sin(q2 - q5) 
        sin_q34 = torch.sin(q3 + q4)
        sin_q23 = torch.sin(q2 + q3)
        cos_q5 = torch.cos(q5)
        sin_q5 = torch.sin(q5)
        cos_q4 = torch.cos(q4)
        sin_q4 = torch.sin(q4)
        cos_q3 = torch.cos(q3)
        sin_q3 = torch.sin(q3)
        sin_q2 = torch.sin(q2)
        
        Mq[:,0,0] = m11 = 0.007460552829*torch.sin(q3 + q4 + q5) - 0.0002254000336*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 + 2.0*q5) + 0.007460552829*torch.sin(2.0*q2 + q3 + q4 + q5) + 0.7014282607*torch.cos(2.0*q2 + q3) - 0.007460552829*torch.sin(q3 + q4 - q5) - 0.007460552829*torch.sin(2.0*q2 + q3 + q4 - q5) - 0.05830173968*torch.sin(2.0*q2 + 2.0*q3 + q4) + 0.001614617887*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 + q5) + 0.8028639871*torch.cos(2.0*q2) + 0.0004508000672*torch.cos(2.0*q5) - 0.001614617887*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 - q5) - 0.0002254000336*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 - 2.0*q5) - 0.063140533*sin_q34 + 0.006888811168*sin_q45 - 0.063140533*torch.sin(2.0*q2 + q3 + q4) - 0.006888811168*torch.sin(q4 - q5) + 0.7014282607*cos_q3 + 0.00765364949*cos_q5 + 0.006888811168*torch.sin(2.0*q2 + 2.0*q3 + q4 + q5) + 0.3129702942*torch.cos(2.0*q2 + 2.0*q3) - 0.05830173968*sin_q4 - 0.005743907935*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4) - 0.006888811168*torch.sin(2.0*q2 + 2.0*q3 + q4 - q5) + 1.28924888
        Mq[:,0,1] = m12 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23 + 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5+ 0.348513447*sin_q2
        Mq[:,0,2] = m13 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23
        Mq[:,0,3] = m14 = 0.01615783641*cos_q234 - 0.0004508000672*cos_q234_2q5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345
        Mq[:,0,4] = m15 = 0.006888811168*sin_q23neg5 - 0.006888811168*sin_q235 - 0.002840479501*cos_q234 - 0.0002987944852*cos_q5neg3neg4neg2 - 0.00352803026*cos_q2345 - 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5
        Mq[:,0,5] = m16 = -0.000138534912*sin_q234*sin_q5
        Mq[:,1,0] = m21 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*torch.cos(q5 - q3 - q4 - q2) + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23 + 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5+ 0.348513447*sin_q2
        Mq[:,1,1] = m22 = 1.402856521*cos_q3 - 0.1166034794*sin_q4 - 0.126281066*cos_q3*sin_q4 - 0.126281066*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.02984221131*sin_q3*sin_q4*sin_q5 + 0.02984221131*cos_q3*cos_q4*sin_q5 + 2.263576776 
        Mq[:,1,2] = m23 = 0.7014282607*cos_q3 - 0.1166034794*sin_q4 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.644094696 
        Mq[:,1,3] = m24 = 0.01377762234*cos_q4*sin_q5 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.01612863983 
        Mq[:,1,4] = m25 = cos_q5*(0.01492110566*sin_q34 + 0.01377762234*sin_q4 - 0.003229235775) 
        Mq[:,1,5] = m26 = 0.000138534912*cos_q5 
        Mq[:,2,0] = m31 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23 
        Mq[:,2,1] = m32 = 0.7014282607*cos_q3 - 0.1166034794*sin_q4 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.644094696 
        Mq[:,2,2] = m33 = 0.02755524467*cos_q4*sin_q5 - 0.1166034794*sin_q4 - 0.001803200269*cos_q5**2 + 0.644094696 
        Mq[:,2,3] = m34 = 0.01377762234*cos_q4*sin_q5 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 + 0.01612863983 
        Mq[:,2,4] = m35 = cos_q5*(0.01377762234*sin_q4 - 0.003229235775) 
        Mq[:,2,5] = m36 = 0.000138534912*cos_q5 
        Mq[:,3,0] = m41 = 0.01615783641*cos_q234 - 0.0004508000672*cos_q234_2q5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 
        Mq[:,3,1] = m42 = 0.01377762234*cos_q4*sin_q5 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.01612863983 
        Mq[:,3,2] = m43 = 0.01377762234*cos_q4*sin_q5 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 + 0.01612863983 
        Mq[:,3,3] = m44 = 0.001803200269*sin_q5**2 + 0.01432543956 
        Mq[:,3,4] = m45 = -0.003229235775*cos_q5 
        Mq[:,3,5] = m46 = 0.000138534912*cos_q5 
        Mq[:,4,0] = m51 = 0.006888811168*sin_q23neg5 - 0.006888811168*sin_q235 - 0.002840479501*cos_q234 - 0.0002987944852*cos_q5neg3neg4neg2 - 0.00352803026*cos_q2345 - 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5
        Mq[:,4,1] = m52 = cos_q5*(0.01492110566*sin_q34 + 0.01377762234*sin_q4 - 0.003229235775) 
        Mq[:,4,2] = m53 = cos_q5*(0.01377762234*sin_q4 - 0.003229235775) 
        Mq[:,4,3] = m54 = -0.003229235775*cos_q5 
        Mq[:,4,4] = m55 = 0.002840479501 
        Mq[:,4,5] = m56 = 0 
        Mq[:,5,0] = m61 = -0.000138534912*sin_q234*sin_q5 
        Mq[:,5,1] = m62 = 0.000138534912*cos_q5 
        Mq[:,5,2] = m63 = 0.000138534912*cos_q5 
        Mq[:,5,3] = m64 = 0.000138534912*cos_q5 
        Mq[:,5,4] = m65 = 0 
        Mq[:,5,5] = m66 = 0.000138534912 
            
        return Mq
    
    def compute_reflected_mass(self, q, u):
        """ Computes the reflected mass of the UR5 manipulator
        along some direction u
    
        Args:
            q (6x1 torch tensor): joint angles
            u (6x1 torch tensor): direction vector
    
        Returns:
            mu (float): reflected mass along u
        """
        
        
        #horizon = q.shape[1]
        start = time.time()
        J = self.compute_analytical_jacobian(q)
        #J = torch.eye(6 , dtype=torch.float32).to('cuda')
        #J = J.repeat(horizon, 1, 1)
        end = time.time() 
        print("Time to create Jacobian: ", end - start)
        start = time.time()
        Mq = self.compute_inertia_matrix(q)
        end = time.time()
        print("Time to compute inertia matrix: ", end - start)
        J_T = J.permute(0,2,1)
        start = time.time()
        M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:,:3,:3]
        u_T = u.permute(0,2,1)
        mu = 1/(torch.matmul(u_T, torch.matmul(M_x_inv, u)).squeeze())
        end = time.time()
        print("Time to compute reflected mass: ", end - start)
        return mu
    
    def compute_reflected_mass_with_precompiled_jacobian(self, q, u):
        """ Computes the reflected mass of the UR5 manipulator
        along some direction u
    
        Args:
            q (6x1 torch tensor): joint angles
            u (6x1 torch tensor): direction vector
    
        Returns:
            mu (float): reflected mass along u
        """
        
        
        #horizon = q.shape[1]
        start = time.time()
        J = self.compute_precompiled_analytical_jacobian(q)
        #J = torch.eye(6 , dtype=torch.float32).to('cuda')
        #J = J.repeat(horizon, 1, 1)
        end = time.time() 
        print("Time to create Jacobian: ", end - start)
        start = time.time()
        Mq = self.compute_inertia_matrix(q)
        end = time.time()
        print("Time to compute inertia matrix: ", end - start)
        J_T = J.permute(0,2,1)
        start = time.time()
        M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:,:3,:3]
        u_T = u.permute(0,2,1)
        mu = 1/(torch.matmul(u_T, torch.matmul(M_x_inv, u)).squeeze())
        end = time.time()
        print("Time to compute reflected mass: ", end - start)
        return mu
    
    def compute_precompiled_reflected_mass(self, q, u):
        """ Computes the reflected mass of the UR5 manipulator
        along some direction u
    
        Args:
            q (6x1 torch tensor): joint angles
            u (6x1 torch tensor): direction vector
    
        Returns:
            mu (float): reflected mass along u
        """
        return self.aot_compute_reflected_mass(q, u)
    
    def compute_precompiled_reflected_mass_with_precompiled_jacobian(self, q, u):
        """ Computes the reflected mass of the UR5 manipulator
        along some direction u
    
        Args:
            q (6x1 torch tensor): joint angles
            u (6x1 torch tensor): direction vector
    
        Returns:
            mu (float): reflected mass along u
        """
        return self.aot_compute_reflected_mass_with_precompiled_jacobian(q, u)
    
        
        
        