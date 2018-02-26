import numpy as np
from numpy import linalg as la
"""
Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.
"""


class TurbulenceKEpsDataProcessor:
    """
    Inherits from DataProcessor class.  This class is specific to processing turbulence data to predict
    the anisotropy tensor based on the mean strain rate (Sij) and mean rotation rate (Rij) tensors
    """
    @staticmethod
    def calc_Sij_Rij(grad_u, tke, eps, cap=7.):
        """
        Calculates the strain rate and rotation rate tensors.  Normalizes by k and eps:
        Sij = k/eps * 0.5* (grad_u  + grad_u^T)
        Rij = k/eps * 0.5* (grad_u  - grad_u^T)
        :param grad_u: num_points X 3 X 3
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate epsilon
        :param cap: This is the max magnitude that Sij or Rij components are allowed.  Greater values
                    are capped at this level
        :return: Sij, Rij: num_points X 3 X 3 tensors
        """

        num_points = grad_u.shape[0]
        eps = np.maximum(eps, 1e-8)
        tke_eps = tke / eps
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        for i in xrange(num_points):
            Sij[i, :, :] = tke_eps[i] * 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            Rij[i, :, :] = tke_eps[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        Sij[Sij > cap] = cap
        Sij[Sij < -cap] = -cap
        Rij[Rij > cap] = cap
        Rij[Rij < -cap] = -cap

        # Because we enforced limits on maximum Sij values, we need to re-enforce trace of 0
        for i in range(num_points):
            Sij[i, :, :] = Sij[i, :, :] - 1./3. * np.eye(3)*np.trace(Sij[i, :, :])
        return Sij, Rij

    def calc_scalar_basis(self, Sij, Rij, cap=2.0):
        """
        Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
        this returns a set of normalized scalar invariants
        :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
        :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3) * 2.0
        >>> B[0, 1, 0] = 1.0
        >>> B[0, 0, 1] = -1.0
        >>> tdp = TurbulenceKEpsDataProcessor()
        >>> tdp.mu = 0
        >>> tdp.std = 0
        >>> scalar_basis = tdp.calc_scalar_basis(A, B, is_scale=False)
        >>> print scalar_basis
        [[ 12.  -2.  24.  -4.  -8.]]
        """
        #DataProcessor.calc_scalar_basis(self, Sij, is_train=is_train)
        
        num_points = Sij.shape[0]
        num_invariants = 5
        invariants = np.zeros((num_points, num_invariants))
        for i in range(num_points):
            invariants[i, 0] = np.trace(np.dot(Sij[i, :, :], Sij[i, :, :]))
            invariants[i, 1] = np.trace(np.dot(Rij[i, :, :], Rij[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(Sij[i, :, :], np.dot(Sij[i, :, :], Sij[i, :, :])))
            invariants[i, 3] = np.trace(np.dot(Rij[i, :, :], np.dot(Rij[i, :, :], Sij[i, :, :])))
            invariants[i, 4] = np.trace(np.dot(np.dot(Rij[i, :, :], Rij[i, :, :]), np.dot(Sij[i, :, :], Sij[i, :, :])))

        return invariants

    def calc_tensor_basis(self, Sij, Rij, quadratic_only=False, is_scale=True):
        """
        Given Sij and Rij, it calculates the tensor basis
        :param Sij: normalized strain rate tensor
        :param Rij: normalized rotation rate tensor
        :param quadratic_only: True if only linear and quadratic terms are desired.  False if full basis is desired.
        :return: T_flat: num_points X num_tensor_basis X 9 numpy array of tensor basis.
                        Ordering is 11, 12, 13, 21, 22, ...
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, 1, 0] = 3.0
        >>> B[0, 0, 1] = -3.0
        >>> tdp = TurbulenceKEpsDataProcessor()
        >>> tb = tdp.calc_tensor_basis(A, B, is_scale=False)
        >>> print tb[0, :, :]
        [[  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -3.   0.   0.   0.  -3.   0.   0.   0.   6.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]
         [ -6.   0.   0.   0.  -6.   0.   0.   0.  12.]
         [  0.   0.   0.   0.   0.   0.   0.   0.   0.]]
        """
        num_points = Sij.shape[0]
        if not quadratic_only:
            num_tensor_basis = 10
        else:
            num_tensor_basis = 4
        T = np.zeros((num_points, num_tensor_basis, 3, 3))
        for i in range(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            T[i, 0, :, :] = sij
            T[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
            T[i, 2, :, :] = np.dot(sij, sij) - 1./3.*np.eye(3)*np.trace(np.dot(sij, sij))
            T[i, 3, :, :] = np.dot(rij, rij) - 1./3.*np.eye(3)*np.trace(np.dot(rij, rij))
            if not quadratic_only:
                T[i, 4, :, :] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
                T[i, 5, :, :] = np.dot(rij, np.dot(rij, sij)) \
                                + np.dot(sij, np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(sij, np.dot(rij, rij)))
                T[i, 6, :, :] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
                T[i, 7, :, :] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
                T[i, 8, :, :] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) \
                                + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                                - 2./3.*np.eye(3)*np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
                T[i, 9, :, :] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                                - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))
            # Enforce zero trace for anisotropy
            for j in range(num_tensor_basis):
                T[i, j, :, :] = T[i, j, :, :] - 1./3.*np.eye(3)*np.trace(T[i, j, :, :])

        # Scale down to promote convergence
        if is_scale:
            scale_factor = [10, 100, 100, 100, 1000, 1000, 10000, 10000, 10000, 10000]
            for i in range(num_tensor_basis):
                T[:, i, :, :] /= scale_factor[i]

        # Flatten:
        T_flat = np.zeros((num_points, num_tensor_basis, 9))
        for i in range(3):
            for j in range(3):
                T_flat[:, :, 3*i+j] = T[:, :, i, j]
        return T_flat

    def calc_output(self, stresses):
        """
        Given Reynolds stress tensor (num_points X 3 X 3), return flattened non-dimensional anisotropy tensor
        :param stresses: Reynolds stress tensor
        :return: anisotropy_flat: (num_points X 9) anisotropy tensor.  aij = (uiuj)/2k - 1./3. * delta_ij
        """
        num_points = stresses.shape[0]
        anisotropy = np.zeros((num_points, 3, 3))

        for i in range(3):
            for j in range(3):
                tke = 0.5 * (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])
                tke = np.maximum(tke, 1e-8)
                anisotropy[:, i, j] = stresses[:, i, j]/(2.0 * tke)
            anisotropy[:, i, i] -= 1./3.
        anisotropy_flat = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                anisotropy_flat[:, 3*i+j] = anisotropy[:, i, j]
        return (anisotropy_flat,tke)

    @staticmethod
    def calc_rans_anisotropy(grad_u, tke, eps):
        """
        Calculate the Reynolds stress anisotropy tensor (num_points X 9) that RANS would have predicted
        given a linear eddy viscosity hypothesis: a_ij = -2*nu_t*Sij/(2*k) = - C_mu * k / eps * Sij
        :param grad_u: velocity gradient tensor
        :param tke: turbulent kinetic energy
        :param eps: turbulent dissipation rate
        :return: rans_anisotropy
        """
        sij, _ = TurbulenceKEpsDataProcessor.calc_Sij_Rij(grad_u, tke, eps, cap=np.infty)
        c_mu = 0.09

        # Calculate anisotropy tensor (num_points X 3 X 3)
        # Note: Sij is already non-dimensionalized with tke/eps
        rans_anisotropy_matrix = - c_mu * sij

        # Flatten into num_points X 9 array
        num_points = sij.shape[0]
        rans_anisotropy = np.zeros((num_points, 9))
        for i in xrange(3):
            for j in xrange(3):
                rans_anisotropy[:, i*3+j] = rans_anisotropy_matrix[:, i, j]
        return rans_anisotropy

    @staticmethod
    def make_realizable(labels):
        """
        This function is specific to turbulence modeling.
        Given the anisotropy tensor, this function forces realizability
        by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
        Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
        constraints from first step, so this function should be called iteratively to get convergence
        to a realizable state.
        :param labels: the predicted anisotropy tensor (num_points X 9 array)
        """
        numPoints = labels.shape[0]
        A = np.zeros((3, 3))
        for i in range(numPoints):
            # Scales all on-diags to retain zero trace
            if np.min(labels[i, [0, 4, 8]]) < -1./3.:
                labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
            if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
                labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
                labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
                labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
                labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
                labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
                labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

            # Enforce positive semidefinite by pushing evalues to non-negative
            A[0, 0] = labels[i, 0]
            A[1, 1] = labels[i, 4]
            A[2, 2] = labels[i, 8]
            A[0, 1] = labels[i, 1]
            A[1, 0] = labels[i, 1]
            A[1, 2] = labels[i, 5]
            A[2, 1] = labels[i, 5]
            A[0, 2] = labels[i, 2]
            A[2, 0] = labels[i, 2]
            evalues, evectors = np.linalg.eig(A)
            if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
                evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]
            if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
                evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
                A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
                for j in range(3):
                    labels[i, j] = A[j, j]
                labels[i, 1] = A[0, 1]
                labels[i, 5] = A[1, 2]
                labels[i, 2] = A[0, 2]
                labels[i, 3] = A[0, 1]
                labels[i, 7] = A[1, 2]
                labels[i, 6] = A[0, 2]

        return labels
    
    def calc_tke_input(self, Sij, Rij):
        "tke prediction input preparations"
        
        num_points = Sij.shape[0]
        num_tensor_basis = 6
        
        I = np.zeros((num_points, num_tensor_basis))
        
        for i in range(num_points):
            sij = Sij[i, :, :]
            rij = Rij[i, :, :]
            #I[i, 0] = np.trace(sij)
            I[i, 0] = np.trace(np.dot(sij, sij))
            I[i, 1] = np.trace(np.dot(sij, np.dot(sij, sij)))
            I[i, 2] = np.trace(np.dot(rij, rij))
            I[i, 3] = np.trace(np.dot(np.dot(rij, rij), sij))
            I[i, 4] = np.trace(np.dot(np.dot(rij, rij), np.dot(sij,sij)))
            tmp1    = np.dot(rij, rij)
            tmp2    = np.dot(tmp1,sij)
            tmp3    = np.dot(tmp2,rij)
            tmp4    = np.dot(tmp3,np.dot(sij,sij)) 
            I[i, 5] = np.trace(tmp4)
            
            
        # Enforce zero trace for anisotropy if required

        return I
    

    def calc_piml_basis(self, k, ep, grad_u, grad_p, grad_k, vel, cap=7.):
        
        "calc piml basis invariant (47 nos)"

        num_points = grad_u.shape[0]
        
        Sij = np.zeros((num_points, 3, 3))
        Rij = np.zeros((num_points, 3, 3))
        
        for i in xrange(num_points):
            Sij[i, :, :] = 0.5 * (grad_u[i, :, :] + np.transpose(grad_u[i, :, :]))
            Rij[i, :, :] = 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))

        Sij[Sij > cap] = cap
        Sij[Sij < -cap] = -cap
        Rij[Rij > cap] = cap
        Rij[Rij < -cap] = -cap
       
        Ap = np.zeros((num_points, 3, 3))
        Ak = np.zeros((num_points, 3, 3))        

        for i in xrange(num_points):
            Ap[i, :, :] = np.cross(-np.eye(3),grad_p[i, :])
            Ak[i, :, :] = np.cross(-np.eye(3),grad_k[i, :])

        k = np.maximum(k, 1e-8)
        ep_k = ep / k 
        sqk  = np.sqrt(k)
        ep_sqk= ep / sqk
        
        udu  = np.zeros((num_points))
        tmp = np.zeros((3, 3))
        for i in xrange(num_points):        
            tmp[:,0]=grad_u[i,:,0]*vel[i,0]
            tmp[:,1]=grad_u[i,:,1]*vel[i,1]
            tmp[:,2]=grad_u[i,:,2]*vel[i,2]
            udu[i]  = la.norm(tmp)

        #normalize
        for i in xrange(num_points):
            Sij[i, :, :]=Sij[i,:,:]/(la.norm(Sij[i,:,:]) + ep_k[i])
            Rij[i, :, :]=Rij[i,:,:]/(la.norm(Rij[i,:,:]) + 1.0e-8)
            Ap[i, :, :] = Ap[i, :, :] /(la.norm(Ap[i, :, :]) + udu[i])
            Ak[i, :, :] = Ak[i, :, :] /(la.norm(Ak[i, :, :]) + ep_sqk[i])
            
        #prep invariants
        B=np.zeros((num_points, 47))
        
        for i in xrange(num_points):
            s   = Sij[i, :, :]
            r   = Rij[i, :, :]
            ap  = Ap[i, :, :] 
            ak  = Ak[i, :, :]
            
            s2  = np.dot(s,s)
            r2  = np.dot(r,r)
            ap2 = np.dot(ap,ap)
            ak2 = np.dot(ak,ak)
            
            B[i,0] = np.trace(s2)
            B[i,1] = np.trace(np.dot(s2,s))    
            B[i,2] = np.trace(r2)
            B[i,3] = np.trace(ap2)
            B[i,4] = np.trace(ak2)
            B[i,5] = np.trace(np.dot(r2,s))
            B[i,6] = np.trace(np.dot(r2,s2))    
            r2s    = np.dot(r2,s)
            r2sr   = np.dot(r2s,r)
            B[i,7] = np.trace(np.dot(r2sr,s2))
            B[i,8] = np.trace(np.dot(ap2,s))
            B[i,9] = np.trace(np.dot(ap2,s2))            
            ap2s   = np.dot(ap2,s)
            ap2sap = np.dot(ap2s,ap)
            B[i,10] = np.trace(np.dot(ap2sap,s2))             
            B[i,11] = np.trace(np.dot(ak2,s))
            B[i,12] = np.trace(np.dot(ak2,s2))             
            ak2s   = np.dot(ak2,s)
            ak2sak = np.dot(ak2s,ak)
            B[i,13] = np.trace(np.dot(ak2sak,s2))             
            B[i,14] = np.trace(np.dot(r,ap))             
            B[i,15] = np.trace(np.dot(ap,ak))              
            B[i,16] = np.trace(np.dot(r,ak)) 
             
            B[i,17] = np.trace(np.dot(np.dot(r,ap),s))               
            B[i,18] = np.trace(np.dot(np.dot(r,ap),s2))
            B[i,19] = np.trace(np.dot(np.dot(r2,ap),s))              
            B[i,20] = np.trace(np.dot(np.dot(ap2,r),s)) 
            B[i,21] = np.trace(np.dot(np.dot(r2,ap),s2))              
            B[i,22] = np.trace(np.dot(np.dot(ap2,r),s2))             
            B[i,23] = np.trace(np.dot(np.dot(np.dot(r2,s),ap),s2))  
            B[i,24] = np.trace(np.dot(np.dot(np.dot(ap2,s),r),s2)) 
            
            B[i,25] = np.trace(np.dot(np.dot(r,ak),s))               
            B[i,26] = np.trace(np.dot(np.dot(r,ak),s2))
            B[i,27] = np.trace(np.dot(np.dot(r2,ak),s))              
            B[i,28] = np.trace(np.dot(np.dot(ak2,r),s)) 
            B[i,29] = np.trace(np.dot(np.dot(r2,ak),s2))              
            B[i,30] = np.trace(np.dot(np.dot(ak2,r),s2))             
            B[i,31] = np.trace(np.dot(np.dot(np.dot(r2,s),ak),s2))  
            B[i,32] = np.trace(np.dot(np.dot(np.dot(ak2,s),r),s2))             

            B[i,33] = np.trace(np.dot(np.dot(ap,ak),s))               
            B[i,34] = np.trace(np.dot(np.dot(ap,ak),s2))
            B[i,35] = np.trace(np.dot(np.dot(ap2,ak),s))              
            B[i,36] = np.trace(np.dot(np.dot(ak2,ap),s)) 
            B[i,37] = np.trace(np.dot(np.dot(ap2,ak),s2))              
            B[i,38] = np.trace(np.dot(np.dot(ak2,ap),s2))             
            B[i,39] = np.trace(np.dot(np.dot(np.dot(ap2,s),ak),s2))  
            B[i,40] = np.trace(np.dot(np.dot(np.dot(ak2,s),ap),s2)) 

            B[i,41] = np.trace(np.dot(np.dot(r,ap),ak))               
            B[i,42] = np.trace(np.dot(np.dot(np.dot(r,ap),ak),s))              
            B[i,43] = np.trace(np.dot(np.dot(np.dot(r,ak),ap),s))              
            B[i,44] = np.trace(np.dot(np.dot(np.dot(r,ap),ak),s2))              
            B[i,45] = np.trace(np.dot(np.dot(np.dot(r,ak),ap),s2))              
            B[i,46] = np.trace(np.dot(np.dot(np.dot(np.dot(r,ap),s),ak),s2))              
  
    
        return B    
            
            
                    
            
            
            