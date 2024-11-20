import torch
#from .extensions.oc_helper import oc_helper_matrices, select_with_default

from .extensions.oc_helper import oc_helper_matrices, select_with_default
from .torch_geometric_interface import strict_batch_from_row_splits

def arctanhsq(x):
    return torch.arctanh(x)**2


class ObjectCondensation(torch.nn.Module):

    def __init__(self,
                 q_min = 0.1,
                 s_B = 1.,
                 norm_payload = True,
                 weighted_obj_coordinates = 0.,
                 fixed_repulsive_norm = None,
                 beta_scaling_epsilon = 1e-3,
                 v_beta_scaling = arctanhsq,
                 p_beta_scaling = arctanhsq,
                 **kwargs) -> None:
        '''
        Initializes the ObjectCondensation loss module.
        
        This module implements the object condensation loss as described in the paper:
        "Object condensation: one-stage grid-free multi-object reconstruction
        in physics detectors, graph and image data" [arXiv:2002.03605].
        
        Parameters:
            q_min (float): 
                Minimum charge for object condensation potentials. 
                
            s_B (float): 
                Scaling factor for the noise term of the beta potential.
        
            norm_payload (bool): 
                Whether to normalize the payload loss per object.
                If True, the payload loss is divided by the total object contribution
                to normalize it.
        
            weighted_obj_coordinates (float): 
                Weighting factor for object coordinates. 
                Defines the ratio of object-average coordinates to alpha-selected coordinates.
                Must be in the range [0, 1]. For non-zero values, this feature goes beyond
                what is described in the paper.
        
            fixed_repulsive_norm (float, optional): 
                Fixed normalization value for the repulsive loss. 
                If None, the normalization will be calculated dynamically based on 
                the number of points per object.
        
            beta_scaling_epsilon (float): 
                Small epsilon value to stabilize beta scaling calculations.
        
            v_beta_scaling (callable): 
                Function to scale the beta values for the potential loss.
                Default is `arctanhsq`, which computes the square of the hyperbolic arctangent.
        
            p_beta_scaling (callable): 
                Function to scale the beta values for the payload loss.
                Default is `arctanhsq`, which computes the square of the hyperbolic arctangent.
        
            **kwargs: 
                Additional keyword arguments passed to the parent `torch.nn.Module`.
        
        Raises:
            AssertionError: 
                If `weighted_obj_coordinates` is not in the range [0, 1].
        '''
        self.q_min = q_min
        self.s_B = s_B
        self.beta_scaling_epsilon = beta_scaling_epsilon
        self.v_beta_scaling = v_beta_scaling
        self.p_beta_scaling = p_beta_scaling

        self.pl_norm = self._no_op if not norm_payload else self._norm_payload
        self.rep_norm = self._norm_repulsive_fixed if fixed_repulsive_norm is not None else self._norm_repulsive
        self.fixed_repulsive_norm = fixed_repulsive_norm

        assert 0. <= weighted_obj_coordinates <= 1.
        self.weighted_obj_coordinates = weighted_obj_coordinates

        super().__init__(**kwargs)

    def _no_op(self, x, *args):
        return x

    def _norm_payload(self, payload_loss_k_m, M):
        return payload_loss_k_m / (torch.sum(payload_loss_k_m, dim=1, keepdim=True) + 1e-12)

    def _norm_repulsive_fixed(self, rep_loss, N_k):
        # rep_loss as K x 1, N_prs as K x 1
        return rep_loss / self.fixed_repulsive_norm

    def _norm_repulsive(self, rep_loss, N_k):
        return rep_loss / N_k
    
    def _scatter_to_N_indices(self, x_k_m, asso_indices, M):
        '''
        Inputs: 
            x_k_1 (torch.Tensor): The values to scatter, shape (K, M, 1)
        '''
        # Step 1: Use select_with_default to get indices
        M2 = select_with_default(M, torch.arange(asso_indices.size(0)).view(-1, 1), -1)
        
        # Step 3: Flatten valid entries in M2
        valid_mask = M2 >= 0
        n_flat = x_k_m[valid_mask].view(-1)  # Flatten valid points
        m_flat = M2[valid_mask].view(-1)  # Flatten corresponding indices

        assert torch.max(m_flat) < asso_indices.size(0), "m_flat contains out-of-bounds indices."
        
        # Step 4: Scatter the values back to all points
        x = torch.zeros_like(asso_indices, dtype=x_k_m.dtype)
        x[m_flat] = n_flat
        return x
    
    def _mean_per_row_split(self, x, row_splits):
        '''
        x : N x 1
        row_splits : N_rs
        '''
        # Calculate lengths of each row split
        lengths = row_splits[1:] - row_splits[:-1]
        
        # Create indices for each row split
        row_indices = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        
        # Calculate sum per row split
        sum_per_split = torch.zeros(len(lengths), dtype=torch.float32).scatter_add(0, row_indices, x.squeeze(1))
        return sum_per_split / lengths
    
    def _beta_loss(self, beta_k_m):
        """
        Calculate the beta penalty using a continuous max approximation via LogSumExp
        and an additional penalty term for faster convergence.
    
        Args:
            beta_k_m (torch.Tensor): Tensor of shape (K, M, 1) containing the beta values.
    
        Returns:
            torch.Tensor: Tensor of shape (K, 1) containing the beta penalties.
        """
        eps = 1e-3
        # Continuous max approximation using LogSumExp
        beta_pen = 1. - eps * torch.logsumexp(beta_k_m / eps, dim=1)  # Sum over M
    
        # Add penalty for faster convergence
        beta_pen += 1. - torch.clamp(torch.sum(beta_k_m, dim=1), min=0., max=1.)
        
        return beta_pen


    def get_alpha_indices(self, beta_k_m, M):
        '''
        Calculates the arg max of beta_k_m in dimension 1.
        '''
        m_idxs = torch.argmax(beta_k_m, dim=1).squeeze(1)
        return M[torch.arange(M.size(0)), m_idxs]

    def V_repulsive_func(self, distsq):
        '''
        Calculates the repulsive potential function.
        It is in a dedicated function to allow for easy replacement in inherited classes.
        '''
        return torch.relu(1. - torch.sqrt(distsq + 1e-6)) #hinge
    
    def V_attractive_func(self, distsq):
        '''
        Calculates the attractive potential function.
        It is in a dedicated function to allow for easy replacement in inherited classes.
        '''
        return distsq
    
    def alpha_features(self, x, x_k_m, alpha_indices):
        '''
        Returns the features of the alphas.
        In other implementations, this can be a weighted function.

        Returns K, 1, F where F is the number of features.
        '''
        x_a = (1. - self.weighted_obj_coordinates ) * x[alpha_indices]
        x_a = x_a + self.weighted_obj_coordinates * torch.mean(x_k_m, dim=1)
        return x_a.view(-1, 1, x.size(1))

    def forward(self, beta, coords, asso_idx, row_splits):
        '''
        Inputs:
            beta (torch.Tensor): The beta values for each object, shape (N, 1)
            coords (torch.Tensor): The cluster coordinates of the objects, shape (N, C)
            asso_idx (torch.Tensor): The association indices of the objects, shape (N, 1)
                                        By convention, noise is marked by and index of -1.
                                        All objects have indices >= 0.
            row_splits (torch.Tensor): The row splits tensor, shape (N_rs)
        '''
        #check inputs
        assert beta.dim() == 2 and beta.size(1) == 1
        assert coords.dim() == 2 and coords.size(1) >= 1
        assert asso_idx.dim() == 2 and asso_idx.size(1) == 1

        asso_idx = asso_idx.squeeze(1)

        # get the matrices, row splits will be encoded in M and M_not
        # and are not needed after this
        M, M_not, obj_per_rs = oc_helper_matrices(asso_idx, row_splits) # M is (K, M), M_not is (K, N_prs)

        # Use repeat_interleave to assign batch indices
        batch_idx = strict_batch_from_row_splits(row_splits)
        K_k = select_with_default(M, obj_per_rs[batch_idx].view(-1,1), 1)[:,0] #for normalisation, (K, 1)

        beta_scale = self.v_beta_scaling(beta / (1. + self.beta_scaling_epsilon)) + self.q_min # K x 1

        # get the potential loss
        beta_scale_k_m = select_with_default(M, beta_scale, 0.) # K x M x 1
        beta_k_m = select_with_default(M, beta, 0.) # K x M x 1

        # mask
        mask_k_m = select_with_default(M, torch.ones_like(beta_scale), 0.) # K x M x 1

        # get argmax in dim 1 of beta_scale_k_m
        alpha_indices = self.get_alpha_indices(beta_scale_k_m, M)

        # get the coordinates of the alphas
        coords_k_m = select_with_default(M, coords, 0.)
        coords_k = self.alpha_features(coords, coords_k_m, alpha_indices)

        beta_scale_k = beta_scale[alpha_indices] # K x 1

        ## Attractive potential
        # get the distances
        distsq_k_m = torch.sum((coords_k - coords_k_m)**2, dim=2, keepdim=True) # K x M x 1
        
        # get the attractive potential
        V_attractive = mask_k_m * beta_scale_k.view(-1,1,1) * self.V_attractive_func(distsq_k_m) * beta_scale_k_m # K x M x 1

        # mean over V' and mean over K
        L_V_k = torch.sum(V_attractive, dim=1) / (torch.sum(mask_k_m, dim=1) + 1e-6) # K x 1
        L_V = torch.sum(L_V_k / K_k ) # scalar

        ## Repulsive potential
        coords_k_n = select_with_default(M_not, coords, 0.)
        beta_scale_k_n = select_with_default(M_not, beta_scale, 0.)

        mask_k_n = select_with_default(M_not, torch.ones_like(beta_scale), 0.)

        # get the distances
        distsq_k_n = torch.sum((coords_k_n - coords_k)**2, dim=2, keepdim=True)  # K x N' x 1
        V_repulsive = mask_k_n * beta_scale_k.view(-1,1,1) * self.V_repulsive_func(distsq_k_n) * beta_scale_k_n # K x N' x 1

        # mean over N and mean over K
        L_rep_k = self.rep_norm(torch.sum(V_repulsive, dim=1), torch.sum(mask_k_n, dim=1) + 1e-6) # K x 1
        L_rep = torch.sum(L_rep_k / K_k) # scalar

        ## Payload scaling
        pl_scaling = self.p_beta_scaling(beta / (1. + self.beta_scaling_epsilon))
        pl_scaling_k_m = select_with_default(M, pl_scaling, 0.) # K x M x 1
        pl_scaling_k_m = self.pl_norm(pl_scaling_k_m, M) # K x M x 1

        #scatter back V = V_attractive + V_repulsive and pl_scaling to (N, 1)

        L_k_m = (L_V_k + L_rep_k).view(-1, 1, 1).repeat(1, M.size(1), 1)
        L_V_rep = self._scatter_to_N_indices(L_k_m, asso_idx, M) # N x 1

        pl_scaling = self._scatter_to_N_indices(pl_scaling_k_m, asso_idx, M) # N x 1

        # create the beta loss
        L_b_k = self._beta_loss(beta_k_m) # K x 1
        L_b = torch.sum(L_b_k / K_k) 

        # add noise term
        L_noise = self.s_B *  beta * (asso_idx < 0).view(-1,1)  #norm per rs here too
        L_noise = torch.mean(self._mean_per_row_split(L_noise, row_splits))

        L_b = L_b + L_noise

        return L_V, L_rep, L_b, pl_scaling, L_V_rep

