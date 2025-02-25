import torch
from torch import nn
from torch.autograd import grad

from newtonnet.layers import Dense
from newtonnet.layers.shells import ShellProvider
from newtonnet.layers.scalers import ScaleShift, TrainableScaleShift
from newtonnet.layers.cutoff import CosineCutoff, PolynomialCutoff
from newtonnet.layers.representations import RadialBesselLayer
from newtonnet.layers.activations import get_activation_by_string


class NewtonNet(nn.Module):
    """
    Molecular Newtonian Message Passing

    Parameters
    ----------
    resolution: int
        number of radial functions to describe interatomic distances

    n_features: int
        number of neurons in the latent layer. This number will remain fixed in the entire network except
        for the last fully connected network that predicts atomic energies.

    activation: function
        activation function from newtonnet.layers.activations
        you can aslo get it by string from newtonnet.layers.activations.get_activation_by_string

    n_interactions: int, default: 3
        number of interaction blocks
    dropout
    max_z
    cutoff
    cutoff_network
    normalizer
    normalize_atomic
    requires_dr
    device
    create_graph
    shared_interactions
    return_latent
    """

    def __init__(
        self,
        device,
        activation="swish",
        resolution=20,
        n_features=128,
        n_interactions=3,
        dropout=0.0,
        max_z=10,
        cutoff=5.0,
        cutoff_network="poly",
        normalizer=(0.0, 1.0),
        normalize_atomic=False,
        requires_dr=False,
        create_graph=False,
        shared_interactions=False,
        return_latent=True,
        layer_norm=False,
        atomic_properties=False,
        pair_properties=False,
        double_update_latent=False,
        pbc=False,
        aggregation="sum",
    ):
        super(NewtonNet, self).__init__()

        self.requires_dr = requires_dr
        self.create_graph = create_graph
        self.return_intermediate = return_latent
        self.pbc = pbc

        shell_cutoff = None
        if pbc:
            # make the cutoff here a little bit larger so that it can be handled with differentiable cutoff layer in interaction block
            shell_cutoff = cutoff * 1.1

        self.shell = ShellProvider(
            return_vecs=True, normalize_vecs=True, pbc=pbc, cutoff=shell_cutoff
        )
        self.distance_expansion = RadialBesselLayer(resolution, cutoff, device=device)

        # atomic embedding
        self.n_features = n_features
        self.embedding = nn.Embedding(max_z, n_features, padding_idx=0)

        if type(activation) is str:
            activation = get_activation_by_string(activation)

        # d1 message
        self.n_interactions = n_interactions
        if shared_interactions:
            # use the same message instance (hence the same weights)
            self.dycalc = nn.ModuleList(
                [
                    DynamicsCalculator(
                        n_features=n_features,
                        resolution=resolution,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        double_update_latent=double_update_latent,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.dycalc = nn.ModuleList(
                [
                    DynamicsCalculator(
                        n_features=n_features,
                        resolution=resolution,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        double_update_latent=double_update_latent,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # layer norm
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nn.ModuleList(
                [nn.LayerNorm(n_features) for _ in range(n_interactions)]
            )

        # final dense network
        self.atomic_energy = AtomicProperty(n_features, activation, dropout)

        self.normalize_atomic = normalize_atomic
        if normalize_atomic:
            self.inverse_normalize = TrainableScaleShift(max_z)
        else:
            if type(normalizer) is dict:
                self.inverse_normalize = nn.ModuleDict(
                    {
                        str(atom_num): ScaleShift(
                            mean=torch.tensor(normalizer[atom_num][0], device=device),
                            stddev=torch.tensor(normalizer[atom_num][1], device=device),
                        )
                        for atom_num in normalizer
                    }
                )
            else:
                self.inverse_normalize = ScaleShift(
                    mean=torch.tensor(normalizer[0], device=device),
                    stddev=torch.tensor(normalizer[1], device=device),
                )

        # Optional Atomic Property prediction
        self.atomic_properties = atomic_properties
        if atomic_properties:
            self.atomic_property = AtomicProperty(n_features, activation, dropout)
        # Optional Atomic Pair Property prediction
        self.pair_properties = pair_properties
        if pair_properties:
            self.pair_property = PairProperty(n_features, activation)

        self.aggregation = aggregation

    def forward(self, data):
        Z = data["Z"]
        R = data["R"]
        N = data["N"]
        NM = data["NM"]
        AM = data["AM"]
        if "lattice" in data:
            lattice = data["lattice"]
        else:
            lattice = None

        # initiate main containers
        a = self.embedding(Z)  # B,A,nf
        f_dir = torch.zeros_like(R)  # B,A,3
        f_dynamics = torch.zeros(
            R.size() + (self.n_features,), device=R.device
        )  # B,A,3,nf
        r_dynamics = torch.zeros(
            R.size() + (self.n_features,), device=R.device
        )  # B,A,3,nf
        e_dynamics = torch.zeros_like(a)  # B,A,nf

        # require grad
        if self.requires_dr:
            R.requires_grad_()

        # store intermediate representations
        if self.return_intermediate:
            hs = [(a,)]

        # compute distances (B,A,N) and distance vectors (B,A,N,3)
        if "D" in data:
            distances = data["D"]
            distance_vector = data["V"]
        else:
            distances, distance_vector, N, NM = self.shell(R, N, NM, lattice)

        # comput d1 representation (B, A, N, G)
        rbf = self.distance_expansion(distances)

        # compute interaction block and update atomic embeddings
        for i_interax in range(self.n_interactions):
            # print('iter: ', i_interax)

            # messages
            a, msij, f_dir, f_dynamics, r_dynamics, e_dynamics = self.dycalc[i_interax](
                a,
                rbf,
                distances,
                distance_vector,
                N,
                NM,
                f_dir,
                f_dynamics,
                r_dynamics,
                e_dynamics,
            )

            if self.layer_norm:
                a = self.norm[i_interax](a)

            if self.return_intermediate:
                hs.append((a, f_dir, f_dynamics, r_dynamics, e_dynamics))

        # Output
        output = dict()

        if self.atomic_properties:
            Ai = self.atomic_property(a)  # B,A,1
            # if self.normalize_atomic:
            #     Ai = self.inverse_normalize(Ai, Z)
            # else:
            #     for atomic_type in self.inverse_normalize:
            #         atomic_filter = Z == int(atomic_type)
            #         Ai[atomic_filter] = self.inverse_normalize[atomic_type](Ai[atomic_filter])
            output["Ai"] = Ai.squeeze(-1)  # B,A

        if self.pair_properties:
            Pij = self.pair_property(msij)  # B,A,N,1
            output["Pij"] = Pij.squeeze(-1)

        Ei = self.atomic_energy(a)
        if self.normalize_atomic:
            Ei = self.inverse_normalize(Ei, Z)

        # inverse normalize
        Ei = Ei * AM[..., None]  # (B,A,1)
        if self.aggregation == "sum":
            E = torch.sum(Ei, 1)  # (B,1)
        elif self.aggregation == "mean":
            E = torch.mean(Ei, 1)
        elif self.aggregation == "max":
            E = torch.max(Ei, 1).values
        if not self.normalize_atomic:
            E = self.inverse_normalize(E)

        if self.requires_dr:
            dE = grad(
                E,
                R,
                grad_outputs=torch.ones_like(E),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            dE = -1.0 * dE

        else:
            dE = data["F"]

        if self.return_intermediate:
            output.update({"E": E, "F": dE, "Ei": Ei, "hs": hs, "F_latent": f_dir})
        else:
            return output.update({"E": E, "F": dE, "Ei": Ei, "F_latent": f_dir})

        return output


class DynamicsCalculator(nn.Module):
    def __init__(
        self,
        n_features,
        resolution,
        activation,
        cutoff,
        cutoff_network,
        double_update_latent=True,
        epsilon=1e-8,
    ):
        super(DynamicsCalculator, self).__init__()

        self.n_features = n_features
        self.epsilon = epsilon

        # non-directional message passing
        self.phi_rbf = Dense(resolution, n_features, activation=None)

        self.phi_a = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

        # cutoff layer used in interaction block
        if cutoff_network == "poly":
            self.cutoff_network = PolynomialCutoff(cutoff, p=9)
        elif cutoff_network == "cosine":
            self.cutoff_network = CosineCutoff(cutoff)

        # directional message passing
        self.phi_f = Dense(n_features, 1, activation=None, bias=False)
        self.phi_f_scale = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )
        self.phi_r = nn.Sequential(
            Dense(
                n_features, n_features, activation=activation, xavier_init_gain=0.001
            ),
            Dense(n_features, n_features, activation=None),
        )
        self.phi_r_ext = nn.Sequential(
            Dense(n_features, n_features, activation=activation, bias=False),
            Dense(n_features, n_features, activation=None, bias=False),
        )

        self.phi_e = nn.Sequential(
            Dense(n_features, n_features, activation=activation),
            Dense(n_features, n_features, activation=None),
        )

        self.double_update_latent = double_update_latent

    def gather_neighbors(self, inputs, N):
        n_features = inputs.size()[-1]
        n_dim = inputs.dim()
        b, a, n = N.size()  # batch, atoms, neighbors size

        if n_dim == 3:
            N = N.view(-1, a * n, 1)  # B,A*N,1
            N = N.expand(-1, -1, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, n_features)  # B,A,N,n_features

        elif n_dim == 4:
            N = N.view(-1, a * n, 1, 1)  # B,A*N,1,1
            N = N.expand(-1, -1, 3, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, 3, n_features)  # B,A,N,3,n_features

    def sum_neighbors(self, x, mask, dim=2, avg=False):
        """

        Parameters
        ----------
        x: torch.tensor
            usually of shape B,A,N,nf
        mask: torch.tensor
            usually of shape B,A,N
        dim: int
            the dimension to sum

        avg: bool
            if True, returns the average output by dividing the sum by number of neighbors.

        Returns
        -------
        out: torch.tensor
            usually of shape B,A

        """
        dim_diff = x.dim() - mask.dim()
        for _ in range(dim_diff):
            mask = mask.unsqueeze(-1)

        x = x * mask
        out = torch.sum(x, dim=dim)

        if avg:
            n_atoms = torch.sum(mask, dim)
            n_atoms = torch.max(n_atoms, other=torch.ones_like(n_atoms))
            out = out / n_atoms

        return out

    def forward(
        self,
        a,
        rbf,
        distances,
        distance_vector,
        N,
        NM,
        f_dir,
        f_dynamics,
        r_dynamics,
        e_dynamics,
    ):
        # map decomposed distances
        rbf_msij = self.phi_rbf(rbf)  # B,A,N,nf

        # cutoff
        C = self.cutoff_network(distances)
        rbf_msij = rbf_msij * C.unsqueeze(-1)

        # map atomic features
        a_msij = self.phi_a(a)  # B,A,3*nf

        # copy central atom features for the element-wise multiplication
        ai_msij = a_msij.repeat(1, 1, rbf_msij.size(2))
        ai_msij = ai_msij.view(rbf_msij.size())  # B,A,N,nf

        # look up neighboring atoms features based on the schnet contiuous filter implementation
        aj_msij = self.gather_neighbors(a_msij, N)  # B,A,N,nf

        # symmetric feature multiplication
        mij = rbf_msij * aj_msij
        msij = mij * ai_msij

        # update a with invariance
        if self.double_update_latent:
            a = a + self.sum_neighbors(msij, NM, dim=2)

        # Dynamics: Force Module
        F_ij = self.phi_f(msij) * distance_vector  # B,A,N,3
        F_i_dir = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3
        f_dir = f_dir + F_i_dir

        F_ij = self.phi_f_scale(msij).unsqueeze(-2) * F_ij.unsqueeze(-1)  # B,A,N,3,nf
        F_i = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3,nf

        f_dynamics = f_dynamics + F_i  # update

        # Dynamics: Displacement Module
        dr_i = self.phi_r(a).unsqueeze(-2) * F_i  # B,A,3,nf

        dr_j = self.gather_neighbors(r_dynamics, N)  # B,A,N,3,nf
        dr_j = self.phi_r_ext(msij).unsqueeze(-2) * dr_j  # B,A,N,3,nf
        dr_ext = self.sum_neighbors(dr_j, NM, dim=2, avg=False)  # B,A,3,nf

        r_dynamics = r_dynamics + dr_i + dr_ext  # update

        # Dynamics: Energy Module
        de_i = -1.0 * torch.sum(f_dynamics * r_dynamics, dim=-2)  # B,A,nf
        de_i = self.phi_e(a) * de_i
        a = a + de_i
        e_dynamics = e_dynamics + de_i

        return a, msij, f_dir, f_dynamics, r_dynamics, e_dynamics


class AtomicProperty(nn.Module):
    def __init__(self, n_features, activation, dropout):
        super(AtomicProperty, self).__init__()
        self.phi_a = nn.Sequential(
            Dense(n_features, 128, activation=activation, dropout=dropout, norm=False),
            Dense(128, 64, activation=activation, dropout=dropout, norm=False),
            Dense(64, 1, activation=None, dropout=0.0, norm=False),
        )

    def forward(self, a):
        return self.phi_a(a)


class PairProperty(nn.Module):
    def __init__(self, n_features, activation):
        super(PairProperty, self).__init__()
        self.phi_p = nn.Sequential(
            Dense(n_features, 128, activation=activation, bias=False),
            Dense(128, 64, activation=activation, bias=False),
            Dense(64, 1, activation=activation, bias=False),
        )

    def forward(self, msij):
        pij = self.phi_p(msij)  # B,A,N
        return pij
