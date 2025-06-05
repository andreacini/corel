from lib.nn.encoders.stgnn_encoder import STGNNEncoder
from lib.nn.layers.knn_graph_learning import DifferentiableKnnGraphLayer

class CoRelEncoder(STGNNEncoder):
    def __init__(self,
                 input_size: int,
                 n_instances: int,
                 n_neighbors: int,
                 hidden_size: int,
                 emb_size: int,
                 temporal_layers: int,
                 gnn_layers: int,
                 exog_size: int = 0,
                 conv_type: str = "iso",
                 activation: str = 'elu',
                 cat_emb: bool = True,
                 sparsify_gradient: bool = True,
                 at_most_k: bool = True,
                 dropout_emb: float = 0.0,):
        super(CoRelEncoder, self).__init__(
            input_size=input_size,
            n_instances=n_instances,
            hidden_size=hidden_size,
            emb_size=emb_size,
            temporal_layers=temporal_layers,
            gnn_layers=gnn_layers,
            exog_size=exog_size,
            conv_type=conv_type,
            activation=activation,
            cat_emb=cat_emb,
            dropout_emb=dropout_emb,
        )
        self.graph_learning_module = DifferentiableKnnGraphLayer(
            n_nodes=n_instances,
            k=n_neighbors,
            tau=1,
            sparsify_gradient=sparsify_gradient,
            at_most_k=at_most_k
        )

    def forward(self, x, u=None, **kwargs):
        """"""
        if self.emb is not None:
            emb = self.emb()
        else:
            emb = None
        edge_index, edge_weight = self.graph_learning_module(x, emb)
        disjoint_mode = self.training and not self.graph_learning_module.inference_mode
        x = super(CoRelEncoder, self).forward(x,
                                              u=u,
                                              edge_index=edge_index,
                                              edge_weight=edge_weight,
                                              disjoint=disjoint_mode)
        return x
