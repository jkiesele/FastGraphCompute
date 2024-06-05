from typing import Optional, Tuple
import torch


def select_knn(
    x: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    inmask: Optional[torch.Tensor] = None,
    max_radius: float = 1e9,
    mask_mode: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Finds for each element in :obj:`x` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        max_radius (float): Maximum distance to nearest neighbours. (default: :obj:`1e9`)
        mask_mode (int): ??? (default: :obj:`1`)

    :rtype: :class:`Tuple`[`LongTensor`,`FloatTensor`]

    .. code-block:: python

        import torch
        from torch_cmspepr import select_knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        assign_index = select_knn(x, 2, batch_x)
    """
    x = x.view(-1, 1) if x.dim() == 1 else x
    x = x.contiguous()

    mask: torch.Tensor = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
    if inmask is not None:
        mask = inmask

    # Compute row_splits
    if batch_x is None:
        row_splits: torch.Tensor = torch.tensor(
            [0, x.shape[0]], dtype=torch.int32, device=x.device
        )
    else:
        assert x.size(0) == batch_x.size(0)
        batch_size = int(batch_x.max()) + 1

        # Get number of hits per event
        counts = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, batch_x, torch.ones_like(batch_x, dtype=torch.int32))

        # Convert counts to row_splits by using cumsum.
        # row_splits must start with 0 and end with x.size(0), and has length +1 w.r.t.
        # batch_size.
        # e.g. for 2 events with 5 and 4 hits, row_splits would be [0, 5, 9]
        row_splits = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
        torch.cumsum(counts, 0, out=row_splits[1:])

    if x.device == torch.device('cpu'):
        return torch.ops.select_knn_cpu.select_knn_cpu(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )
    else:
        return torch.ops.select_knn_cuda.select_knn_cuda(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )