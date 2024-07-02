import torch


class ParameterMapper(torch.nn.Module):
    """
    Wrapper for the parameter mapping model
    to be used within the TorchDrum Plugin.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        damp: torch.Tensor,
        patch: torch.Tensor,
        param_project: torch.nn.Module = torch.nn.Identity(),
    ):
        super().__init__()
        self.model = model
        self.register_buffer("patch", patch)
        self.register_buffer("damp", damp)
        self.projection = param_project

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_mod = self.model(self.projection(x))
        params = param_mod * self.damp
        params = torch.clip(params, -1.0, 1.0)
        params = torch.cat([params, self.patch], dim=0)
        return params
