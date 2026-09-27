import torch


def combine_profile_and_scenario(
    profile_tensor: torch.Tensor,
    scenario_matrix,
) -> torch.Tensor:

    scenario_dense = scenario_matrix.toarray()

    scenario_tensor = torch.tensor(
        scenario_dense,
        dtype=torch.float32
    )

    profile_tensor = profile_tensor.unsqueeze(0)

    return torch.cat(
        [profile_tensor, scenario_tensor],
        dim=1
    )