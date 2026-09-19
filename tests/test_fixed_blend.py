from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from final_adsorption_model.model import ComponentSpec, FinalAdsorptionModel


def test_locked_weights_sum_to_one() -> None:
    project = Path(__file__).resolve().parents[1]
    model = FinalAdsorptionModel(project)
    assert [component.model_name for component in model.components] == [
        "LightGBMSingle10_BAG_L1",
        "WeightedEnsemble_L2",
    ]
    assert np.isclose(sum(component.weight for component in model.components), 1.0)
    assert np.isclose(model.components[0].weight, 0.626)
    assert np.isclose(model.components[1].weight, 0.374)

