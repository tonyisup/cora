from cora_infans.cli import parse_simulation_cli_args
from cora_infans.simulation import SimulationParams



def test_parse_simulation_cli_args_returns_typed_params_and_output_mode() -> None:
    params, output_mode = parse_simulation_cli_args(
        [
            "--simulate-frame",
            "--timestamp-ms",
            "250",
            "--dominant-frequency-hz",
            "350.0",
            "--channel-energies",
            "0.1,0.2,0.8,0.3",
            "--initial-prediction-hz",
            "300.0",
            "--saliency-threshold",
            "0.95",
            "--mic-geometry",
            "front_left,front_right,rear_right,rear_left",
            "--output",
            "pretty",
        ]
    )

    assert params == SimulationParams(
        timestamp_ms=250,
        dominant_frequency_hz=350.0,
        channel_energies=[0.1, 0.2, 0.8, 0.3],
        initial_prediction_hz=300.0,
        saliency_threshold=0.95,
        mic_channel_geometry=(
            "front_left",
            "front_right",
            "rear_right",
            "rear_left",
        ),
    )
    assert output_mode == "pretty"



def test_parse_simulation_cli_args_returns_defaults_when_only_flag_is_present() -> None:
    params, output_mode = parse_simulation_cli_args(["--simulate-frame"])

    assert params == SimulationParams()
    assert output_mode == "json"
