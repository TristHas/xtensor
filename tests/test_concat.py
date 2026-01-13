import torch

from xtensor import DataTensor, Dataset, concat


def _datatensor(values, *, x_coords):
    data = torch.as_tensor(values, dtype=torch.float32)
    coords = {"x": torch.as_tensor(x_coords, dtype=torch.float32)}
    return DataTensor(data, coords, ("x",))


def test_concat_datatensor_existing_dimension():
    left = _datatensor([0.0, 1.0], x_coords=[0.0, 1.0])
    right = _datatensor([2.0, 3.0], x_coords=[2.0, 3.0])

    combined = concat([left, right], dim="x")

    assert isinstance(combined, DataTensor)
    torch.testing.assert_close(combined.data, torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(combined.coords["x"], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_concat_dataset_new_dimension_and_attrs():
    base_a = _datatensor([1.0, 2.0], x_coords=[0.0, 1.0])
    base_b = _datatensor([3.0, 4.0], x_coords=[0.0, 1.0])
    ds_a = Dataset({"foo": base_a}, attrs={"source": "sensor_A"})
    ds_b = Dataset({"foo": base_b}, attrs={"source": "sensor_B"})

    merged = concat([ds_a, ds_b], dim=("time", [0, 1]))

    assert isinstance(merged, Dataset)
    assert merged.attrs["source"] == "sensor_A"
    foo = merged["foo"]
    assert foo.dims == ("time", "x")
    expected = torch.stack([base_a.data, base_b.data], dim=0)
    torch.testing.assert_close(foo.data, expected)
    time_coord = merged.coords["time"]
    if isinstance(time_coord, torch.Tensor):
        expected_time = torch.tensor([0, 1], dtype=time_coord.dtype, device=time_coord.device)
        torch.testing.assert_close(time_coord, expected_time)
    else:
        assert list(time_coord) == [0, 1]


def test_concat_broadcasts_missing_dimensions():
    batch = torch.arange(2)
    spatial = torch.arange(3)
    time = torch.arange(2)
    variable = torch.tensor([0, 1])

    data_x = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    x = DataTensor(
        data_x,
        {"batch": batch, "spatial": spatial, "time": time, "variable": variable},
        ("batch", "spatial", "time", "variable"),
    )

    data_y = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)
    y = DataTensor(
        data_y,
        {"spatial": spatial, "variable": torch.tensor([2])},
        ("spatial", "variable"),
    )

    combined = concat([x, y], dim="variable")

    assert combined.dims == ("batch", "spatial", "time", "variable")
    broadcast_y = data_y.view(1, 3, 1, 1).expand(2, 3, 2, 1)
    expected = torch.cat([data_x, broadcast_y], dim=-1)
    torch.testing.assert_close(combined.data, expected)
    torch.testing.assert_close(combined.coords["variable"], torch.tensor([0, 1, 2]))
