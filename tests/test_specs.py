from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bikez_crawler.specs import (
    normalize_specs,
    parse_bore_stroke,
    parse_number,
    parse_power,
    parse_price,
    parse_ratio,
    parse_sprockets,
    parse_tire_pressure_kpa,
    parse_torque,
)

SAMPLES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "specs_samples.json").read_text()
)
BY_TAG = {doc["tag"]: doc["specs"] for doc in SAMPLES}


@pytest.mark.parametrize("doc", SAMPLES, ids=[doc["tag"] for doc in SAMPLES])
def test_every_sample_label_is_mapped_and_parsed(doc: dict[str, Any]) -> None:
    assert "unmapped" not in normalize_specs(doc["specs"])


def test_sport_bike_is_fully_normalized() -> None:
    out = normalize_specs(BY_TAG["aprilia_tuono_660_2024"])

    assert out["general"] == {"name": "Aprilia Tuono 660", "year": 2024, "category": "Sport"}
    engine = out["engine"]
    assert engine["displacement_cc"] == 659.0
    assert (engine["power_hp"], engine["power_kw"], engine["power_rpm"]) == (95.0, 69.3, 10500)
    assert (engine["torque_nm"], engine["torque_rpm"]) == (67.0, 8500)
    assert (engine["bore_mm"], engine["stroke_mm"]) == (81.0, 63.9)
    assert engine["compression_ratio"] == 13.5
    assert engine["valves_per_cylinder"] == 4
    assert engine["fuel_consumption_l_per_100km"] == 4.9
    assert engine["co2_g_per_km"] == 113.7
    assert engine["emission_standard"] == "Euro 5"
    assert out["transmission"]["gears"] == 6
    assert out["chassis"]["rake_deg"] == 24.1
    assert out["chassis"]["front_tire"] == "120/70-ZR17"
    assert out["dimensions"]["wet_weight_kg"] == 183.0
    assert out["dimensions"]["seat_height_mm"] == 820.0
    assert out["dimensions"]["fuel_capacity_l"] == 15.0


def test_optional_fields_are_simply_absent() -> None:
    out = normalize_specs(BY_TAG["access_xtreme_enduro_sp300s_2024"])

    assert out["engine"]["power_hp"] == 18.8
    assert "power_rpm" not in out["engine"]
    assert out["transmission"] == {
        "gearbox": "Automatic",
        "gears": None,
        "type": "Chain (final drive)",
        "driveline": "CVT with reverse, H/L gear.",
    }
    assert "unmapped" not in out


def test_site_chrome_never_reaches_the_output() -> None:
    dumped = json.dumps(normalize_specs(BY_TAG["access_xtreme_enduro_sp300s_2024"]))

    for chrome in ("Insurance", "Amazon", "discussion group", "Report missing specs", "rate it"):
        assert chrome not in dumped


@pytest.mark.parametrize(
    ("labels", "group", "key", "value", "expected"),
    [
        (["Model name", "Motorcycle name", "Model"], "general", "name", "Foo 125", "Foo 125"),
        (["Year", "Year model", "Model year", "Year of manufacture"], "general", "year", "2024", 2024),
        (["Category", "Type"], "general", "category", "Sport", "Sport"),
        (["Power", "Output", "Power output", "Effect"], "engine", "power_hp", "10.0 HP (7.4 kW))", 10.0),
        (["Engine type", "Type of engine"], "engine", "type", "Single cylinder", "Single cylinder"),
        (["Displacement", "Engine size"], "engine", "displacement_cc", "125.0 ccm (7.6 cubic inches)", 125.0),
        (["MaxRPM", "MaximumRPM"], "engine", "max_rpm", "12300", 12300),
        (["Bore x stroke", "Borexstroke"], "engine", "bore_mm", "54.0 x 54.5 mm (2.1 x 2.1 inches)", 54.0),
        (["Valves per cylinder", "Valvesper cylinder"], "engine", "valves_per_cylinder", "4", 4),
        (["Front suspension", "Frontsuspension"], "chassis", "front_suspension", "USD fork", "USD fork"),
        (["Rear suspension", "Rearsuspension"], "chassis", "rear_suspension", "Mono shock", "Mono shock"),
    ],
)
def test_label_aliases_map_to_one_field(
    labels: list[str], group: str, key: str, value: str, expected: Any
) -> None:
    for label in labels:
        assert normalize_specs({"Any section": {label: value}})[group][key] == expected


def test_general_and_general_moped_sections_are_equivalent() -> None:
    fields = {"Model": "Foo 50", "Year": "2024"}

    assert normalize_specs({"General information": fields}) == normalize_specs(
        {"General moped information": fields}
    )


def test_dry_and_wet_weight_are_distinct_fields() -> None:
    out = normalize_specs(
        {
            "Physical": {
                "Dry weight": "100.0 kg (220.5 pounds)",
                "Weight incl. oil, gas, etc": "110.0 kg (242.5 pounds)",
            }
        }
    )

    assert out["dimensions"] == {"dry_weight_kg": 100.0, "wet_weight_kg": 110.0}


def test_diameter_is_resolved_from_the_preceding_brake_label() -> None:
    both = {
        "Chassis": {
            "Front brakes": "Double disc",
            "Diameter": "320 mm (12.6 inches)",
            "Rear brakes": "Single disc",
            "Diameter (2)": "220 mm (8.7 inches)",
        }
    }
    rear_only = {"Chassis": {"Rear brakes": "Single disc", "Diameter": "220 mm (8.7 inches)"}}

    assert normalize_specs(both)["chassis"]["front_brake_diameter_mm"] == 320.0
    assert normalize_specs(both)["chassis"]["rear_brake_diameter_mm"] == 220.0
    assert normalize_specs(rear_only)["chassis"] == {
        "rear_brakes": "Single disc",
        "rear_brake_diameter_mm": 220.0,
    }


def test_orphaned_diameter_is_kept_as_unmapped() -> None:
    out = normalize_specs({"Chassis": {"Wheels": "Wire spoked", "Diameter": "220 mm (8.7 inches)"}})

    assert out["unmapped"] == {"Chassis": {"Diameter": "220 mm (8.7 inches)"}}


def test_ignored_sections_and_labels_are_dropped() -> None:
    raw = {
        "General information": {"Rating": "Do you know this bike?", "Year": "2024"},
        "Other specifications": {"Update specs": "Report missing specs", "Starter": "Electric"},
        "Further information": {"Maintenance": "Find parts"},
        "Compare with another motorcycle": {},
    }

    assert normalize_specs(raw) == {"general": {"year": 2024}, "other": {"starter": "Electric"}}


def test_unknown_label_is_kept_under_unmapped() -> None:
    out = normalize_specs({"Engine and transmission": {"Flux capacitor": "1.21 GW"}})

    assert out == {"unmapped": {"Engine and transmission": {"Flux capacitor": "1.21 GW"}}}


def test_unparseable_value_is_kept_under_unmapped() -> None:
    out = normalize_specs({"Physical measures and capacities": {"Dry weight": "heavy"}})

    assert out == {"unmapped": {"Physical measures and capacities": {"Dry weight": "heavy"}}}


def test_empty_input() -> None:
    assert normalize_specs({}) == {}
    assert normalize_specs({"Compare with another motorcycle": {}}) == {}


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("870 mm (34.3 inches) If adjustable, lowest setting.", "mm", 870.0),
        ("14.80 litres (3.91 US gallons)", "litres?", 14.8),
        ("24.1°", "°", 24.1),
        ("0.0834 HP/kg", "HP/kg", 0.0834),
        ("1,250 mm", "mm", 1250.0),
        ("heavy", "kg", None),
        ("870 mmx", "mm", None),
    ],
)
def test_parse_number(value: str, unit: str, expected: float | None) -> None:
    assert parse_number(value, unit) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("18.8 HP (13.7 kW))", {"power_hp": 18.8, "power_kw": 13.7}),
        ("95.0 HP (69.3 kW)) @ 10500 RPM", {"power_hp": 95.0, "power_kw": 69.3, "power_rpm": 10500}),
        ("50.0 HP @ 6000 rpm", {"power_hp": 50.0, "power_rpm": 6000}),
        ("lots", None),
    ],
)
def test_parse_power(value: str, expected: dict[str, Any] | None) -> None:
    assert parse_power(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("58.0 Nm (5.9 kgf-m or 42.8 ft.lbs)", {"torque_nm": 58.0}),
        ("67.0 Nm (6.8 kgf-m or 49.4 ft.lbs) @ 8500 RPM", {"torque_nm": 67.0, "torque_rpm": 8500}),
        ("strong", None),
    ],
)
def test_parse_torque(value: str, expected: dict[str, Any] | None) -> None:
    assert parse_torque(value) == expected


def test_parse_bore_stroke() -> None:
    assert parse_bore_stroke("81.0 x 63.9 mm (3.2 x 2.5 inches)") == {"bore_mm": 81.0, "stroke_mm": 63.9}
    assert parse_bore_stroke("81.0") is None


def test_parse_ratio() -> None:
    assert parse_ratio("13.5:1") == 13.5
    assert parse_ratio("13.5") is None


def test_parse_tire_pressure_kpa() -> None:
    assert parse_tire_pressure_kpa("32 PSI (2.2 Bar or 220 kPa)") == 220.0
    assert parse_tire_pressure_kpa("32 PSI") is None


def test_parse_sprockets() -> None:
    assert parse_sprockets("17/44 (front/rear)") == {"sprocket_front_teeth": 17, "sprocket_rear_teeth": 44}
    assert parse_sprockets("n/a") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "US$ 26945. MSRP depends on country, taxes, accessories, etc.",
            {"amount": 26945.0, "currency": "USD"},
        ),
        (
            "Euro 2999. MSRP depends on country, taxes, accessories, etc.",
            {"amount": 2999.0, "currency": "EUR"},
        ),
    ],
)
def test_parse_price(value: str, expected: dict[str, Any]) -> None:
    assert parse_price(value) == {"price_as_new": expected}


def test_parse_price_rejects_unknown_currency() -> None:
    assert parse_price("Yen 500000.") is None
