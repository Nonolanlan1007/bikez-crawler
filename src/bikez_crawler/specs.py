"""Turns the raw scraped spec table into a stable, API-friendly shape.

``normalize_specs`` is pure and re-runnable against ``specs_raw``: fixing or extending the
mapping never requires re-crawling. Bump ``CURRENT_MAPPING_VERSION`` whenever the output
for existing input changes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

CURRENT_MAPPING_VERSION = 1

Parsed = dict[str, Any]
Parser = Callable[[str], Parsed | None]

_NUM = r"(-?\d[\d,]*(?:\.\d+)?)"
_RPM_TAIL = r"(?: @ (.+?) ?RPM)?"

_IGNORED_SECTIONS = {"further information", "compare with another motorcycle"}
_IGNORED_LABELS = {"rating", "updatespecs"}

_REPEATED_LABEL_SUFFIX = re.compile(r"\s*\(\d+\)$")


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _first_int(text: str) -> int | None:
    match = re.search(r"\d[\d,]*", text)
    return int(match[0].replace(",", "")) if match else None


def parse_number(value: str, unit: str) -> float | None:
    match = re.match(rf"^\s*{_NUM}\s*{unit}(?![A-Za-z])", value)
    return _to_float(match[1]) if match else None


def parse_int(value: str) -> int | None:
    match = re.match(r"^\s*(\d[\d,]*)\s*$", value)
    return int(match[1].replace(",", "")) if match else None


def parse_power(value: str) -> Parsed | None:
    match = re.match(rf"^\s*{_NUM} HP(?: \({_NUM} kW\)\)?)?{_RPM_TAIL}\s*$", value, re.IGNORECASE)
    if not match:
        return None
    parsed: Parsed = {"power_hp": _to_float(match[1])}
    if match[2]:
        parsed["power_kw"] = _to_float(match[2])
    if match[3] and (rpm := _first_int(match[3])) is not None:
        parsed["power_rpm"] = rpm
    return parsed


def parse_torque(value: str) -> Parsed | None:
    match = re.match(rf"^\s*{_NUM} Nm(?: \(.*?\))?{_RPM_TAIL}\s*$", value, re.IGNORECASE)
    if not match:
        return None
    parsed: Parsed = {"torque_nm": _to_float(match[1])}
    if match[2] and (rpm := _first_int(match[2])) is not None:
        parsed["torque_rpm"] = rpm
    return parsed


def parse_bore_stroke(value: str) -> Parsed | None:
    match = re.match(rf"^\s*{_NUM} x {_NUM} mm", value)
    if not match:
        return None
    return {"bore_mm": _to_float(match[1]), "stroke_mm": _to_float(match[2])}


def parse_ratio(value: str) -> float | None:
    match = re.match(rf"^\s*{_NUM}:1\s*$", value)
    return _to_float(match[1]) if match else None


def parse_tire_pressure_kpa(value: str) -> float | None:
    match = re.match(rf"^\s*{_NUM} PSI \({_NUM} Bar or {_NUM} kPa\)", value)
    return _to_float(match[3]) if match else None


def parse_sprockets(value: str) -> Parsed | None:
    match = re.match(r"^\s*(\d+)/(\d+)", value)
    if not match:
        return None
    return {"sprocket_front_teeth": int(match[1]), "sprocket_rear_teeth": int(match[2])}


_CURRENCIES = {"US$": "USD", "Euro": "EUR"}


def parse_price(value: str) -> Parsed | None:
    match = re.match(rf"^\s*(US\$|Euro)\s*{_NUM}", value)
    if not match:
        return None
    return {"price_as_new": {"amount": _to_float(match[2]), "currency": _CURRENCIES[match[1]]}}


def parse_gearbox(value: str) -> Parsed:
    match = re.match(r"^\s*(\d+)-speed", value, re.IGNORECASE)
    return {"gearbox": value, "gears": int(match[1]) if match else None}


def _text(key: str) -> Parser:
    return lambda value: {key: value}


def _number(key: str, unit: str) -> Parser:
    def parse(value: str) -> Parsed | None:
        number = parse_number(value, unit)
        return None if number is None else {key: number}

    return parse


def _int(key: str) -> Parser:
    def parse(value: str) -> Parsed | None:
        number = parse_int(value)
        return None if number is None else {key: number}

    return parse


def parse_plain_number(value: str) -> float | None:
    return _to_float(value) if re.fullmatch(r"\s*\d+(\.\d+)?\s*", value) else None


def _single(key: str, parse_value: Callable[[str], float | int | None]) -> Parser:
    def parse(value: str) -> Parsed | None:
        number = parse_value(value)
        return None if number is None else {key: number}

    return parse


def _label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _REPEATED_LABEL_SUFFIX.sub("", label).lower())


_FIELDS: dict[str, tuple[str, Parser]] = {}


def _add(labels: list[str], group: str, parser: Parser) -> None:
    for label in labels:
        _FIELDS[_label_key(label)] = (group, parser)


_add(["Model name", "Motorcycle name", "Model"], "general", _text("name"))
_add(["Year", "Year model", "Model year", "Year of manufacture"], "general", _int("year"))
_add(["Category", "Type"], "general", _text("category"))
_add(["Price as new"], "general", parse_price)

_add(["Displacement", "Engine size"], "engine", _number("displacement_cc", "ccm"))
_add(["Engine type", "Type of engine"], "engine", _text("type"))
_add(["Engine details"], "engine", _text("details"))
_add(["Power", "Output", "Power output", "Effect"], "engine", parse_power)
_add(["Torque"], "engine", parse_torque)
_add(["Bore x stroke"], "engine", parse_bore_stroke)
_add(["Compression"], "engine", _single("compression_ratio", parse_ratio))
_add(["Valves per cylinder"], "engine", _int("valves_per_cylinder"))
_add(["Fuel system"], "engine", _text("fuel_system"))
_add(["Fuel control"], "engine", _text("fuel_control"))
_add(["Ignition"], "engine", _text("ignition"))
_add(["Lubrication system"], "engine", _text("lubrication"))
_add(["Cooling system"], "engine", _text("cooling"))
_add(["Exhaust system"], "engine", _text("exhaust"))
_add(["Top speed"], "engine", _number("top_speed_kmh", "km/h"))
_add(["0-100 km/h (0-62 mph)"], "engine", _number("acceleration_0_100_s", "seconds?"))
_add(["MaxRPM", "MaximumRPM"], "engine", _int("max_rpm"))
_add(["Fuel consumption"], "engine", _number("fuel_consumption_l_per_100km", r"litres/100 km"))
_add(["Greenhouse gases"], "engine", _number("co2_g_per_km", r"CO2 g/km"))
_add(["Emission details"], "engine", _text("emission_standard"))

_add(["Gearbox"], "transmission", parse_gearbox)
_add(["Transmission type"], "transmission", _text("type"))
_add(["Clutch"], "transmission", _text("clutch"))
_add(["Driveline"], "transmission", _text("driveline"))

_add(["Frame type"], "chassis", _text("frame_type"))
_add(["Rake (fork angle)"], "chassis", _number("rake_deg", "°"))
_add(["Trail"], "chassis", _number("trail_mm", "mm"))
_add(["Front suspension"], "chassis", _text("front_suspension"))
_add(["Front wheel travel"], "chassis", _number("front_wheel_travel_mm", "mm"))
_add(["Rear suspension"], "chassis", _text("rear_suspension"))
_add(["Rear wheel travel"], "chassis", _number("rear_wheel_travel_mm", "mm"))
_add(["Front tire"], "chassis", _text("front_tire"))
_add(["Rear tire"], "chassis", _text("rear_tire"))
_add(["Front brakes"], "chassis", _text("front_brakes"))
_add(["Rear brakes"], "chassis", _text("rear_brakes"))
_add(["Wheels"], "chassis", _text("wheels"))
_add(["Seat"], "chassis", _text("seat"))

_add(["Dry weight"], "dimensions", _number("dry_weight_kg", "kg"))
_add(["Weight incl. oil, gas, etc"], "dimensions", _number("wet_weight_kg", "kg"))
_add(["Seat height"], "dimensions", _number("seat_height_mm", "mm"))
_add(["Alternate seat height"], "dimensions", _number("alternate_seat_height_mm", "mm"))
_add(["Overall length"], "dimensions", _number("overall_length_mm", "mm"))
_add(["Overall width"], "dimensions", _number("overall_width_mm", "mm"))
_add(["Overall height"], "dimensions", _number("overall_height_mm", "mm"))
_add(["Wheelbase"], "dimensions", _number("wheelbase_mm", "mm"))
_add(["Ground clearance"], "dimensions", _number("ground_clearance_mm", "mm"))
_add(["Fuel capacity"], "dimensions", _number("fuel_capacity_l", "litres?"))
_add(["Reserve fuel capacity"], "dimensions", _number("reserve_fuel_capacity_l", "litres?"))
_add(["Oil capacity"], "dimensions", _number("oil_capacity_l", "litres?"))
_add(["Power/weight ratio"], "dimensions", _number("power_weight_ratio_hp_per_kg", "HP/kg"))
_add(["Front percentage of weight"], "dimensions", _single("front_weight_percentage", parse_plain_number))

_add(["Engine oil"], "maintenance", _text("engine_oil"))
_add(["Spark plugs"], "maintenance", _text("spark_plugs"))
_add(["Service interval"], "maintenance", _text("service_interval"))
_add(["Coolant"], "maintenance", _number("coolant_l", "litres?"))
_add(["Brake fluid"], "maintenance", _text("brake_fluid"))
_add(["Fork tube size"], "maintenance", _number("fork_tube_size_mm", "mm"))
_add(["Idle speed"], "maintenance", _single("idle_rpm", _first_int))
_add(["Tire pressure front"], "maintenance", _single("tire_pressure_front_kpa", parse_tire_pressure_kpa))
_add(["Tire pressure rear"], "maintenance", _single("tire_pressure_rear_kpa", parse_tire_pressure_kpa))
_add(["Sprockets"], "maintenance", parse_sprockets)
_add(["Chain size"], "maintenance", _text("chain_size"))
_add(["Chain links"], "maintenance", _int("chain_links"))

_add(["Color options"], "other", _text("color_options"))
_add(["Starter"], "other", _text("starter"))
_add(["Comments"], "other", _text("comments"))
_add(["Instruments"], "other", _text("instruments"))
_add(["Light"], "other", _text("light"))
_add(["Factory warranty"], "other", _text("factory_warranty"))
_add(["Electrical"], "other", _text("electrical"))
_add(["Carrying capacity"], "other", _text("carrying_capacity"))
_add(["Modifications compared to previous model"], "other", _text("modifications"))

_BRAKE_DIAMETER_AFTER = {
    _label_key("Front brakes"): "front_brake_diameter_mm",
    _label_key("Rear brakes"): "rear_brake_diameter_mm",
}
_DIAMETER = _label_key("Diameter")


def _resolve(label: str, previous_key: str | None) -> tuple[str, Parser] | None:
    key = _label_key(label)
    if key == _DIAMETER:
        target = _BRAKE_DIAMETER_AFTER.get(previous_key or "")
        return ("chassis", _number(target, "mm")) if target else None
    return _FIELDS.get(key)


def normalize_specs(raw: dict[str, dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    unmapped: dict[str, dict[str, str]] = {}

    for section, labels in raw.items():
        if section.strip().lower() in _IGNORED_SECTIONS:
            continue
        previous_key: str | None = None
        for label, value in labels.items():
            key = _label_key(label)
            current_previous, previous_key = previous_key, key
            if key in _IGNORED_LABELS:
                continue
            entry = _resolve(label, current_previous)
            parsed = entry[1](value) if entry else None
            if entry is None or parsed is None:
                unmapped.setdefault(section, {})[label] = value
                continue
            result.setdefault(entry[0], {}).update(parsed)

    if unmapped:
        result["unmapped"] = unmapped
    return result
