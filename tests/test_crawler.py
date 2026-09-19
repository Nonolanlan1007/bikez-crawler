from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from bikez_crawler.crawler import table_to_object


def _table(rows: str) -> Tag:
    table = BeautifulSoup(f"<table>{rows}</table>", "lxml").table
    assert table is not None
    return table


def test_extracts_sections_labels_and_values() -> None:
    table = _table(
        "<tr><th>General information</th></tr>"
        "<tr><td><b>Year</b></td><td>2024</td></tr>"
        "<tr><th>Engine and transmission</th></tr>"
        "<tr><td><b>Displacement</b></td><td>659.0 ccm (40.21 cubic inches)</td></tr>"
    )

    assert table_to_object(table) == {
        "General information": {"Year": "2024"},
        "Engine and transmission": {"Displacement": "659.0 ccm (40.21 cubic inches)"},
    }


def test_rows_before_any_header_go_to_general() -> None:
    assert table_to_object(_table("<tr><td><b>Year</b></td><td>2024</td></tr>")) == {
        "General": {"Year": "2024"}
    }


def test_compare_widget_rows_are_skipped() -> None:
    table = _table(
        "<tr><th>Compare with another motorcycle</th></tr>"
        "<tr><td><b>Compare</b></td><td><form><select><option>x</option></select></form></td></tr>"
    )

    assert table_to_object(table) == {"Compare with another motorcycle": {}}


def test_hidden_anti_scraping_noise_is_stripped() -> None:
    table = _table(
        "<tr><th>Engine</th></tr>"
        '<tr><td><b>Power</b></td><td>95.0 HP<span style="display:none">junk</span> (69.3 kW)</td></tr>'
    )

    assert table_to_object(table) == {"Engine": {"Power": "95.0 HP (69.3 kW)"}}


def test_whitespace_in_values_is_collapsed() -> None:
    table = _table(
        "<tr><th>Other</th></tr><tr><td><b>Starter</b></td><td>  Electric \n  &amp;   kick </td></tr>"
    )

    assert table_to_object(table) == {"Other": {"Starter": "Electric & kick"}}


def test_repeated_label_keeps_every_occurrence() -> None:
    table = _table(
        "<tr><th>Chassis</th></tr>"
        "<tr><td><b>Front brakes</b></td><td>Double disc</td></tr>"
        "<tr><td><b>Diameter</b></td><td>320 mm</td></tr>"
        "<tr><td><b>Rear brakes</b></td><td>Single disc</td></tr>"
        "<tr><td><b>Diameter</b></td><td>220 mm</td></tr>"
        "<tr><td><b>Diameter</b></td><td>200 mm</td></tr>"
    )

    assert table_to_object(table) == {
        "Chassis": {
            "Front brakes": "Double disc",
            "Diameter": "320 mm",
            "Rear brakes": "Single disc",
            "Diameter (2)": "220 mm",
            "Diameter (3)": "200 mm",
        }
    }
