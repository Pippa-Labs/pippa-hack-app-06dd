"""Tests for Dry Spell, through LocalRunner so the manifest is enforced.

No test touches the network (conftest.py refuses it). Every request goes
through `App._fetch`, which `Stub` replaces with canned Open-Meteo answers
shaped like the real ones, captured by hand in November 2026.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest
from pippa_agentapp_sdk import (
    InputShapeViolation,
    LocalRunner,
    OutputShapeViolation,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
)

import app as app_module
from app import App, LengthOutOfRange, NoPlaceGiven, PlaceNotFound

# Thursday 19 November 2026, 14:20 in London (GMT, so UTC too). Not on the
# hour on purpose: the hour we are in only counts from now.
NOW = datetime(2026, 11, 19, 14, 20, tzinfo=timezone.utc)
MIDNIGHT = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

BRISTOL_UK = {
    "name": "Bristol", "latitude": 51.45523, "longitude": -2.59665,
    "country_code": "GB", "country": "United Kingdom", "admin1": "England",
    "admin2": "Bristol", "timezone": "Europe/London", "population": 479024,
}
BRISTOL_VA = {
    "name": "Bristol", "latitude": 36.59649, "longitude": -82.18847,
    "country_code": "US", "country": "United States", "admin1": "Virginia",
    "timezone": "America/New_York", "population": 17141,
}
CAMDEN = {
    "name": "Camden Town", "latitude": 51.54, "longitude": -0.14,
    "country_code": "GB", "country": "United Kingdom", "admin1": "England",
    "admin2": "Greater London",
}


def geocode(*results) -> bytes:
    # A name the geocoder does not know is a 200 with no `results` key at all.
    body = {"generationtime_ms": 0.1}
    if results:
        body["results"] = list(results)
    return json.dumps(body).encode()


def forecast(chances, *, mm=None, temps=None, winds=None,
             start=MIDNIGHT, zone="Europe/London", offset=0) -> bytes:
    """An Open-Meteo hourly forecast, one entry per hour from `start`."""
    n = len(chances)
    return json.dumps({
        "latitude": 51.46, "longitude": -2.58,
        "utc_offset_seconds": offset, "timezone": zone,
        "hourly": {
            "time": [int((start + timedelta(hours=i)).timestamp()) for i in range(n)],
            "precipitation_probability": list(chances),
            "precipitation": list(mm) if mm is not None else [0.0] * n,
            "temperature_2m": list(temps) if temps is not None else [8.0] * n,
            "wind_speed_10m": list(winds) if winds is not None else [12.0] * n,
        },
    }).encode()


def hours(*parts) -> list:
    """hours((14, 90), (4, 10), ...) -> 14 hours at 90%, then 4 at 10%, ..."""
    out = []
    for count, value in parts:
        out += [value] * count
    return out


class Stub:
    """Answers each Open-Meteo URL with a canned body, or raises."""

    def __init__(self, geo=None, fc=None):
        self.geo = geo if geo is not None else geocode(BRISTOL_UK)
        self.fc = fc if fc is not None else forecast(hours((72, 90)))
        self.urls: list[str] = []
        self.timeouts: list[float] = []

    def __call__(self, url, timeout=6.0):
        self.urls.append(url)
        self.timeouts.append(timeout)
        answer = self.geo if "geocoding-api" in url else self.fc
        if isinstance(answer, BaseException):
            raise answer
        return answer


def run(inputs=None, stub=None, *, explicit=False, now=NOW) -> Result:
    app = App()
    app._fetch = stub or Stub()
    return LocalRunner(app).call(
        "dry.window", {"place": "Bristol", **(inputs or {})}, granted_secrets={},
        now=now, request_id="t1", explicit=explicit,
    ).result


def http_error(code, headers=None):
    return urllib.error.HTTPError("https://example.com", code, "error", headers or {}, io.BytesIO(b""))


# ── The manifest ─────────────────────────────────────────────────────────────


def test_the_manifest_is_valid():
    manifest = App().manifest
    assert manifest.app_id == "hack.app_06dd"
    assert manifest.name == "Dry Spell"
    assert manifest.version == "0.1.0"
    assert [o.id for o in manifest.outcomes] == ["dry.window"]
    assert " by " not in f" {manifest.summary.lower()} "


# ── Finding the spell ────────────────────────────────────────────────────────


def test_dry_now_counts_from_now_and_runs_until_the_rain():
    fc = forecast(hours((14, 90), (4, 10), (1, 60), (53, 90)))
    data = run(stub=Stub(fc=fc)).data
    assert data["found"] is True
    assert data["dry_now"] is True
    assert data["start"] == "Thursday 14:20"
    assert data["end"] == "Thursday 18:00"
    assert data["hours_found"] == 3.7
    assert data["max_rain_chance_percent"] == 10
    assert data["dry_beyond_forecast"] is False
    assert data["place_used"] == "Bristol, England, United Kingdom"
    assert data["timezone_used"] == "Europe/London"
    assert data["hours_needed"] == 1.0
    assert data["threshold_percent"] == 20
    assert "least_wet_start" not in data


def test_raining_now_finds_when_it_stops():
    fc = forecast(hours((16, 80), (5, 15), (51, 70)))
    data = run(stub=Stub(fc=fc)).data
    assert data["dry_now"] is False
    assert data["start"] == "Thursday 16:00"
    assert data["end"] == "Thursday 21:00"
    assert data["hours_found"] == 5.0


def test_a_spell_too_short_is_skipped():
    fc = forecast(hours((15, 80), (2, 10), (1, 80), (5, 5), (49, 80)))
    data = run({"hours_needed": 3}, Stub(fc=fc)).data
    assert data["start"] == "Thursday 18:00"
    assert data["end"] == "Thursday 23:00"
    assert data["hours_needed"] == 3


def test_the_part_of_this_hour_already_gone_does_not_count():
    # 14:00-16:00 is dry, but from 14:20 that is only 1.7 hours.
    fc = forecast(hours((14, 80), (2, 10), (2, 80), (3, 10), (51, 80)))
    data = run({"hours_needed": 2}, Stub(fc=fc)).data
    assert data["start"] == "Thursday 18:00"
    assert data["hours_found"] == 3.0


def test_hours_already_gone_are_ignored():
    # Dry all morning, raining from 14:00: the morning is no help now.
    fc = forecast(hours((14, 0), (2, 90), (56, 10)))
    data = run(stub=Stub(fc=fc)).data
    assert data["start"] == "Thursday 16:00"
    assert data["hours_searched"] == 58


def test_forecast_rain_breaks_a_spell_whatever_the_chance():
    mm = [0.0] * 72
    mm[16] = 0.4
    fc = forecast(hours((72, 10)), mm=mm)
    data = run(stub=Stub(fc=fc)).data
    assert data["start"] == "Thursday 14:20"
    assert data["end"] == "Thursday 16:00"


def test_twenty_percent_is_dry_and_twenty_one_is_not():
    fc = forecast(hours((14, 90), (2, 20), (1, 21), (55, 90)))
    data = run(stub=Stub(fc=fc)).data
    assert data["end"] == "Thursday 16:00"
    assert data["max_rain_chance_percent"] == 20


def test_a_spell_running_off_the_end_of_the_forecast_says_so():
    fc = forecast(hours((16, 90), (56, 0)))
    result = run(stub=Stub(fc=fc))
    assert result.data["dry_beyond_forecast"] is True
    assert result.data["end"] == "Sunday 00:00"
    assert result.data["hours_found"] == 56.0
    assert "at least until Sunday 00:00" in result.narrative


def test_weather_during_the_spell_is_summarised():
    temps = [5.0] * 72
    winds = [10.0] * 72
    temps[16], temps[17] = 9.46, 3.0
    winds[17] = 31.25
    winds[19] = 99.0           # after the spell: not counted
    fc = forecast(hours((16, 80), (3, 10), (53, 80)), temps=temps, winds=winds)
    data = run(stub=Stub(fc=fc)).data
    assert data["max_temp_c"] == 9.5
    assert data["min_temp_c"] == 3.0
    assert data["max_wind_kph"] == 31.2


def test_an_hour_with_no_published_chance_is_not_called_dry():
    chances = hours((16, 90), (2, 10), (1, 10), (53, 90))
    chances[17] = None
    fc = forecast(chances)
    data = run(stub=Stub(fc=fc)).data
    assert data["start"] == "Thursday 16:00"
    assert data["end"] == "Thursday 17:00"


def test_no_spell_found_points_at_the_least_wet_stretch():
    chances = hours((72, 70))
    chances[30], chances[31] = 35, 40
    chances[50], chances[51] = 35, 40      # the same: the earlier one wins
    result = run({"hours_needed": 2}, Stub(fc=forecast(chances)))
    data = result.data
    assert data["found"] is False
    assert data["dry_now"] is False
    assert data["hours_found"] == 0.0
    assert "start" not in data and "end" not in data
    assert "max_rain_chance_percent" not in data
    assert data["least_wet_start"] == "Friday 06:00"
    assert data["least_wet_rain_chance_percent"] == 40
    assert result.narrative.startswith("No dry spell of 2 hours in Bristol")
    assert "Friday 06:00" in result.narrative


def test_times_are_in_the_places_own_zone():
    # New York is UTC-5 in November: 14:20 UTC is 09:20 there.
    start = datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc)   # local midnight
    fc = forecast(hours((11, 80), (3, 10), (58, 80)), start=start,
                  zone="America/New_York", offset=-18000)
    data = run({"region": "Virginia"}, Stub(geo=geocode(BRISTOL_UK, BRISTOL_VA), fc=fc)).data
    assert data["place_used"] == "Bristol, Virginia, United States"
    assert data["timezone_used"] == "America/New_York"
    assert data["start"] == "Thursday 11:00"
    assert data["end"] == "Thursday 14:00"


def test_an_unknown_zone_name_falls_back_to_the_offset():
    fc = forecast(hours((72, 10)), zone="Mars/Olympus", offset=3600)
    data = run(stub=Stub(fc=fc)).data
    assert data["start"] == "Thursday 15:20"
    assert data["timezone_used"] == "UTC+01:00"


def test_the_narrative_says_dry_now_and_the_rule():
    fc = forecast(hours((14, 90), (4, 10), (54, 90)))
    sentence = run(stub=Stub(fc=fc)).narrative
    assert sentence == (
        "It is dry in Bristol, England, United Kingdom now and should stay dry "
        "until Thursday 18:00 (3.7 hours) (dry here means a 20% chance of rain "
        "or less, and none forecast)."
    )


def test_the_narrative_for_a_later_spell():
    fc = forecast(hours((16, 80), (1, 15), (55, 70)))
    sentence = run(stub=Stub(fc=fc)).narrative
    assert sentence.startswith(
        "The next dry spell in Bristol, England, United Kingdom starts "
        "Thursday 16:00 and lasts until Thursday 17:00 (1 hour)"
    )


# ── Where ────────────────────────────────────────────────────────────────────


def test_the_largest_place_of_that_name_wins_without_a_region():
    data = run(stub=Stub(geo=geocode(BRISTOL_UK, BRISTOL_VA))).data
    assert data["place_used"] == "Bristol, England, United Kingdom"


@pytest.mark.parametrize("region", ["UK", "united kingdom", "England", "GB", "Britain"])
def test_a_country_or_region_picks_the_match(region):
    stub = Stub(geo=geocode(BRISTOL_VA, BRISTOL_UK))
    assert run({"region": region}, stub).data["place_used"].endswith("United Kingdom")


def test_a_county_name_matches_inside_a_longer_one():
    stub = Stub(geo=geocode(CAMDEN))
    data = run({"place": "Camden Town", "region": "London"}, stub).data
    assert data["place_used"] == "Camden Town, England, United Kingdom"


def test_a_region_after_a_comma_is_split_off():
    # The geocoder answers nothing for "Bristol, Virginia" as one name.
    stub = Stub(geo=geocode(BRISTOL_UK, BRISTOL_VA))
    data = run({"place": "Bristol, Virginia"}, stub).data
    assert data["place_used"] == "Bristol, Virginia, United States"
    assert "name=Bristol&" in stub.urls[0]


def test_the_place_name_is_url_encoded():
    stub = Stub()
    run({"place": "Milton  Keynes"}, stub)
    assert "name=Milton+Keynes&" in stub.urls[0]


def test_the_forecast_is_asked_for_at_the_places_coordinates():
    stub = Stub()
    run(stub=stub)
    url = stub.urls[1]
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    for part in ("latitude=51.455", "longitude=-2.59", "timezone=auto",
                 "timeformat=unixtime", "forecast_days=3"):
        assert part in url
    assert stub.timeouts == [5.0, 6.0]


def test_an_unknown_place_is_named_back():
    with pytest.raises(PlaceNotFound) as failure:
        run({"place": "Xyzzyville"}, Stub(geo=geocode()))
    assert failure.value.user_message == "I do not know where Xyzzyville is."


def test_a_place_not_in_the_region_says_so():
    with pytest.raises(PlaceNotFound) as failure:
        run({"region": "Narnia"}, Stub(geo=geocode(BRISTOL_UK, BRISTOL_VA)))
    assert failure.value.user_message == "I could not find Bristol in Narnia."


def test_a_result_without_coordinates_is_skipped():
    broken = {"name": "Bristol", "country": "Nowhere"}
    data = run(stub=Stub(geo=geocode(broken, BRISTOL_UK))).data
    assert data["place_used"] == "Bristol, England, United Kingdom"


@pytest.mark.parametrize("place", ["", "   ", None])
def test_no_place_asks_for_one(place):
    inputs = {} if place is None else {"place": place}
    app = App()
    app._fetch = Stub()
    with pytest.raises(NoPlaceGiven) as failure:
        LocalRunner(app).call("dry.window", inputs, granted_secrets={}, now=NOW,
                              request_id="t1")
    assert "which town or city" in failure.value.user_message


# ── How long ─────────────────────────────────────────────────────────────────


def test_half_hours_are_kept():
    assert run({"hours_needed": 1.5}).data["hours_needed"] == 1.5


@pytest.mark.parametrize("value", [0, -2])
def test_no_length_is_refused(value):
    with pytest.raises(LengthOutOfRange) as failure:
        run({"hours_needed": value})
    assert failure.value.user_message == "Tell me how many hours of dry weather you need."


def test_too_long_is_refused_with_the_limit():
    with pytest.raises(LengthOutOfRange) as failure:
        run({"hours_needed": 30})
    assert failure.value.user_message == (
        "I can look for a dry spell of up to 24 hours, so 30 hours is too long.")


def test_a_length_that_is_not_a_number_never_reaches_the_app():
    with pytest.raises(InputShapeViolation):
        run({"hours_needed": "a while"})


# ── When Open-Meteo misbehaves ───────────────────────────────────────────────


def test_rate_limited_is_retryable_with_the_wait():
    stub = Stub(geo=http_error(429, {"Retry-After": "7"}))
    with pytest.raises(RateLimitedError) as failure:
        run(stub=stub)
    assert failure.value.retry_after_seconds == 7
    assert failure.value.retryable is True


@pytest.mark.parametrize("error", [
    http_error(500),
    http_error(503),
    TimeoutError("timed out"),
    urllib.error.URLError("no route"),
    ConnectionResetError("reset"),
])
def test_an_unreachable_service_is_retryable(error):
    with pytest.raises(ServiceUnavailableError) as failure:
        run(stub=Stub(fc=error))
    assert failure.value.retryable is True


def test_a_rejected_place_name_is_not_found():
    with pytest.raises(PlaceNotFound):
        run(stub=Stub(geo=http_error(400)))


def test_a_rejected_forecast_is_our_problem_not_the_users():
    with pytest.raises(ServiceUnavailableError):
        run(stub=Stub(fc=http_error(400)))


@pytest.mark.parametrize("body", [
    b"<html>maintenance</html>",
    b'{"error": true}',
    b'{"hourly": {"time": ["not a time"]}}',
])
def test_a_forecast_in_the_wrong_shape_is_unavailable(body):
    with pytest.raises(ServiceUnavailableError):
        run(stub=Stub(fc=body))


def test_a_forecast_entirely_in_the_past_is_unavailable():
    with pytest.raises(ServiceUnavailableError):
        run(stub=Stub(fc=forecast(hours((10, 10)))))


def test_the_real_fetch_sends_a_user_agent_and_a_timeout(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen["agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return Response(b"{}")

    monkeypatch.setattr(app_module.urllib.request, "urlopen", fake_urlopen)
    assert App()._fetch("https://api.open-meteo.com/x", timeout=4.0) == b"{}"
    assert seen == {"agent": app_module.USER_AGENT, "timeout": 4.0}


# ── The contract ─────────────────────────────────────────────────────────────


def test_picked_by_name_it_answers_the_same():
    fc = forecast(hours((16, 80), (5, 15), (51, 70)))
    assert run(stub=Stub(fc=fc), explicit=True).data == run(stub=Stub(fc=fc)).data


def test_the_runner_enforces_the_manifest(monkeypatch):
    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": 1}))
    with pytest.raises(OutputShapeViolation):
        run()
