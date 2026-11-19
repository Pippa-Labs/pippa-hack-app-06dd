"""Dry Spell: when the rain stops at a named place, and how long it stays dry.

Two requests to Open-Meteo, neither needing a key:

1. its geocoder turns the place name into coordinates, and
2. its forecast gives the hourly chance of rain, rainfall, temperature and
   wind there for the next three days, in the place's own time zone.

The app then walks the hours from now and returns the first unbroken run of
"dry" hours at least as long as the one asked for. "Dry" is a stated rule, not
a score: every hour has a rain chance of THRESHOLD_PERCENT or less and no rain
forecast in it. The rule goes back in the output, so the answer is checkable.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
    load_manifest_file,
    validate_manifest,
)

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "pippa-agent-app-dry-spell/0.1 (https://github.com/Pippa-Labs)"

#: An hour is dry when its chance of rain is at most this and no rain falls.
THRESHOLD_PERCENT = 20
#: Rainfall below this many millimetres in an hour counts as none.
RAIN_MM = 0.1
DEFAULT_HOURS = 1.0
MAX_HOURS = 24.0
FORECAST_DAYS = 3
#: The geocoder sorts by size, so the first of these is the best-known place
#: of that name; more are only needed to find one in the region asked for.
CANDIDATES = 10
GEOCODE_TIMEOUT = 5.0
FORECAST_TIMEOUT = 6.0
HOUR = timedelta(hours=1)

#: What people call a country, mapped to the code the geocoder returns.
COUNTRY_ALIASES = {
    "uk": "GB", "u.k.": "GB", "united kingdom": "GB", "britain": "GB",
    "great britain": "GB", "gb": "GB",
    "us": "US", "u.s.": "US", "usa": "US", "u.s.a.": "US", "america": "US",
    "united states": "US", "united states of america": "US",
}


class NoPlaceGiven(AgentAppError):
    user_message = (
        "Tell me which town or city to check, for example "
        "\"when will it stop raining in Leeds\"."
    )
    retryable = False


class PlaceNotFound(AgentAppError):
    user_message = "I could not find anywhere with that name."
    retryable = False


class LengthOutOfRange(AgentAppError):
    user_message = f"I can look for a dry spell of up to {MAX_HOURS:g} hours."
    retryable = False


@dataclass(frozen=True)
class Place:
    label: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Hour:
    start: datetime          # aware, UTC
    rain_chance: int | None
    rain_mm: float | None
    temp_c: float | None
    wind_kph: float | None

    @property
    def end(self) -> datetime:
        return self.start + HOUR

    @property
    def dry(self) -> bool:
        if self.rain_chance is None or self.rain_chance > THRESHOLD_PERCENT:
            return False
        return self.rain_mm is None or self.rain_mm < RAIN_MM


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _fetch(self, url: str, timeout: float = 6.0) -> bytes:
        """Every outbound request goes through here, so tests replace it."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    # ── The call ─────────────────────────────────────────────────────────────

    def handle(self, call: Call) -> Result:
        if call.outcome_id != "dry.window":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")

        name, region = self._place_words(call.inputs.get("place"), call.inputs.get("region"))
        needed = self._hours_needed(call.inputs.get("hours_needed"))
        now: datetime = call.context.now.astimezone(timezone.utc)

        place = self._geocode(name, region)
        zone, hours = self._forecast(place)
        hours = [h for h in hours if h.end > now]
        if not hours:
            raise ServiceUnavailableError("the forecast had no hours after now")
        horizon = hours[-1].end

        data = {
            "place_used": place.label,
            "timezone_used": getattr(zone, "key", None) or str(zone),
            "found": False,
            "dry_now": False,
            "hours_found": 0.0,
            "dry_beyond_forecast": False,
            "hours_needed": needed,
            "threshold_percent": THRESHOLD_PERCENT,
            "hours_searched": round((horizon - now) / HOUR),
        }

        spell = self._first_spell(hours, now, needed)
        if spell is not None:
            run = spell
            start = max(run[0].start, now)
            end = run[-1].end
            data.update(
                found=True,
                dry_now=start == now,
                start=_clock(start, zone),
                end=_clock(end, zone),
                hours_found=round((end - start) / HOUR, 1),
                dry_beyond_forecast=end >= horizon,
                max_rain_chance_percent=max(h.rain_chance for h in run),
            )
            for field, values, pick in (
                ("max_temp_c", [h.temp_c for h in run], max),
                ("min_temp_c", [h.temp_c for h in run], min),
                ("max_wind_kph", [h.wind_kph for h in run], max),
            ):
                known = [v for v in values if v is not None]
                if known:
                    data[field] = round(pick(known), 1)
        else:
            best = self._least_wet(hours, now, needed)
            if best is not None:
                when, chance = best
                data["least_wet_start"] = _clock(when, zone)
                data["least_wet_rain_chance_percent"] = chance

        return Result(data=data, narrative=_sentence(data, name))

    # ── Inputs ───────────────────────────────────────────────────────────────

    @staticmethod
    def _place_words(place, region) -> tuple[str, str]:
        """The bare place name, and any region given with it.

        The geocoder finds "Bristol" but answers nothing for "Bristol, UK", so
        anything after a comma moves to the region when no region was given.
        """
        name = " ".join(str(place or "").split())
        hint = " ".join(str(region or "").split())
        if "," in name:
            name, _, rest = name.partition(",")
            name = name.strip()
            hint = hint or rest.strip()
        if not name:
            raise NoPlaceGiven("no place in the message")
        return name, hint

    @staticmethod
    def _hours_needed(value) -> float:
        if value is None or value == "":
            return DEFAULT_HOURS
        hours = float(value)
        if not math.isfinite(hours) or hours <= 0 or hours > MAX_HOURS:
            error = LengthOutOfRange(f"hours_needed out of range: {value!r}")
            if math.isfinite(hours) and hours > MAX_HOURS:
                error.user_message = (
                    f"I can look for a dry spell of up to {MAX_HOURS:g} hours, "
                    f"so {hours:g} hours is too long."
                )
            else:
                error.user_message = "Tell me how many hours of dry weather you need."
            raise error
        return round(hours, 2)

    # ── Open-Meteo ───────────────────────────────────────────────────────────

    def _geocode(self, name: str, region: str) -> Place:
        url = GEOCODE_URL + "?" + urlencode(
            {"name": name, "count": CANDIDATES, "language": "en", "format": "json"}
        )
        body = self._get_json(url, GEOCODE_TIMEOUT, lambda: _not_found(name, region))
        # An unknown name is a 200 with no `results` key at all.
        results = body.get("results") if isinstance(body, dict) else None
        results = [r for r in results or [] if _has_coordinates(r)]
        if region:
            results = [r for r in results if _in_region(r, region)]
        if not results:
            raise _not_found(name, region)
        best = results[0]
        parts = [best.get("name") or name]
        for key in ("admin1", "country"):
            part = best.get(key)
            if part and part not in parts:
                parts.append(part)
        return Place(", ".join(parts), float(best["latitude"]), float(best["longitude"]))

    def _forecast(self, place: Place) -> tuple[tzinfo, list[Hour]]:
        url = FORECAST_URL + "?" + urlencode({
            "latitude": f"{place.latitude:.4f}",
            "longitude": f"{place.longitude:.4f}",
            "hourly": "precipitation_probability,precipitation,temperature_2m,wind_speed_10m",
            "timezone": "auto",
            "timeformat": "unixtime",
            "forecast_days": FORECAST_DAYS,
        })
        body = self._get_json(
            url, FORECAST_TIMEOUT,
            lambda: ServiceUnavailableError(f"forecast refused {place.label!r}"),
        )
        try:
            hourly = body["hourly"]
            times = hourly["time"]
            columns = [hourly.get(k) or [] for k in (
                "precipitation_probability", "precipitation", "temperature_2m", "wind_speed_10m")]
            hours = [
                Hour(
                    start=datetime.fromtimestamp(int(t), timezone.utc),
                    rain_chance=_int_or_none(_at(columns[0], i)),
                    rain_mm=_float_or_none(_at(columns[1], i)),
                    temp_c=_float_or_none(_at(columns[2], i)),
                    wind_kph=_float_or_none(_at(columns[3], i)),
                )
                for i, t in enumerate(times)
            ]
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
            raise ServiceUnavailableError(f"forecast came back in an unexpected shape: {exc}")
        return _zone(body), hours

    def _get_json(self, url: str, timeout: float, rejected):
        try:
            raw = self._fetch(url, timeout=timeout)
        except urllib.error.HTTPError as exc:      # before URLError: it IS one
            if exc.code in (400, 404, 422):
                raise rejected()
            if exc.code == 429:
                raise RateLimitedError(
                    f"open-meteo rate limited: {exc.code}",
                    retry_after_seconds=_retry_after(exc),
                )
            raise ServiceUnavailableError(f"open-meteo returned {exc.code}")
        except TimeoutError as exc:                 # not a URLError
            raise ServiceUnavailableError(f"open-meteo did not answer in time: {exc}")
        except urllib.error.URLError as exc:
            raise ServiceUnavailableError(f"could not reach open-meteo: {exc.reason}")
        except OSError as exc:
            raise ServiceUnavailableError(f"connection to open-meteo failed: {exc}")
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ServiceUnavailableError(f"open-meteo sent something that is not JSON: {exc}")

    # ── The search ───────────────────────────────────────────────────────────

    @staticmethod
    def _first_spell(hours: list[Hour], now: datetime, needed: float) -> list[Hour] | None:
        """The first unbroken run of dry hours lasting at least `needed` from now.

        A run is followed to its end, so the answer says how long it lasts,
        not only that it is long enough. The hour we are in counts from now,
        not from the top of the hour.
        """
        run: list[Hour] = []
        for hour in hours + [None]:
            if hour is not None and hour.dry and (not run or hour.start == run[-1].end):
                run.append(hour)
                continue
            if run and (run[-1].end - max(run[0].start, now)) / HOUR >= needed:
                return run
            run = [hour] if hour is not None and hour.dry else []
        return None

    @staticmethod
    def _least_wet(hours: list[Hour], now: datetime, needed: float):
        """(start, worst chance) of the stretch of `needed` hours least likely to rain.

        Earliest wins a tie. Hours with no published chance are skipped over.
        """
        span = max(1, math.ceil(needed))
        best = None
        for i in range(len(hours) - span + 1):
            window = hours[i:i + span]
            if any(h.rain_chance is None for h in window):
                continue
            if any(b.start != a.end for a, b in zip(window, window[1:])):
                continue
            worst = max(h.rain_chance for h in window)
            if best is None or worst < best[1]:
                best = (max(window[0].start, now), worst)
        return best


# ── Helpers ──────────────────────────────────────────────────────────────────


def _not_found(name: str, region: str) -> PlaceNotFound:
    error = PlaceNotFound(f"no geocoding match for {name!r} in {region!r}")
    error.user_message = (
        f"I could not find {name} in {region}." if region
        else f"I do not know where {name} is."
    )
    return error


def _has_coordinates(result) -> bool:
    return (
        isinstance(result, dict)
        and isinstance(result.get("latitude"), (int, float))
        and isinstance(result.get("longitude"), (int, float))
    )


def _in_region(result: dict, region: str) -> bool:
    wanted = region.strip().lower()
    code = COUNTRY_ALIASES.get(wanted)
    if code and (result.get("country_code") or "").upper() == code:
        return True
    if (result.get("country_code") or "").lower() == wanted:
        return True
    # "London" should find a place whose county is "Greater London".
    names = [result.get(k) for k in ("country", "admin1", "admin2")]
    return any(isinstance(n, str) and wanted in n.lower() for n in names)


def _zone(body: dict) -> tzinfo:
    """The place's own time zone, from the forecast."""
    name = body.get("timezone")
    if isinstance(name, str) and name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    offset = body.get("utc_offset_seconds")
    if isinstance(offset, int):
        return timezone(timedelta(seconds=offset))
    return timezone.utc


def _at(column: list, i: int):
    return column[i] if i < len(column) else None


def _int_or_none(value) -> int | None:
    return None if value is None else int(round(float(value)))


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _retry_after(exc: urllib.error.HTTPError) -> int | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _clock(moment: datetime, zone: tzinfo) -> str:
    """'Friday 14:00' in the place's own time. No %-d: it raises on Windows."""
    local = moment.astimezone(zone)
    return f"{local.strftime('%A')} {local.strftime('%H:%M')}"


def _hours_text(hours: float) -> str:
    if hours == 1:
        return "1 hour"
    return f"{hours:g} hours"


def _sentence(data: dict, asked: str) -> str:
    where = data["place_used"]
    rule = f"(dry here means a {data['threshold_percent']}% chance of rain or less, and none forecast)"
    if not data["found"]:
        line = (
            f"No dry spell of {_hours_text(data['hours_needed'])} in {where} "
            f"in the next {data['hours_searched']} hours {rule}."
        )
        if "least_wet_start" in data:
            line += (
                f" The least wet stretch starts {data['least_wet_start']}, "
                f"with up to {data['least_wet_rain_chance_percent']}% chance of rain."
            )
        return line
    until = (
        f"at least until {data['end']}, the end of the forecast"
        if data["dry_beyond_forecast"] else f"until {data['end']}"
    )
    lasts = _hours_text(data["hours_found"])
    if data["dry_now"]:
        return f"It is dry in {where} now and should stay dry {until} ({lasts}) {rule}."
    return (
        f"The next dry spell in {where} starts {data['start']} and lasts {until} "
        f"({lasts}) {rule}."
    )
