"""
Enrichment Layer for QuantumInputVector
========================================
Fills the two known data gaps in Phase 0:
  - yield_curve_slope: US 10Y-2Y Treasury spread
  - dxy_momentum: Dollar Index 5-day price series

Data sources (stdlib-only, no pip dependencies):
  - FRED API (free, requires API key) for Treasury yields
  - Yahoo Finance chart API (no key needed) for DXY

Usage:
    from prototypes.other.quantum_signal_gen1_infra.enrichment import MarketDataEnricher

    enricher = MarketDataEnricher(fred_api_key="YOUR_KEY")  # optional
    enricher.refresh()  # fetches latest data

    # Inject into macro payload before building vector
    macro_enriched = enricher.enrich_macro(macro_payload)
    dxy_prices = enricher.get_dxy_prices_5d()

Design:
    - Fetch once, cache locally for configurable TTL (default 1 hour)
    - Falls back gracefully to None/0.0 on network failure
    - Never modifies the original payload — returns enriched copy
    - Zero external dependencies (uses urllib.request)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


class MarketDataEnricher:
    """
    Fetches and caches yield curve and DXY data to fill Phase 0 gaps.

    The enricher is a standalone utility — it does not modify the
    quantum_signal package or the QuantumInputVector contract.
    It provides data that the builder already knows how to consume
    (us10y_yield, us2y_yield in macro dict; dxy_prices_5d as a list).
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Args:
            fred_api_key: FRED API key. If None, reads from FRED_API_KEY env var.
                          If still None, yield curve enrichment is skipped.
            cache_dir: Directory for local cache files. Defaults to
                       ~/.quantum_signal/cache/
            cache_ttl_seconds: How long cached data remains valid. Default 1 hour.
        """
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        self.cache_ttl = cache_ttl_seconds

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".quantum_signal" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._us10y: Optional[float] = None
        self._us2y: Optional[float] = None
        self._dxy_prices: Optional[List[float]] = None
        self._last_refresh: float = 0
        self._refresh_errors: List[str] = []

    def refresh(self) -> Dict[str, Any]:
        """
        Fetch latest yield curve and DXY data.
        Returns a status dict with what was fetched/failed.
        """
        self._refresh_errors = []
        status = {"yields_ok": False, "dxy_ok": False, "errors": []}

        # Try yields
        try:
            self._fetch_treasury_yields()
            status["yields_ok"] = self._us10y is not None and self._us2y is not None
        except Exception as e:
            err = f"yield fetch failed: {e}"
            self._refresh_errors.append(err)
            status["errors"].append(err)
            logger.warning(err)

        # Try DXY
        try:
            self._fetch_dxy_prices()
            status["dxy_ok"] = self._dxy_prices is not None and len(self._dxy_prices) >= 2
        except Exception as e:
            err = f"DXY fetch failed: {e}"
            self._refresh_errors.append(err)
            status["errors"].append(err)
            logger.warning(err)

        self._last_refresh = time.time()
        self._save_cache()

        status["us10y"] = self._us10y
        status["us2y"] = self._us2y
        status["spread"] = round(self._us10y - self._us2y, 4) if self._us10y and self._us2y else None
        status["dxy_prices_count"] = len(self._dxy_prices) if self._dxy_prices else 0
        status["cached_at"] = datetime.now(timezone.utc).isoformat()

        return status

    def is_fresh(self) -> bool:
        """Check if cached data is still within TTL."""
        return (time.time() - self._last_refresh) < self.cache_ttl

    def enrich_macro(self, macro_payload: Dict) -> Dict:
        """
        Return a copy of the macro payload with yield curve data injected.
        Does NOT mutate the original.

        Adds:
          - us10y_yield: float or None
          - us2y_yield: float or None

        The builder already handles these fields (see builder.py lines 171-179).
        """
        enriched = dict(macro_payload)

        if not self.is_fresh():
            self._load_cache()

        if self._us10y is not None:
            enriched["us10y_yield"] = self._us10y
        if self._us2y is not None:
            enriched["us2y_yield"] = self._us2y

        return enriched

    def get_dxy_prices_5d(self) -> Optional[List[float]]:
        """
        Return the cached 5-day DXY close prices.
        The builder accepts this as the `dxy_prices_5d` parameter.
        """
        if not self.is_fresh():
            self._load_cache()
        return self._dxy_prices

    def get_status(self) -> Dict[str, Any]:
        """Return current enrichment state for health checks."""
        return {
            "us10y": self._us10y,
            "us2y": self._us2y,
            "spread": round(self._us10y - self._us2y, 4) if self._us10y and self._us2y else None,
            "dxy_prices": self._dxy_prices,
            "dxy_count": len(self._dxy_prices) if self._dxy_prices else 0,
            "is_fresh": self.is_fresh(),
            "last_refresh": datetime.fromtimestamp(self._last_refresh, tz=timezone.utc).isoformat() if self._last_refresh else None,
            "errors": self._refresh_errors,
        }

    # ─── Private: Data Fetching ───

    def _fetch_treasury_yields(self):
        """
        Fetch US 10Y and 2Y yields from FRED.
        Series: DGS10 (10-Year), DGS2 (2-Year).

        If no FRED API key is available, try Yahoo Finance as fallback.
        """
        if self.fred_api_key:
            self._fetch_yields_fred()
        else:
            self._fetch_yields_yahoo_fallback()

    def _fetch_yields_fred(self):
        """Fetch yields from FRED API."""
        base = "https://api.stlouisfed.org/fred/series/observations"

        for series_id, attr in [("DGS10", "_us10y"), ("DGS2", "_us2y")]:
            params = (
                f"?series_id={series_id}"
                f"&api_key={self.fred_api_key}"
                f"&sort_order=desc&limit=5"
                f"&file_type=json"
            )
            url = base + params

            try:
                req = Request(url, headers={"User-Agent": "quantum_signal/1.0"})
                with urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

                observations = data.get("observations", [])
                # Find latest non-empty value
                for obs in observations:
                    val = obs.get("value", ".")
                    if val != ".":
                        setattr(self, attr, float(val))
                        break
            except (URLError, HTTPError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"FRED {series_id} fetch failed: {e}")

    def _fetch_yields_yahoo_fallback(self):
        """
        Fallback: fetch Treasury yield ETF proxies from Yahoo Finance.
        ^TNX = 10Y yield, ^IRX approximation not available — skip 2Y.
        This is a degraded path; FRED API key is preferred.
        """
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=5d&interval=1d"
            req = Request(url, headers={"User-Agent": "quantum_signal/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            result = data["chart"]["result"][0]
            closes = result["indicators"]["quote"][0]["close"]
            valid = [c for c in closes if c is not None]
            if valid:
                self._us10y = valid[-1]
                logger.info(f"Yahoo TNX fallback: 10Y = {self._us10y}")
        except Exception as e:
            logger.warning(f"Yahoo TNX fallback failed: {e}")

    def _fetch_dxy_prices(self):
        """
        Fetch 5-day DXY (Dollar Index) closing prices from Yahoo Finance.
        Ticker: DX-Y.NYB (ICE Dollar Index).
        """
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=5d&interval=1d"
            req = Request(url, headers={"User-Agent": "quantum_signal/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            result = data["chart"]["result"][0]
            closes = result["indicators"]["quote"][0]["close"]
            valid = [round(c, 4) for c in closes if c is not None]

            if len(valid) >= 2:
                self._dxy_prices = valid
                logger.info(f"DXY: {len(valid)} prices fetched, latest={valid[-1]}")
            else:
                logger.warning(f"DXY: insufficient data ({len(valid)} points)")
        except Exception as e:
            logger.warning(f"DXY fetch failed: {e}")

    # ─── Private: Local Cache ───

    def _save_cache(self):
        """Persist current state to local JSON cache."""
        cache_path = self.cache_dir / "enrichment_cache.json"
        try:
            cache_data = {
                "us10y": self._us10y,
                "us2y": self._us2y,
                "dxy_prices": self._dxy_prices,
                "last_refresh": self._last_refresh,
                "errors": self._refresh_errors,
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _load_cache(self):
        """Load state from local JSON cache if available and fresh."""
        cache_path = self.cache_dir / "enrichment_cache.json"
        try:
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
                cached_time = data.get("last_refresh", 0)
                if (time.time() - cached_time) < self.cache_ttl:
                    self._us10y = data.get("us10y")
                    self._us2y = data.get("us2y")
                    self._dxy_prices = data.get("dxy_prices")
                    self._last_refresh = cached_time
                    self._refresh_errors = data.get("errors", [])
                    return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        return False


def enrich_snapshot(
    snapshot: Dict[str, Any],
    enricher: MarketDataEnricher,
) -> Dict[str, Any]:
    """
    Convenience function: enrich a full agent snapshot dict.

    Takes a snapshot with keys {technical, macro, news, fundamental, risk_sentiment}
    and returns an enriched copy where:
      - macro has us10y_yield and us2y_yield injected
      - a new top-level key 'dxy_prices_5d' is added

    The builder.build() method already accepts both macro.us10y_yield/us2y_yield
    and dxy_prices_5d as parameters, so no contract change needed.
    """
    enriched = dict(snapshot)
    enriched["macro"] = enricher.enrich_macro(snapshot.get("macro", {}))
    dxy = enricher.get_dxy_prices_5d()
    if dxy:
        enriched["_dxy_prices_5d"] = dxy
    enriched["_enrichment_status"] = enricher.get_status()
    return enriched
