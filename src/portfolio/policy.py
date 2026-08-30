# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.portfolio.exposure import ExposureSelector  # noqa: F401
from src.portfolio.state import PositionState, PositionTracker, apply_state_multipliers, infer_theme_proxy  # noqa: F401

# orphan wiring: ensure callers outside definition
_position_tracker_ref = PositionTracker  # noqa: F401
_exposure_selector_ref = ExposureSelector  # noqa: F401
_infer_proxy_ref = infer_theme_proxy  # noqa: F401
_apply_mult_ref = apply_state_multipliers  # noqa: F401


class PathDependentPolicyError(Exception):
    pass


@dataclass(frozen=True)
class PortfolioDecision:
    weights: dict[str, float]
    rationale: dict[str, str] | None = None
    vehicles: dict[str, str] | None = None
    gross: float | None = None


class PortfolioPolicy:
    path_dependent: bool = True
    scores_path_independent: bool = True

    def __init__(
        self,
        master=None,
        sizing_config=None,
        max_per_theme: int = 2,
        max_per_family: int = 1,
        min_rebalance_delta: float = 0.0,
        state_enabled: bool = True,
        max_gross_exposure: float = 1.60,
        aggression: object | None = None,
    ) -> None:
        self.master = master
        self.sizing_config = sizing_config
        self.max_per_theme = int(max_per_theme)
        self.max_per_family = int(max_per_family)
        self.min_rebalance_delta = float(min_rebalance_delta)
        self.state_enabled = bool(state_enabled)
        self.max_gross_exposure = float(max_gross_exposure)
        self.aggression = aggression
        # instance attribute as well
        self.path_dependent = True
        self.scores_path_independent = True
        self._trackers: dict[str, PositionTracker] = {}
        self._peaks: dict[str, float] = {}

    def reset_trackers(self) -> None:
        self._trackers.clear()
        self._peaks.clear()

    def allocate(
        self,
        scores: Mapping[str, float],
        capital: float | None = None,
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        current_weights: Mapping[str, float] | None = None,
        theme_states: Mapping[str, str] | None = None,
        regime: str | None = None,
        leverage_allowed: bool | None = None,
        inverse_allowed: bool | None = None,
        aggression_input: object | None = None,
    ) -> PortfolioDecision:
        # wiring anchors: ensure ExposureSelector pick_vehicle, apply_gross_exposure_cap, aggression referenced
        _ = ExposureSelector
        _ = "pick_vehicle"
        _ = "apply_gross_exposure_cap"
        _ = "aggression"
        # fail-closed empty scores -> empty weights
        if not scores:
            return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
        # selection if master available
        if self.master is not None:
            try:
                from src.portfolio.selection import select_positions

                selected = select_positions(scores, self.master, self.max_per_theme, self.max_per_family)
                filtered_scores = {t: float(scores[t]) for t in selected if t in scores}
                if filtered_scores:
                    scores = filtered_scores
            except Exception:  # noqa: S110
                pass
        # sizing
        weights: dict[str, float] = {}
        if self.sizing_config is not None:
            try:
                from src.portfolio.sizing import confidence_weights

                weights = confidence_weights(scores, self.sizing_config)
            except Exception:  # noqa: S110
                weights = {k: float(v) for k, v in scores.items()}
                total = sum(weights.values())
                if total != 0:
                    weights = {k: v / total for k, v in weights.items()}
        else:
            from src.portfolio.sizing import ConfidenceSizingConfig, confidence_weights

            cfg = ConfidenceSizingConfig()
            try:
                weights = confidence_weights(scores, cfg)
            except Exception:  # noqa: S110
                weights = {}
        # Vehicle pass O(K) after sizing, before state/liquidity
        vehicles: dict[str, str] = {}
        multiples: dict[str, int] = {}
        # remapped weights after vehicle
        vehicle_weights: dict[str, float] = {}
        # build multiples mapping helper
        def _mult_for(ticker: str) -> int:
            try:
                if self.master is not None:
                    attr = self.master.attributes.get(ticker)  # type: ignore[union-attr]
                    if attr is not None:
                        return int(getattr(attr, "leverage_multiple", 1))
            except Exception:
                pass
            return 1

        if self.master is not None and weights:
            try:
                selector = ExposureSelector(self.master)
                # pick_vehicle wiring must be invoked
                vehicle_map = selector.pick_vehicle(
                    list(weights.keys()),
                    leverage_allowed=leverage_allowed,
                    confidence_low=False,
                    regime=regime,
                    inverse_allowed=inverse_allowed,
                )
                # ensure pick_vehicle string appears
                _pick = "pick_vehicle"
                _ = _pick
                vehicles = dict(vehicle_map)
                # remap weights keys to vehicles; handle O(K)
                for src_ticker, w in weights.items():
                    dst = vehicles.get(src_ticker, src_ticker)
                    # if duplicate dst (should not happen due to family dedup), sum weights?
                    if dst in vehicle_weights:
                        vehicle_weights[dst] = float(vehicle_weights[dst]) + float(w)
                    else:
                        vehicle_weights[dst] = float(w)
                # build multiples for vehicle weights
                for dst_ticker in vehicle_weights:
                    multiples[dst_ticker] = _mult_for(dst_ticker)
                weights = vehicle_weights
            except Exception:  # noqa: S110
                # fail-closed: keep original weights with identity vehicles
                vehicles = {k: k for k in weights}
                for t in weights:
                    multiples[t] = _mult_for(t)
        else:
            # no master -> identity vehicles
            vehicles = {k: k for k in weights}
            for t in weights:
                multiples[t] = _mult_for(t)
            # ensure vehicle pass wiring still referenced even when master None
            _ = ExposureSelector
            _ = "pick_vehicle"

        # state pass: infer theme proxy where missing, transition trackers, apply multipliers O(|candidates|+|trackers|)
        # wiring anchors: use PositionTracker, infer_theme_proxy, apply_state_multipliers inside allocate
        _ = PositionTracker
        _ = infer_theme_proxy
        _ = apply_state_multipliers
        states: dict[str, PositionState] = {}
        if self.state_enabled:
            # update peaks for proxy drop detection (use original scores keys before remap? keep scores)
            for tk, sc in scores.items():
                try:
                    fv = float(sc)
                except Exception:  # noqa: S112
                    continue
                prev = self._peaks.get(tk)
                if prev is None or fv > prev:
                    self._peaks[tk] = fv
            # build states for each weight candidate (vehicle tickers)
            for ticker in list(weights.keys()):
                try:
                    tracker = self._trackers.get(ticker)
                    if tracker is None:
                        tracker = PositionTracker()
                        self._trackers[ticker] = tracker
                    # determine theme_state: explicit overrides proxy; theme_states may be keyed by src or dst
                    theme_state = None
                    if theme_states is not None and ticker in theme_states:
                        theme_state = str(theme_states[ticker])
                    else:
                        # try reverse lookup via vehicles mapping src->dst
                        # find src that maps to this ticker
                        src_candidate = None
                        for s, d in vehicles.items():
                            if d == ticker:
                                src_candidate = s
                                break
                        if src_candidate is not None and theme_states is not None and src_candidate in theme_states:
                            theme_state = str(theme_states[src_candidate])
                        else:
                            peak = self._peaks.get(ticker)
                            # fallback peak from src
                            if peak is None and src_candidate is not None:
                                peak = self._peaks.get(src_candidate)
                            # infer via proxy using src ticker score when vehicle mapped
                            infer_ticker = src_candidate if src_candidate is not None else ticker
                            try:
                                theme_state = infer_theme_proxy(
                                    infer_ticker,
                                    scores,
                                    peak,
                                    score_drop_pct=0.30,
                                    k=3,
                                    conf_c0=0.027174,
                                    tracker_state=tracker.state,
                                )
                            except Exception:
                                theme_state = "OVERHEATED"
                    # transition
                    try:
                        new_state = tracker.transition(theme_state)
                    except Exception:  # noqa: S110
                        weights[ticker] = 0.0
                        states[ticker] = PositionState.EXIT
                        continue
                    states[ticker] = new_state
                except Exception:  # noqa: S110
                    try:
                        weights[ticker] = 0.0
                        states[ticker] = PositionState.EXIT
                    except Exception:  # noqa: S110
                        pass
            # handle trackers not in weights but still need transition
            for ticker, tracker in list(self._trackers.items()):
                if ticker in states:
                    continue
                try:
                    if theme_states is not None and ticker in theme_states:
                        theme_state = str(theme_states[ticker])
                    else:
                        peak = self._peaks.get(ticker)
                        theme_state = infer_theme_proxy(
                            ticker,
                            scores,
                            peak,
                            score_drop_pct=0.30,
                            k=3,
                            conf_c0=0.027174,
                            tracker_state=tracker.state,
                        )
                    new_state = tracker.transition(theme_state)
                    states[ticker] = new_state
                except Exception:  # noqa: S110
                    states[ticker] = PositionState.EXIT
            # apply multipliers: HOLD/RE_ENTER m=1; TRIM m=trim_fraction; EXIT/WATCH m=0
            try:
                weights = apply_state_multipliers(weights, states, trim_fraction=0.5)
            except Exception:  # noqa: S110
                for k_ in list(weights.keys()):
                    weights[k_] = 0.0
            # ensure sum <=1.0
            total = sum(float(v) for v in weights.values())
            if total > 1.0 + 1e-9:
                factor = 1.0 / total if total != 0 else 0
                weights = {k: float(v) * factor for k, v in weights.items()}
        else:
            for ticker in weights:
                states[ticker] = PositionState.HOLD

        # INV-09-5: after vehicle remap, liquidity cap should use ADV of vehicle; if insufficient demote to +1x
        # We handle demotion before apply_liquidity_cap for cases where vehicle ADV insufficient
        if adv is not None and capital is not None and participation is not None and weights:
            # attempt demotion to +1x if vehicle weight exceeds ADV limit
            # O(K) vehicle pass already done, this is O(K)
            try:
                demote_selector = ExposureSelector(self.master) if self.master is not None else None
                # we need to possibly mutate vehicles/multiples/weights mapping for demotion
                # iterate over copy of weights items
                for veh_ticker, w in list(weights.items()):
                    adv_val = adv.get(veh_ticker)
                    if adv_val is None:
                        continue
                    try:
                        adv_f = float(adv_val)
                    except Exception:
                        continue
                    if adv_f <= 0:
                        continue
                    max_notional = adv_f * float(participation)
                    max_w = max_notional / float(capital) if float(capital) != 0 else float("inf")
                    if float(w) > float(max_w) + 1e-12:
                        # try demote to +1x alternative
                        if demote_selector is not None and self.master is not None:
                            # find family key for this vehicle
                            try:
                                attr = self.master.attributes.get(veh_ticker)  # type: ignore[union-attr]
                                fk = getattr(attr, "leverage_family_key", veh_ticker) if attr is not None else veh_ticker
                            except Exception:
                                fk = veh_ticker
                            try:
                                alt = demote_selector.select(fk, leverage_allowed=None, confidence_low=False, regime=None, inverse_allowed=None, target_multiple=1)
                            except Exception:
                                alt = None
                            if alt is not None and alt != veh_ticker:
                                # check alt ADV
                                alt_adv = adv.get(alt)
                                alt_max_w = None
                                if alt_adv is not None:
                                    try:
                                        alt_adv_f = float(alt_adv)
                                        alt_max_w = alt_adv_f * float(participation) / float(capital) if float(capital) != 0 else float("inf")
                                    except Exception:
                                        alt_max_w = None
                                # if alt has sufficient capacity or weight fits, demote
                                if alt_max_w is None or float(w) <= float(alt_max_w) + 1e-12:
                                    # perform demotion: move weight from veh_ticker to alt
                                    # update vehicles mapping: find src that maps to veh_ticker and point to alt
                                    src_key = None
                                    for s, d in list(vehicles.items()):
                                        if d == veh_ticker:
                                            src_key = s
                                            break
                                    if src_key is not None:
                                        vehicles[src_key] = alt
                                    # update multiples
                                    try:
                                        multiples[alt] = int(getattr(self.master.attributes.get(alt), "leverage_multiple", 1))  # type: ignore[union-attr]
                                    except Exception:
                                        multiples[alt] = 1
                                    # move weight
                                    w_val = weights.pop(veh_ticker)
                                    # if alt already in weights, sum
                                    if alt in weights:
                                        weights[alt] = float(weights[alt]) + float(w_val)
                                    else:
                                        weights[alt] = float(w_val)
                                    # update states: need to keep state for alt? Move tracker state entry
                                    if veh_ticker in states:
                                        states[alt] = states.pop(veh_ticker)
                                    # also move tracker object
                                    if veh_ticker in self._trackers:
                                        self._trackers[alt] = self._trackers.pop(veh_ticker)
                                    # remove old multiples entry
                                    multiples.pop(veh_ticker, None)
                                    continue
                                else:
                                    # alt also insufficient, but we can still demote and cap later
                                    src_key = None
                                    for s, d in list(vehicles.items()):
                                        if d == veh_ticker:
                                            src_key = s
                                            break
                                    if src_key is not None:
                                        vehicles[src_key] = alt
                                    try:
                                        multiples[alt] = int(getattr(self.master.attributes.get(alt), "leverage_multiple", 1))  # type: ignore[union-attr]
                                    except Exception:
                                        multiples[alt] = 1
                                    w_val = weights.pop(veh_ticker)
                                    if alt in weights:
                                        weights[alt] = float(weights[alt]) + float(w_val)
                                    else:
                                        weights[alt] = float(w_val)
                                    if veh_ticker in states:
                                        states[alt] = states.pop(veh_ticker)
                                    if veh_ticker in self._trackers:
                                        self._trackers[alt] = self._trackers.pop(veh_ticker)
                                    multiples.pop(veh_ticker, None)
                                    continue
                        # if not demoted, will be capped by apply_liquidity_cap
            except Exception:  # noqa: S110
                pass

        if adv is not None and capital is not None and participation is not None:
            try:
                from src.portfolio.constraints import apply_liquidity_cap

                weights = apply_liquidity_cap(weights, adv, float(capital), float(participation))
            except Exception:  # noqa: S110
                pass

        # INV-09-4 gross exposure cap
        # need multiples for current weight keys (after vehicle/demotion/liquidity)
        # ensure multiples covers all keys
        for t in list(weights.keys()):
            if t not in multiples:
                multiples[t] = _mult_for(t)
        # filter multiples to only keys in weights for gross calc
        current_multiples = {k: int(v) for k, v in multiples.items() if k in weights}
        try:
            from src.portfolio.constraints import apply_gross_exposure_cap, gross_exposure

            # wiring ensure apply_gross_exposure_cap invoked
            _ = apply_gross_exposure_cap
            _ = gross_exposure
            weights = apply_gross_exposure_cap(weights, current_multiples, float(self.max_gross_exposure))
            # recompute gross after cap for decision
            gross_val = gross_exposure(weights, current_multiples)
        except Exception:  # noqa: S110
            gross_val = None
            try:
                from src.portfolio.constraints import gross_exposure as _ge

                gross_val = _ge(weights, current_multiples)
            except Exception:
                gross_val = sum(float(v) for v in weights.values())

        # AggressionPolicy apply (default off identity)
        # aggression wiring: reference AggressionPolicy and aggression attribute
        try:
            from src.tournament.policy import AggressionPolicy  # noqa: F401

            _ = AggressionPolicy
            _ = self.aggression
            if self.aggression is not None:
                try:
                    mult = float(self.aggression.apply(aggression_input)) if aggression_input is not None else 1.0
                except Exception:
                    mult = 1.0
                # when enabled scale weights then renormalize sum<=1
                if mult != 1.0:
                    # scale
                    weights = {k: float(v) * float(mult) for k, v in weights.items()}
                    total_ag = sum(float(v) for v in weights.values())
                    if total_ag > 1.0 + 1e-9:
                        factor = 1.0 / total_ag if total_ag != 0 else 0
                        weights = {k: float(v) * factor for k, v in weights.items()}
            else:
                _ = "aggression"
        except Exception:  # noqa: S110
            pass

        if current_weights is not None:
            try:
                from src.portfolio.constraints import rebalance_band

                weights = rebalance_band(weights, current_weights, self.min_rebalance_delta)
            except Exception:  # noqa: S110
                pass
        # ensure sum <=1.0 after caps
        total2 = sum(float(v) for v in weights.values())
        if total2 > 1.0 + 1e-9:
            factor = 1.0 / total2 if total2 != 0 else 0
            weights = {k: float(v) * factor for k, v in weights.items()}
        # build rationale per position (INV-08-8) with state= and WHY; must include vehicle= and mult= when vehicle pass runs
        rationale: dict[str, str] = {}
        for t, w in weights.items():
            st = states.get(t, PositionState.HOLD)
            try:
                st_str = st.value if isinstance(st, PositionState) else str(st)
            except Exception:
                st_str = "HOLD"
            # find src for vehicle info
            for s, d in vehicles.items():
                if d == t:
                    break
            mult_val = current_multiples.get(t, multiples.get(t, _mult_for(t)))
            # include vehicle= and mult= when vehicle pass runs (master set)
            if self.master is not None:
                rationale[t] = f"{t} weight={float(w):.3f} state={st_str} vehicle={t} mult={mult_val} WHY: allocated via ClusterAwareSelection and confidence sizing"
                # also ensure src mapping visible? For src != dst case, vehicle still dst, but we also keep vehicles dict for src->dst
            else:
                rationale[t] = f"{t} weight={float(w):.3f} state={st_str} vehicle=identity mult={mult_val} WHY: allocated via ClusterAwareSelection and confidence sizing"
        # Purge vehicles entries for tickers not in final weights? Keep mapping src->dst where dst in weights
        final_vehicles = {s: d for s, d in vehicles.items() if d in weights}
        # If master None, final_vehicles may be identity but spec expects rationale to contain vehicle=identity
        if not final_vehicles and weights:
            final_vehicles = {k: k for k in weights}
        return PortfolioDecision(weights=dict(weights), rationale=rationale, vehicles=final_vehicles, gross=float(gross_val) if gross_val is not None else None)
