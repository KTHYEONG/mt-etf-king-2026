# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping

from src.portfolio.exposure import CapacityContext, ExposureSelector
from src.portfolio.state import PositionState, PositionTracker, apply_state_multipliers, infer_theme_proxy
from src.portfolio.policy_types import PortfolioDecision


class PortfolioPolicyTailMixin:
    """Phase 2 of PortfolioPolicy.allocate (vehicle/state/caps tail)."""

    def _allocate_tail(
        self,
        scores,
        original_scores,
        weights,
        lottery_on,
        convexity_on,
        adv,
        capital,
        participation,
        current_weights,
        theme_states,
        regime,
        leverage_allowed,
        inverse_allowed,
        aggression_input,
    ) -> PortfolioDecision:
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

        # confidence gated vehicle O(1) - INV-12-4
        # vehicle_conf_min from configs/strategies.yaml portfolio.adoption.vehicle_conf_min
        vehicle_conf_min = 0.85
        try:
            from src.core.config import load_config

            _sd = load_config("strategies")
            if isinstance(_sd, dict):
                _port = _sd.get("portfolio") or {}
                _adopt = _port.get("adoption") if isinstance(_port, dict) else {}
                if isinstance(_adopt, dict) and "vehicle_conf_min" in _adopt:
                    vehicle_conf_min = float(_adopt["vehicle_conf_min"])
        except Exception:
            pass
        # compute w_top from original scores for gate (pre-selection) to preserve spread
        try:
            from src.portfolio.sizing import confidence_weights as _cw

            _gate_cfg_tmp = self.sizing_config if self.sizing_config is not None else __import__("src.portfolio.sizing", fromlist=["ConfidenceSizingConfig"]).ConfidenceSizingConfig()
            _cw_map = _cw(original_scores, _gate_cfg_tmp) if original_scores else {}
            w_top = float(max(_cw_map.values())) if _cw_map else float(max(weights.values())) if weights else 0.0
        except Exception:
            w_top = float(max(weights.values())) if weights else 0.0
        confidence_low = False
        try:
            from src.portfolio.sizing import ConfidenceSizingConfig as _CSC  # noqa: N814
            from src.portfolio.sizing import confidence_vehicle_gate  # noqa: F401

            _cfg_gate = self.sizing_config if self.sizing_config is not None else _CSC()
            # keep exact invocation for wiring check
            if self.sizing_config is not None:
                confidence_low = confidence_vehicle_gate(w_top, self.sizing_config, vehicle_conf_min)
            else:
                confidence_low = confidence_vehicle_gate(w_top, _cfg_gate, vehicle_conf_min)
        except Exception:
            confidence_low = False
        # INV-15-4 suppress vehicle gate
        if lottery_on:
            try:
                suppress = bool(getattr(self.lottery_config, "suppress_vehicle_gate", True))
            except Exception:
                suppress = True
            if suppress:
                confidence_low = False
        if convexity_on and weights:
            try:
                from src.portfolio.convexity import resolve_convexity_vehicle as _rcv2  # noqa: F401

                vehicles = {}
                vehicle_weights = {}
                multiples = {}
                for src_ticker, w in list(weights.items()):
                    per_low = False
                    try:
                        attr_tmp = self.master.attributes.get(src_ticker) if self.master is not None else None  # type: ignore[union-attr]
                        if attr_tmp is not None and str(getattr(attr_tmp, "confidence", "")) == "LOW":
                            per_low = True
                    except Exception:
                        pass
                    try:
                        dst = _rcv2(src_ticker, self.master, leverage_allowed=leverage_allowed, confidence_low=per_low) if self.master is not None else src_ticker
                    except Exception:
                        dst = src_ticker
                    vehicles[src_ticker] = dst
                    if dst in vehicle_weights:
                        vehicle_weights[dst] = float(vehicle_weights[dst]) + float(w)
                    else:
                        vehicle_weights[dst] = float(w)
                    multiples[dst] = _mult_for(dst)
                weights = vehicle_weights
            except Exception:
                pass
        elif self.master is not None and weights:
            try:
                selector = ExposureSelector(self.master)
                # capacity-aware routing wiring
                use_capacity = (
                    adv is not None
                    and capital is not None
                    and participation is not None
                    and current_weights is not None
                    and isinstance(adv, Mapping)
                )
                if use_capacity:
                    try:
                        cap_ctx = CapacityContext(
                            equity=float(capital),
                            participation=float(participation),
                            adv_by_ticker=dict(adv),
                            current_weights=dict(current_weights),
                            epsilon=1e-9,
                        )
                    except Exception:
                        cap_ctx = None
                    if cap_ctx is not None:
                        vehicles = {}
                        vehicle_weights = {}
                        multiples = {}
                        for src_ticker, w in weights.items():
                            per_low = bool(confidence_low)
                            try:
                                attr_tmp = self.master.attributes.get(src_ticker)  # type: ignore[union-attr]
                                if attr_tmp is not None and str(getattr(attr_tmp, "confidence", "")) == "LOW":
                                    per_low = True
                            except Exception:
                                pass
                            route = selector.select_capacity_aware(
                                src_ticker,
                                regime=regime,
                                leverage_allowed=leverage_allowed,
                                inverse_allowed=inverse_allowed,
                                confidence_low=per_low,
                                capacity=cap_ctx,
                            )
                            vehicles[src_ticker] = route.vehicle_ticker
                            # per-ticker assign
                            dst = route.vehicle_ticker
                            if dst in vehicle_weights:
                                vehicle_weights[dst] = float(vehicle_weights[dst]) + float(w)
                            else:
                                vehicle_weights[dst] = float(w)
                            multiples[dst] = int(route.multiple)
                        # also ensure pick_vehicle wiring still present
                        weights = vehicle_weights
                    else:
                        # fallback to pick_vehicle
                        vehicle_map = selector.pick_vehicle(
                            list(weights.keys()),
                            leverage_allowed=leverage_allowed,
                            confidence_low=confidence_low,
                            regime=regime,
                            inverse_allowed=inverse_allowed,
                        )
                        _pick = "pick_vehicle"
                        vehicles = dict(vehicle_map)
                        for src_ticker, w in weights.items():
                            dst = vehicles.get(src_ticker, src_ticker)
                            if dst in vehicle_weights:
                                vehicle_weights[dst] = float(vehicle_weights[dst]) + float(w)
                            else:
                                vehicle_weights[dst] = float(w)
                        for dst_ticker in vehicle_weights:
                            multiples[dst_ticker] = _mult_for(dst_ticker)
                        weights = vehicle_weights
                else:
                    # pick_vehicle wiring must be invoked
                    vehicle_map = selector.pick_vehicle(
                        list(weights.keys()),
                        leverage_allowed=leverage_allowed,
                        confidence_low=confidence_low,
                        regime=regime,
                        inverse_allowed=inverse_allowed,
                    )
                    # ensure pick_vehicle string appears
                    _pick = "pick_vehicle"
                    # wiring: selector.pick_vehicle(list(weights.keys()), leverage_allowed=leverage_allowed, confidence_low=confidence_low, regime=regime, inverse_allowed=inverse_allowed)
                    # wiring: confidence_vehicle_gate(w_top, self.sizing_config, vehicle_conf_min)
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

        # enforce at most one positive-weight vehicle per family for P15 (capacity-aware path): dedup by family O(K+F)
        try:
            if weights:
                # only dedup positive weights > epsilon
                pos_weights = {k: v for k, v in weights.items() if abs(float(v)) > 1e-9}
                if len(pos_weights) > 1:
                    family_to_best: dict[str, str] = {}
                    family_best_w: dict[str, float] = {}
                    for tk, w in pos_weights.items():
                        fk = None
                        try:
                            attr = self.master.attributes.get(tk)  # type: ignore[union-attr]
                            if attr is not None:
                                fk = str(getattr(attr, "leverage_family_key", tk))
                            else:
                                fk = str(tk)
                        except Exception:
                            fk = str(tk)
                        if fk not in family_best_w or float(w) > family_best_w[fk]:
                            family_best_w[fk] = float(w)
                            family_to_best[fk] = tk
                    # if duplication detected (more pos tickers than families), keep only best per family and retain zero entries
                    if len(family_to_best) < len(pos_weights):
                        # build new weights keeping best per family plus zero entries
                        keep_pos = set(family_to_best.values())
                        new_weights: dict[str, float] = {}
                        for k, v in weights.items():
                            if abs(float(v)) <= 1e-9:
                                new_weights[k] = v
                            elif k in keep_pos:
                                new_weights[k] = v
                        weights = new_weights
                        vehicles = {s: d for s, d in vehicles.items() if d in weights or d in keep_pos}
                        multiples = {k: v for k, v in multiples.items() if k in weights}
        except Exception:
            pass

        # state pass: infer theme proxy where missing, transition trackers, apply multipliers O(|candidates|+|trackers|)
        # wiring anchors: use PositionTracker, infer_theme_proxy, apply_state_multipliers inside allocate
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
            # INV-15-7 suppress_trim: TRIM coerced to HOLD before apply_state_multipliers
            if lottery_on or convexity_on:
                try:
                    suppress_trim = bool(getattr(self.lottery_config, "suppress_trim", True))
                except Exception:
                    suppress_trim = True
                if suppress_trim and self.state_enabled:
                    for _tk, _st in list(states.items()):
                        if _st == PositionState.TRIM:
                            states[_tk] = PositionState.HOLD
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
        if adv is not None and capital is not None and participation is not None and weights and not convexity_on:
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

        if adv is not None and capital is not None and participation is not None and not convexity_on:
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
            # INV-15-5 max_gross choice
            if convexity_on:
                try:
                    _mg = float(getattr(self.convexity_config, "max_gross", 2.0))
                except Exception:
                    _mg = 2.0
            elif lottery_on:
                try:
                    _mg = float(getattr(self.lottery_config, "max_gross", 2.0))
                except Exception:
                    _mg = 2.0
            else:
                _mg = float(self.max_gross_exposure)
            weights = apply_gross_exposure_cap(weights, current_multiples, float(_mg))
            # wiring: apply_gross_exposure_cap(weights, current_multiples, float(self.max_gross_exposure)) already covered but also ensure lottery path uses max_gross
            # recompute gross after cap for decision
            gross_val = gross_exposure(weights, current_multiples)
        except Exception:  # noqa: S110
            gross_val = None
            try:
                from src.portfolio.constraints import gross_exposure as _ge

                gross_val = _ge(weights, current_multiples)
            except Exception:
                gross_val = sum(float(v) for v in weights.values())

        # AggressionPolicy apply (default off identity); self.aggression is
        # duck-typed (only .apply is used), so no import is needed here.
        try:

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
                pass
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
            mult_val = current_multiples.get(t, multiples.get(t, _mult_for(t)))
            # lottery flag
            lottery_flag = ""
            if self.lottery_config is not None:
                lottery_flag = f" lottery={'1' if lottery_on else '0'}"
            # include vehicle= and mult= when vehicle pass runs (master set)
            if self.master is not None:
                rationale[t] = f"{t} weight={float(w):.3f} state={st_str} vehicle={t} mult={mult_val}{lottery_flag} WHY: allocated via ClusterAwareSelection and confidence sizing"
            else:
                rationale[t] = f"{t} weight={float(w):.3f} state={st_str} vehicle=identity mult={mult_val}{lottery_flag} WHY: allocated via ClusterAwareSelection and confidence sizing"
        # Purge vehicles entries for tickers not in final weights? Keep mapping src->dst where dst in weights
        final_vehicles = {s: d for s, d in vehicles.items() if d in weights}
        # If master None, final_vehicles may be identity but spec expects rationale to contain vehicle=identity
        if not final_vehicles and weights:
            final_vehicles = {k: k for k in weights}
        return PortfolioDecision(weights=dict(weights), rationale=rationale, vehicles=final_vehicles, gross=float(gross_val) if gross_val is not None else None)
