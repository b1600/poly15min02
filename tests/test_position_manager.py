import math

from poly15m.config import Settings
from poly15m.positions.manager import PositionManager, kelly_fraction
from poly15m.pricing.fair_value import FairValue

SETTINGS = Settings()


def make_pm():
    return PositionManager(Settings())


def test_kelly_fraction_matches_manual_formula():
    assert math.isclose(kelly_fraction(0.7, 0.5), 0.4)


def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(0.4, 0.5) == 0.0  # fair value below price -> negative edge -> clamp to 0


def test_kelly_fraction_zero_at_price_boundaries():
    assert kelly_fraction(0.9, 0.0) == 0.0
    assert kelly_fraction(0.9, 1.0) == 0.0


def test_record_fill_updates_inventory():
    pm = make_pm()
    pm.record_fill("cond1", "up", 0.5, 20.0, 0.1)
    inv = pm.get_inventory("cond1")
    assert inv.up.size == 20.0
    assert math.isclose(inv.up.cost_basis, 10.1)
    assert inv.down.size == 0.0
    assert inv.matched_size == 0.0
    assert inv.net_directional == 20.0


def test_matched_arb_triggers_when_combined_ask_under_one():
    pm = make_pm()
    asks_up = [(0.45, 100.0)]
    asks_down = [(0.45, 100.0)]
    intents = pm._check_matched_arb("cond1", "tok_up", "tok_down", asks_up, asks_down)
    assert len(intents) == 2
    up_intent = next(i for i in intents if i.outcome == "up")
    down_intent = next(i for i in intents if i.outcome == "down")
    assert up_intent.size == down_intent.size == SETTINGS.paper_trade_size
    assert up_intent.reason == down_intent.reason == "matched_arb"
    assert up_intent.limit_price == 0.45


def test_matched_arb_none_when_combined_ask_over_one():
    pm = make_pm()
    asks_up = [(0.51, 100.0)]
    asks_down = [(0.51, 100.0)]
    assert pm._check_matched_arb("cond1", "tok_up", "tok_down", asks_up, asks_down) == []


def test_matched_arb_capped_by_max_notional():
    tight_settings = Settings(max_notional_per_market=5.0)
    pm = PositionManager(tight_settings)
    asks_up = [(0.45, 1000.0)]
    asks_down = [(0.45, 1000.0)]
    intents = pm._check_matched_arb("cond1", "tok_up", "tok_down", asks_up, asks_down)
    assert len(intents) == 2
    notional = intents[0].size * (intents[0].limit_price + intents[1].limit_price)
    assert notional <= tight_settings.max_notional_per_market + 1e-9
    assert intents[0].size < tight_settings.paper_trade_size  # actually got scaled down


def test_hedge_fulfillment_completes_a_cheap_other_leg():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    inv.up.size = 20.0
    inv.up.cost_basis = 20.0 * 0.4  # avg cost 0.4

    intent = pm._check_hedge_fulfillment("cond1", "tok_up", "tok_down", inv, [], [(0.5, 100.0)])

    assert intent is not None
    assert intent.outcome == "down"
    assert intent.reason == "temporal_hedge"
    assert intent.size == 20.0
    assert intent.limit_price == 0.5
    assert math.isclose(intent.edge, 1.0 - (0.4 + 0.5))


def test_hedge_fulfillment_none_when_other_leg_too_expensive():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    inv.up.size = 20.0
    inv.up.cost_basis = 20.0 * 0.9  # avg cost 0.9 -> max hedge price is negative

    intent = pm._check_hedge_fulfillment("cond1", "tok_up", "tok_down", inv, [], [(0.5, 100.0)])
    assert intent is None


def test_hedge_fulfillment_none_when_no_imbalance():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    intent = pm._check_hedge_fulfillment("cond1", "tok_up", "tok_down", inv, [(0.5, 100.0)], [(0.5, 100.0)])
    assert intent is None


def test_directional_kelly_sizes_via_kelly_formula():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    intent = pm._check_directional_kelly(
        "cond1", "tok_up", "up", fair_p=0.7, asks=[(0.5, 1000.0)],
        t_remaining=300.0, sigma=None, inv=inv,
    )
    assert intent is not None
    assert intent.reason == "directional_kelly"
    assert intent.limit_price == 0.5
    # f* = 0.4; notional = bankroll * kelly_fraction * f* = 1000*0.15*0.4 = 60 -> size 120,
    # then capped by paper_trade_size (20) before the notional cap even applies
    assert intent.size == SETTINGS.paper_trade_size


def test_directional_kelly_none_when_edge_below_threshold():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    intent = pm._check_directional_kelly(
        "cond1", "tok_up", "up", fair_p=0.51, asks=[(0.5, 1000.0)],
        t_remaining=300.0, sigma=None, inv=inv,
    )
    assert intent is None


def test_decide_trades_prioritizes_matched_arb_over_hedge_and_directional():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    inv.up.size = 20.0
    inv.up.cost_basis = 20.0 * 0.4  # existing imbalance that could also be hedged
    fair_value = FairValue(p_up=0.9, p_down=0.1, deviation=3.0)  # would also justify a big directional bet

    intents = pm.decide_trades(
        "cond1", "tok_up", "tok_down", fair_value,
        asks_up=[(0.45, 100.0)], asks_down=[(0.45, 100.0)],  # qualifies for matched arb
        t_remaining=300.0, sigma=None,
    )

    assert len(intents) == 2
    assert all(i.reason == "matched_arb" for i in intents)


def test_decide_trades_picks_stronger_directional_edge():
    pm = make_pm()
    # combined asks (1.10) rule out matched arb -- both legs still clear
    # min_edge_to_trade individually, "up" more strongly than "down"
    fair_value = FairValue(p_up=0.9, p_down=0.7, deviation=3.0)

    intents = pm.decide_trades(
        "cond1", "tok_up", "tok_down", fair_value,
        asks_up=[(0.55, 1000.0)],
        asks_down=[(0.55, 1000.0)],
        t_remaining=300.0, sigma=None,
    )

    assert len(intents) == 1
    assert intents[0].outcome == "up"
    assert intents[0].reason == "directional_kelly"


def test_decide_trades_blocks_new_directional_at_imbalance_cap():
    pm = make_pm()
    inv = pm.get_inventory("cond1")
    inv.up.size = SETTINGS.max_inventory_imbalance
    inv.up.cost_basis = SETTINGS.max_inventory_imbalance * 0.4
    fair_value = FairValue(p_up=0.9, p_down=0.1, deviation=3.0)

    intents = pm.decide_trades(
        "cond1", "tok_up", "tok_down", fair_value,
        asks_up=[(0.5, 1000.0)],
        asks_down=[(0.6, 1000.0)],  # combined 1.1 -> no matched arb; too rich to hedge with cost 0.4
        t_remaining=300.0, sigma=None,
    )
    assert intents == []


def test_compute_resolution_pnl_winning_side():
    pm = make_pm()
    pm.record_fill("cond1", "up", 0.5, 20.0, 0.2)
    pnl = pm.compute_resolution_pnl("cond1", "up")
    assert math.isclose(pnl, 20.0 - (10.0 + 0.2))


def test_compute_resolution_pnl_losing_side():
    pm = make_pm()
    pm.record_fill("cond1", "up", 0.5, 20.0, 0.2)
    pnl = pm.compute_resolution_pnl("cond1", "down")
    assert math.isclose(pnl, -(10.0 + 0.2))


def test_compute_resolution_pnl_none_for_unknown_market():
    pm = make_pm()
    assert pm.compute_resolution_pnl("nonexistent", "up") is None


def test_drop_market_removes_inventory():
    pm = make_pm()
    pm.record_fill("cond1", "up", 0.5, 10.0, 0.0)
    dropped = pm.drop_market("cond1")
    assert dropped is not None
    assert "cond1" not in pm.inventory
