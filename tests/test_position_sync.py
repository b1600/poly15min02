from poly15m.execution.position_sync import summarize_positions

# Real shape observed live from https://data-api.polymarket.com/positions
SAMPLE_RESPONSE = [
    {
        "proxyWallet": "0x0000000000000000000000000000000000000001",
        "asset": "69119940120690198080427862456277899176151352667092917892731174838615809844766",
        "conditionId": "0xc587bda904f031a973ad3cb57128ca011bfab0f45e6cb3734ed2227c4d4be419",
        "size": 5000,
        "avgPrice": 0,
        "initialValue": 0,
        "currentValue": 0,
        "cashPnl": 0,
        "percentPnl": 0,
        "totalBought": 0,
        "realizedPnl": 0,
        "percentRealizedPnl": 0,
        "curPrice": 0,
        "redeemable": True,
        "mergeable": False,
        "title": "Will Oh Se-hoon win the 2026 Seoul Mayoral Election",
        "slug": "will-oh-se-hoon-win-the-2026-seoul-mayoral-election",
        "outcome": "No",
        "outcomeIndex": 1,
        "endDate": "2026-06-03",
        "negativeRisk": True,
    }
]


def test_summarize_positions_extracts_expected_fields():
    summary = summarize_positions(SAMPLE_RESPONSE)
    assert summary == [
        {
            "condition_id": "0xc587bda904f031a973ad3cb57128ca011bfab0f45e6cb3734ed2227c4d4be419",
            "token_id": "69119940120690198080427862456277899176151352667092917892731174838615809844766",
            "outcome": "No",
            "size": 5000,
            "avg_price": 0,
            "redeemable": True,
            "title": "Will Oh Se-hoon win the 2026 Seoul Mayoral Election",
        }
    ]


def test_summarize_positions_empty_list():
    assert summarize_positions([]) == []


def test_summarize_positions_tolerates_missing_fields():
    summary = summarize_positions([{"conditionId": "0xabc"}])
    assert summary[0]["condition_id"] == "0xabc"
    assert summary[0]["size"] is None
