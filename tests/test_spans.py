from jeson.spans import anls, best_run, clean, exact, text_of


def w(text, x0, x1, line):
    return [text, 1.0, [x0, 0, x1, 10], line]


def test_clean_and_exact():
    assert clean(" Ltd., ") == "Ltd."
    assert exact("Mehta  Traders LTD.", "mehta traders ltd.")
    assert anls("Mehta Traders", "Mehta Tradrs") > 0.9 and anls("abc", "xyz") == 0.0
    assert anls(None, None) == 1.0 and anls(None, "x") == 0.0


def test_best_run_finds_the_truth_inside_a_page():
    words = [w("Bill", 0, 20, 0), w("to:", 22, 40, 0), w("Mehta", 0, 40, 1), w("Traders", 42, 90, 1), w("Ltd.,", 92, 120, 1), w("Pune", 0, 30, 2)]
    i, j, sim = best_run(words, "Mehta Traders Ltd.")
    assert (i, j, sim) == (2, 4, 1.0)


def test_text_of_drops_a_label_beside_a_wrapped_value():
    # value on lines 0 and 2 (x 110-670), the label 'Terms:' (x 60-105) sits beside the value's second line
    words = [w("Goods", 110, 300, 0), w("once", 310, 670, 0), w("Terms:", 60, 105, 1), w("sold.", 110, 200, 2)]
    assert text_of(words, 0, 3) == "Goods once sold."
