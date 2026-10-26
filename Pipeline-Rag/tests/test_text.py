from rag.text import strip_accents, tokenize


def test_tokenize_removes_attached_punctuation():
    assert tokenize("El plazo es de 30 días, solicite el reembolso.") == [
        "plazo", "30", "dias", "solicite", "reembolso",
    ]


def test_tokenize_is_accent_and_case_insensitive():
    assert tokenize("GARANTÍA") == tokenize("garantia") == ["garantia"]


def test_tokenize_keeps_full_numbers():
    # Las cifras exactas son la señal léxica más útil en documentos financieros.
    assert "4,5" in tokenize("La tasa sube de 4,0% a 4,5% anual")
    assert "10.000" in tokenize("saldos superiores a Bs 10.000.")
    assert "0,05" in tokenize("recargo de 0,05% diario")


def test_tokenize_drops_stopwords_but_can_keep_them():
    assert "de" not in tokenize("plazo de reembolso")
    assert "de" in tokenize("plazo de reembolso", remove_stopwords=False)


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize("   ...  ") == []


def test_strip_accents_handles_ene():
    assert strip_accents("año") == "ano"
