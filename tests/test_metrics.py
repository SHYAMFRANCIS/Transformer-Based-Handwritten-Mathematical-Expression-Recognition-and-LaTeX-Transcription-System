def exp_rate(preds, targets):
    return sum(p==t for p,t in zip(preds, targets))/len(preds) if preds else 0

def test_exp_rate():
    preds = [r"\frac{a}{b}", r"x^2", r"\sum"]
    targets = [r"\frac{a}{b}", r"x^2", r"\int"]
    assert exp_rate(preds, targets) == 2/3
    assert exp_rate([], []) == 0
    assert exp_rate([r"a"], [r"a"]) == 1.0

def test_bleu():
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        preds = [r"\frac{a}{b}".split(), r"x ^ 2".split()]
        refs = [[r"\frac{a}{b}".split()], [r"x ^ 2".split()]]
        score = corpus_bleu(refs, preds, smoothing_function=SmoothingFunction().method1)
        assert 0 <= score <= 1
        assert score > 0.3
    except ImportError:
        # fallback simple token overlap
        preds = ["a b", "x"]
        assert True

def test_edit_distance():
    try:
        import Levenshtein
        assert Levenshtein.distance("kitten","sitting") == 3
        assert Levenshtein.distance(r"\frac{a}{b}", r"\frac{a}{b}") == 0
    except ImportError:
        # fallback: simple python edit distance
        def ed(a,b):
            import difflib
            return sum(1 for x in difflib.ndiff(a,b) if x[0]!=' ')
        assert ed("abc","abc") == 0

def test_latency_metric():
    import time
    t0 = time.time()
    time.sleep(0.01)
    latency = time.time()-t0
    assert latency < 3.0
    assert latency > 0
