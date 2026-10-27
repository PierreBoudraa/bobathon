try:
    from sklearn.dummy import DummyRegressor
    print("DummyRegressor: OK")
except ImportError as e:
    print("DummyRegressor: FAIL -", e)

try:
    from sklearn.linear_model import Ridge
    print("Ridge: OK")
except ImportError as e:
    print("Ridge: FAIL -", e)

try:
    from sklearn.model_selection import cross_val_score
    print("cross_val_score: OK")
except ImportError as e:
    print("cross_val_score: FAIL -", e)

try:
    import skore
    print("skore import: OK")
except ImportError as e:
    print("skore import: FAIL -", e)
