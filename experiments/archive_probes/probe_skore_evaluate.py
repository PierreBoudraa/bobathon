import skore
print("skore imported OK")

try:
    from skore import evaluate
    print("skore.evaluate imported OK")
except ImportError as e:
    print("skore.evaluate FAIL:", e)
