from pathlib import Path

for p in Path("scoring/pillars").glob("p*.py"):
    with open(p, "r", encoding="utf-8") as f:
        code = f.read()

    if "joblib.load" in code and "_MODEL_CACHE" not in code:
        # Add cache definition
        cache_header = "_MODEL_CACHE = {}\n\ndef _get_cached_artifacts():\n    if not _MODEL_CACHE:\n        if MODEL_PATH.exists() and FEATS_PATH.exists():\n            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)\n            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)\n    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')\n"
        
        # Replace un-cached load inside score()
        old_load = """        try:
            model = joblib.load(MODEL_PATH)
            features = joblib.load(FEATS_PATH)"""
        
        new_load = """        try:
            model, features = _get_cached_artifacts()
            if model is None or features is None:
                return s"""

        code = code.replace("FEATS_PATH = ", cache_header + "\nFEATS_PATH = ")
        code = code.replace(old_load, new_load)

        with open(p, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Added in-memory caching to {p.name}")

print("All pillar models successfully upgraded to In-Memory Singleton Caching!")
