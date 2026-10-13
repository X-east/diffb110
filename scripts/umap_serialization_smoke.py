from pathlib import Path

import joblib
import numpy as np
import umap
from scipy import sparse


output = Path('/public/home/fzmn1/diffb110_full_umap_20260818/output')
x = sparse.random(5000, 128, density=0.08, format='csr', random_state=1).toarray().astype(np.float32)
model = umap.UMAP(metric='cosine', n_jobs=4, random_state=None).fit(x)
path = output / 'serialization_smoke.joblib'
joblib.dump(model, path)
loaded = joblib.load(path)
projected = loaded.transform(x[:10])
(output / 'serialization_smoke.ok').write_text(
    f'shape={projected.shape} finite={bool(np.isfinite(projected).all())}\n',
    encoding='utf-8',
)
