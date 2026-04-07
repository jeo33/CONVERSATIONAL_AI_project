from datasets import load_dataset
from api import H2OSession

print("Loading dataset...")
ds = load_dataset("cnn_dailymail", "3.0.0")
sample = ds["validation"][0]
article   = sample["article"]
reference = sample["highlights"]

print(f"Article length : {len(article)} chars")
print(f"Reference      : {reference[:120]}\n")

sess = H2OSession()

results = sess.compare_modes(
    text=article,
    reference=reference,
    modes=["full", "random_b4", "h2o_b4_r2", "h2o_b4_r1"],
    seed=0,
)

sess.cleanup()
