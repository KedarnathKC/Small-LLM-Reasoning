from torch.utils.data import Sampler
import random
import math

class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, threshold_column, threshold, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold_column = threshold_column
        self.threshold = threshold

        # Convert column to list for easier indexing
        column_values = dataset[self.threshold_column]

        self.above_indices = [i for i, v in enumerate(column_values) if v >= threshold]
        self.below_indices = [i for i, v in enumerate(column_values) if v < threshold]

        assert len(self.above_indices) > 0 and len(self.below_indices) > 0, "Both sides of the threshold must be represented."

    def __iter__(self):
        # Shuffle at the start of each epoch
        random.shuffle(self.above_indices)
        random.shuffle(self.below_indices)

        used_above = set()
        used_below = set()

        above_pool = self.above_indices.copy()
        below_pool = self.below_indices.copy()

        while above_pool:
            num_above = max(1, self.batch_size // 10)
            num_below = self.batch_size - num_above

            if len(above_pool) < num_above:
                num_above = len(above_pool)
                num_below = self.batch_size - num_above

            batch_above = [above_pool.pop() for _ in range(num_above)]
            used_above.update(batch_above)

            # If not enough below left, refill from unused ones
            if len(below_pool) < num_below:
                refill = list(set(self.below_indices) - used_below)
                if not refill:  # If all used, reset
                    refill = self.below_indices.copy()
                random.shuffle(refill)
                below_pool.extend(refill)

            batch_below = [below_pool.pop() for _ in range(num_below)]
            used_below.update(batch_below)

            yield batch_below + batch_above

    def __len__(self):
        return math.ceil(len(self.above_indices) / max(1, self.batch_size // 10))
