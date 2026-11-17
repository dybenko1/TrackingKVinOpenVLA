import unittest

from experiments.kv_cache.experiments.trackb_talk2dino_siglip.caption_round_sampler import CaptionRoundBatchSampler


class CaptionRoundBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        # Unequal annotation counts exercise later rounds and partial batches.
        self.image_ids = [10, 10, 10, 10, 20, 20, 30, 40, 40, 50, 50, 50, 60]

    def test_all_rows_once_and_no_repeated_image_in_any_batch(self):
        sampler = CaptionRoundBatchSampler(self.image_ids, batch_size=3, seed=9)
        rows = [row for batch in sampler for row in batch]
        self.assertEqual(sorted(rows), list(range(len(self.image_ids))))
        self.assertEqual(len(rows), len(set(rows)))
        for batch in sampler:
            batch_ids = [self.image_ids[row] for row in batch]
            self.assertEqual(len(batch_ids), len(set(batch_ids)))

    def test_partial_batches_are_preserved(self):
        sampler = CaptionRoundBatchSampler(self.image_ids, batch_size=4, seed=9)
        batches = list(sampler)
        self.assertTrue(any(len(batch) < 4 for batch in batches))
        self.assertEqual(sum(map(len, batches)), len(self.image_ids))

    def test_different_epochs_rotate_captions(self):
        sampler = CaptionRoundBatchSampler(self.image_ids, batch_size=3, seed=9)
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        # First-round row selected for at least one multi-caption image changes.
        first_row_to_image = {row: self.image_ids[row] for batch in first[:2] for row in batch}
        second_row_to_image = {row: self.image_ids[row] for batch in second[:2] for row in batch}
        first_selected = {image: min(row for row, item in first_row_to_image.items() if item == image)
                          for image in set(self.image_ids) if self.image_ids.count(image) > 1}
        second_selected = {image: min(row for row, item in second_row_to_image.items() if item == image)
                           for image in set(self.image_ids) if self.image_ids.count(image) > 1}
        # With the seeded permutations, one or more images must rotate.
        self.assertNotEqual(first_selected, second_selected)

    def test_same_seed_and_epoch_repeats_exactly(self):
        a = CaptionRoundBatchSampler(self.image_ids, batch_size=3, seed=17)
        b = CaptionRoundBatchSampler(self.image_ids, batch_size=3, seed=17)
        a.set_epoch(4)
        b.set_epoch(4)
        self.assertEqual(list(a), list(b))
        self.assertEqual(len(a), len(list(a)))

    def test_single_caption_images_and_small_partial_round(self):
        sampler = CaptionRoundBatchSampler([1, 1, 2, 3, 4], batch_size=3, seed=0)
        batches = list(sampler)
        self.assertEqual(sorted(row for batch in batches for row in batch), list(range(5)))
        self.assertEqual([len(batch) for batch in batches], [3, 1, 1])


if __name__ == "__main__":
    unittest.main()
