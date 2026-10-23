import unittest

from prepare_rewritten_training import split_samples


class SplitTests(unittest.TestCase):
    def test_related_charts_and_shared_images_stay_together(self):
        samples = [{"id": str(i), "image": "images/%d.png" % i,
                    "provenance": [{"original_image_id": "original_%d.png" % i}]}
                   for i in range(40)]
        samples += [
            {"id": "edit_a", "image": "images/edit_a.png",
             "provenance": [{"original_image_id": "original_0.png"}]},
            {"id": "edit_b", "image": "images/edit_b.png",
             "provenance": [{"original_image_id": "original_0.png"},
                            {"original_image_id": "original_1.png"}]},
            {"id": "shared_image", "image": "images/edit_b.png",
             "provenance": [{"original_image_id": "original_2.png"}]},
        ]
        train, val = split_samples(samples, 0.25, 42)
        self.assertTrue(train and val)
        train_ids = {s["id"] for s in train}
        val_ids = {s["id"] for s in val}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train) + len(val), len(samples))
        related = {"0", "1", "2", "edit_a", "edit_b", "shared_image"}
        self.assertTrue(related <= train_ids or related <= val_ids)
        original_ids = lambda rows: {p["original_image_id"] for s in rows for p in s["provenance"]}
        self.assertFalse(original_ids(train) & original_ids(val))
        self.assertFalse({s["image"] for s in train} & {s["image"] for s in val})
        reversed_train, reversed_val = split_samples(list(reversed(samples)), 0.25, 42)
        self.assertEqual(train_ids, {s["id"] for s in reversed_train})
        self.assertEqual(val_ids, {s["id"] for s in reversed_val})


if __name__ == "__main__":
    unittest.main()
