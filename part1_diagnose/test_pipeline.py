"""Offline checks of split isolation, epoch/label alignment and ERD arithmetic."""
import contextlib
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch, mock_open

import mne
import numpy as np

import data
from erd import erd_percent
from evaluation import split_subjects, standardize_from_train


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(1, 11), 4)

    def test_partitions_are_complete_and_subject_disjoint(self):
        for seed in [0, 1, 2]:
            splits = split_subjects(self.groups, seed)
            np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(40))
            subject_sets = [set(self.groups[ix]) for ix in splits]
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                self.assertFalse(subject_sets[a] & subject_sets[b])
            self.assertEqual(subject_sets[2], {9, 10})
            for a, b in zip(splits, split_subjects(self.groups, seed)):
                np.testing.assert_array_equal(a, b)

    def test_too_few_subjects_rejected(self):
        with self.assertRaises(ValueError):
            split_subjects(np.array([1, 1, 2, 2]))

    def test_held_out_perturbation_cannot_change_training_preprocessing(self):
        X = np.random.default_rng(0).normal(size=(40, 3, 20)).astype(np.float32)
        tr, va, te = split_subjects(self.groups)
        Z, mean, std = standardize_from_train(X, tr)
        changed = X.copy()
        changed[np.r_[va, te]] = 10000
        Z2, mean2, std2 = standardize_from_train(changed, tr)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)
        np.testing.assert_array_equal(Z[tr], Z2[tr])
        np.testing.assert_allclose(Z[tr].mean(axis=(0, 2)), 0, atol=1e-6)
        self.assertGreater(float(Z2[te].mean()), 1000)


class SignalTests(unittest.TestCase):
    def test_valid_crop_at_recording_end_is_not_dropped(self):
        info = mne.create_info(["C3"], 160, "eeg")
        raw = mne.io.RawArray(np.zeros((1, 30 * 160)), info, verbose="error")
        raw.set_annotations(mne.Annotations([3, 25], [4, 4], ["T1", "T2"]))
        events = data.get_events(raw)
        old = mne.Epochs(raw, events, data.EVENT_ID, -2, 5, baseline=None,
                         preload=True, verbose="error")
        self.assertEqual(len(old), 1)  # unused tail runs past the recording
        with patch.object(data, "load_raw", return_value=raw):
            X, y, _ = data.epoch_subject(1)
        self.assertEqual(len(X), 2)
        np.testing.assert_array_equal(y, [0, 1])

    def test_annotation_labels_stay_aligned_after_crop(self):
        info = mne.create_info(["C3", "Cz", "C4"], 160, "eeg")
        raw = mne.io.RawArray(
            np.random.default_rng(0).normal(0, 1e-6, (3, 160 * 30)),
            info, verbose="error")
        raw.set_annotations(mne.Annotations([3, 11, 19], [4, 4, 4], ["T2", "T0", "T1"]))
        with patch.object(data, "load_raw", return_value=raw):
            X, y, names = data.epoch_subject(1)
        self.assertEqual(X.shape, (2, 3, 481))
        np.testing.assert_array_equal(y, [1, 0])
        self.assertEqual(names, ["C3", "Cz", "C4"])
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(X.dtype, np.float32)

    def test_half_rest_power_is_negative_fifty_percent(self):
        info = mne.create_info(["C3"], 160, "eeg")
        times = np.arange(7 * 160 + 1) / 160 - 2
        power = np.where(times < 0, 10.0, 5.0)[None, None, :]
        ep = mne.EpochsArray(power, info, tmin=-2, events=np.array([[0, 0, 1]]),
                             event_id={"hands": 1}, baseline=None, verbose="error")
        erd, t = erd_percent(ep, "hands")
        np.testing.assert_allclose(erd[:, t < 0], 0)
        np.testing.assert_allclose(erd[:, t >= 0], -50)

    def test_zero_reference_power_rejected(self):
        info = mne.create_info(["C3"], 160, "eeg")
        ep = mne.EpochsArray(np.zeros((1, 1, 1121)), info, tmin=-2,
                             baseline=None, verbose="error")
        with self.assertRaisesRegex(ValueError, "positive"):
            erd_percent(ep, "1")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TrainingTests(unittest.TestCase):
    def test_validation_selects_checkpoint_and_test_is_opened_once(self):
        import torch
        import train

        X = np.random.default_rng(7).normal(size=(40, 2, 32)).astype(np.float32)
        y = np.tile([0, 1], 20).astype(np.int64)
        groups = np.repeat(np.arange(1, 11), 4)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))
        scores = iter([0.4, 0.8, 0.6, 0.5])
        states, calls = [], []

        def fake_evaluate(net, features, labels):
            states.append({k: v.clone() for k, v in net.state_dict().items()})
            calls.append(features.copy())
            return next(scores)

        with patch.object(train, "build_dataset", return_value=(X, y, groups, ["C3", "C4"])), \
             patch.object(train, "EEGNet", return_value=model), \
             patch.object(train, "evaluate", side_effect=fake_evaluate), \
             patch.object(train.torch, "save") as save_checkpoint, \
             patch.object(sys, "argv", ["train.py", "--epochs", "3"]), \
             patch("builtins.open", mock_open()), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        self.assertEqual(len(calls), 4)  # 3 validation calls, then one test call
        for key in states[-1]:
            torch.testing.assert_close(states[-1][key], states[1][key])
        save_checkpoint.assert_called_once()
        saved = save_checkpoint.call_args.args[0]
        for key in states[1]:
            torch.testing.assert_close(saved["state_dict"][key], states[1][key])
        tr, va, te = split_subjects(groups)
        Z, _, _ = standardize_from_train(X, tr)
        for observed in calls[:-1]:
            np.testing.assert_array_equal(observed, Z[va])
        np.testing.assert_array_equal(calls[-1], Z[te])


if __name__ == "__main__":
    unittest.main()
